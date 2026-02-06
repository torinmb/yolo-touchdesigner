// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import { INPUT_W, INPUT_H } from "../config.js";
import { iou } from "../utils/math.js";

function decodeYOLO_or_OBB(tensor, thr, topk, isObb) {
    const d = tensor.data;
    const sh = tensor.dims;
    if (sh.length !== 3) return [];

    const C0 = sh[1],
        N0 = sh[2];
    let dim, num, channelsFirst;
    if (C0 < N0) {
        // Typical Ultralytics: [1, C, N]
        dim = C0;
        num = N0;
        channelsFirst = true;
    } else {
        // Fallback: [1, N, C]
        dim = N0;
        num = C0;
        channelsFirst = false;
    }

    const get = channelsFirst
        ? (c, i) => d[c * num + i]
        : (c, i) => d[i * dim + c];

    const W = INPUT_W,
        H = INPUT_H;
    const out = [];

    // Indices
    const ANGLE_IDX = isObb ? 4 : -1;
    const OBJ_IDX = isObb ? 5 : -1;
    const CLS_START = isObb ? 6 : 4;

    for (let i = 0; i < num; i++) {
        const cx = get(0, i);
        const cy = get(1, i);
        const w = get(2, i);
        const h = get(3, i);

        // Confidence
        const obj = OBJ_IDX >= 0 && OBJ_IDX < dim ? get(OBJ_IDX, i) : 1.0;

        // Best class
        let bestC = -1,
            bestS = -1;
        for (let c = CLS_START; c < dim; c++) {
            const s = get(c, i) * obj;
            if (s > bestS) {
                bestS = s;
                bestC = c - CLS_START;
            }
        }
        if (bestS < thr) continue;

        // Build AABB
        let bx, by, bw, bh;
        if (w > 0 && h > 0 && w <= W * 2 && h <= H * 2) {
            bw = w;
            bh = h;
            bx = cx - w / 2;
            by = cy - h / 2;
        } else {
            const x2 = w,
                y2 = h;
            bx = cx;
            by = cy;
            bw = x2 - cx;
            bh = y2 - cy;
        }

        // Clamp
        bx = Math.max(0, Math.min(W - 1, bx));
        by = Math.max(0, Math.min(H - 1, by));
        bw = Math.max(1, Math.min(W - bx, bw));
        bh = Math.max(1, Math.min(H - by, bh));

        const det = { box: [bx, by, bw, bh], label: bestC, score: bestS };

        // Optional angle
        if (ANGLE_IDX >= 0 && ANGLE_IDX < dim) {
            const theta = get(ANGLE_IDX, i);
            if (Number.isFinite(theta)) det.angle = theta;
        }

        out.push(det);
    }

    out.sort((a, b) => b.score - a.score);
    if (out.length > topk) out.length = topk;
    return out;
}

export function decodeYOLOPose(
    raw_tensor,
    score_threshold = 0.45,
    topk,
    W = INPUT_W,
    H = INPUT_H,
) {
    const sh = raw_tensor.dims;
    const data = raw_tensor.data;
    if (sh.length !== 3) return [];
    let C = sh[1],
        N = sh[2],
        layout = "CN";
    const looksPose = (c) => c > 5 && (c - 5) % 3 === 0;
    if (!looksPose(C)) {
        C = sh[2];
        N = sh[1];
        if (!looksPose(C)) return [];
        layout = "NC";
    }
    const K = ((C - 5) / 3) | 0;
    const get = (c, n) => (layout === "CN" ? data[c * N + n] : data[n * C + c]);
    const out = [];
    for (let i = 0; i < N; i++) {
        const score = get(4, i);
        if (score <= score_threshold) continue;
        const cx = get(0, i),
            cy = get(1, i),
            w = get(2, i),
            h = get(3, i);
        const bx = cx - 0.5 * w;
        const by = cy - 0.5 * h;
        const keypoints = new Array(K);
        for (let kp = 0; kp < K; kp++) {
            const base = 5 + kp * 3;
            keypoints[kp] = {
                x: get(base + 0, i),
                y: get(base + 1, i),
                score: get(base + 2, i),
            };
        }
        out.push({ box: [bx, by, w, h], label: 0, score, keypoints });
    }
    out.sort((a, b) => b.score - a.score);
    if (out.length > topk) out.length = topk;
    return out;
}

export function nmsPerClass(dets, iouThr, topk) {
    const byClass = new Map();
    for (const d of dets) {
        if (!byClass.has(d.label)) byClass.set(d.label, []);
        byClass.get(d.label).push(d);
    }
    const keepAll = [];
    for (const arr of byClass.values()) {
        arr.sort((a, b) => b.score - a.score);
        const keep = [];
        for (const a of arr) {
            let ok = true;
            for (const k of keep) {
                if (iou(a.box, k.box) > iouThr) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                keep.push(a);
                if (keep.length >= topk) break;
            }
        }
        keepAll.push(...keep);
    }
    keepAll.sort((a, b) => b.score - a.score);
    if (keepAll.length > topk) keepAll.length = topk;
    return keepAll;
}

export function decodeYOLOv26(tensor, score_threshold, topk) {
    const data = tensor.data;
    const dims = tensor.dims;
    // Expect [1, 300, 6] or [1, N, 6]
    // Layout: x1, y1, x2, y2, score, cls

    if (dims.length !== 3 || dims[2] !== 6) return [];

    const N = dims[1];
    const stride = 6;
    const out = [];

    for (let i = 0; i < N; i++) {
        const off = i * stride;
        const score = data[off + 4];
        if (score < score_threshold) continue;

        const x1 = data[off + 0];
        const y1 = data[off + 1];
        const x2 = data[off + 2];
        const y2 = data[off + 3];
        const cls = data[off + 5];

        // Convert xyxy -> xywh
        const w = x2 - x1;
        const h = y2 - y1;
        out.push({
            box: [x1, y1, w, h],
            label: cls,
            score: score,
        });
    }

    out.sort((a, b) => b.score - a.score);
    if (out.length > topk) out.length = topk;
    return out;
}

export function decodeYOLOv26Pose(tensor, score_threshold, topk) {
    const data = tensor.data;
    const dims = tensor.dims;
    // Expect [1, 300, 57]
    // Layout: x1, y1, x2, y2, score, cls, [x, y, v] * 17

    // Check rudimentary validity
    if (dims.length !== 3) return [];
    const C = dims[2];
    // 4 box + 1 score + 1 class = 6.
    // Rest are keypoints? (C - 6) / 3

    const N = dims[1];
    const stride = C;
    const numKpt = Math.floor((C - 6) / 3);
    const out = [];

    for (let i = 0; i < N; i++) {
        const off = i * stride;
        const score = data[off + 4];
        if (score < score_threshold) continue;

        const x1 = data[off + 0];
        const y1 = data[off + 1];
        const x2 = data[off + 2];
        const y2 = data[off + 3];
        const cls = data[off + 5];

        // Keypoints start at offset 6
        const keypoints = [];
        for (let k = 0; k < numKpt; k++) {
            const base = off + 6 + k * 3;
            keypoints.push({
                x: data[base + 0],
                y: data[base + 1],
                score: data[base + 2],
            });
        }

        const w = x2 - x1;
        const h = y2 - y1;

        out.push({
            box: [x1, y1, w, h],
            label: cls,
            score: score,
            keypoints: keypoints,
        });
    }

    out.sort((a, b) => b.score - a.score);
    if (out.length > topk) out.length = topk;
    return out;
}

export function decodeYOLOSeg(outs, outputNames, scoreThr, topk) {
    let detT = null,
        protoT = null;
    for (const name of outputNames) {
        const t = outs[name];
        if (t.dims.length === 4) protoT = t;
        else if (t.dims.length === 3) detT = t;
    }
    if (!detT || !protoT) return null;

    const pDims = protoT.dims; // [1, 32, MH, MW]
    const MH = pDims[2],
        MW = pDims[3],
        MaskC = pDims[1];
    const protoData = protoT.data;

    let dets = [];
    // We no longer keep maskCoeffs separate, we attach them to the 'det' object
    // so they survive the sorting and NMS process together.

    const dDims = detT.dims;
    const dData = detT.data;

    if (dDims.length === 3 && dDims[1] === 300 && dDims[2] >= 6 + MaskC) {
        const stride = dDims[2];
        for (let i = 0; i < 300; i++) {
            const off = i * stride;
            const score = dData[off + 4];
            if (score > scoreThr) {
                const x1 = dData[off + 0];
                const y1 = dData[off + 1];
                const x2 = dData[off + 2];
                const y2 = dData[off + 3];

                dets.push({
                    box: [x1, y1, x2, y2],
                    label: dData[off + 5],
                    score,
                    // Store coefficients directly on the object to keep them linked
                    coeffs: dData.subarray(off + 6, off + 6 + MaskC),
                });
            }
        }
    } else {
        return null;
    }

    if (dets.length === 0)
        return { width: MW, height: MH, data: new Float32Array(MW * MH) };

    // --- APPLY NMS (Cleanup overlapping duplicates) ---
    // This fixes the "fighting layers" by removing duplicate boxes
    // before we even start drawing.
    // Using 0.45 IoU threshold (standard YOLO value)
    dets = nmsPerClass(dets, 0.45, topk);

    const outMap = new Float32Array(MW * MH);
    const MH_MW = MH * MW;

    const scaleX = INPUT_W / MW;
    const scaleY = INPUT_H / MH;

    // Iterate objects (Now clean and unique!)
    for (let k = 0; k < dets.length; k++) {
        const clsID = dets[k].label;
        const coeffs = dets[k].coeffs; // Retrieve attached coeffs
        const box = dets[k].box;

        const mx1 = Math.max(0, Math.floor(box[0] / scaleX));
        const my1 = Math.max(0, Math.floor(box[1] / scaleY));
        const mx2 = Math.min(MW, Math.ceil(box[2] / scaleX));
        const my2 = Math.min(MH, Math.ceil(box[3] / scaleY));

        for (let y = my1; y < my2; y++) {
            for (let x = mx1; x < mx2; x++) {
                const i = y * MW + x;

                let sum = 0;
                for (let c = 0; c < MaskC; c++) {
                    sum += coeffs[c] * protoData[c * MH_MW + i];
                }

                // Smooth Sigmoid (0.0 to 1.0)
                const alpha = 1.0 / (1.0 + Math.exp(-sum));

                // PACKING: MAX COMPOSITING
                const currentVal = outMap[i];
                const currentAlpha = currentVal - Math.floor(currentVal);

                if (alpha * 0.999 > currentAlpha) {
                    outMap[i] = clsID + 1.0 + alpha * 0.99;
                }
            }
        }
    }

    return { width: MW, height: MH, data: outMap };
}

export { decodeYOLO_or_OBB };
