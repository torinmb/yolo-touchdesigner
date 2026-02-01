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

export { decodeYOLO_or_OBB };
