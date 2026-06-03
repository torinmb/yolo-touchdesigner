// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import {
    INPUT_W,
    INPUT_H,
    PERSON_SEG_ONLY,
    SEG_DECAY_LIMIT,
} from "../config.js";
import { iou } from "../utils/math.js";

function decodeYOLO_or_OBB(tensor, thr, topk, isObb, W = INPUT_W, H = INPUT_H) {
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

    const out = [];

    // Indices
    // Standard YOLO OBB (v8/v11) Layout: [cx, cy, w, h, class0, class1, ..., classN, angle]
    // Angle is the LAST channel.
    const ANGLE_IDX = isObb ? dim - 1 : -1;
    const OBJ_IDX = -1;
    const CLS_START = 4;
    const CLS_END = isObb ? dim - 1 : dim;

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
        for (let c = CLS_START; c < CLS_END; c++) {
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

export function decodeYOLOv26OBB(
    tensor,
    score_threshold,
    topk,
    W = INPUT_W,
    H = INPUT_H,
) {
    const data = tensor.data;
    const dims = tensor.dims;
    // Expect [1, 300, 7]
    // Layout: cx, cy, w, h, angle, score, cls

    if (dims.length !== 3 || dims[2] !== 7) return [];

    const N = dims[1];
    const stride = 7;
    const out = [];

    for (let i = 0; i < N; i++) {
        const off = i * stride;

        // Corrected for v26/DETR OBB layout:
        // [cx, cy, w, h, score, cls, angle]
        const score = data[off + 4]; // Index 4 is Score
        if (score < score_threshold) continue;

        const cx = data[off + 0];
        const cy = data[off + 1];
        const w = data[off + 2];
        const h = data[off + 3];
        const cls = data[off + 5]; // Index 5 is Class
        const angle = data[off + 6]; // Index 6 is Angle

        let bx = cx - 0.5 * w;
        let by = cy - 0.5 * h;
        let bw = w;
        let bh = h;

        // Clamp to image bounds to match generic decoder
        bx = Math.max(0, Math.min(W - 1, bx));
        by = Math.max(0, Math.min(H - 1, by));
        bw = Math.max(1, Math.min(W - bx, bw));
        bh = Math.max(1, Math.min(H - by, bh));

        out.push({
            box: [bx, by, bw, bh],
            label: cls,
            score: score,
            angle: angle,
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

let segPipeline = null;
let segProtoBuffer = null;
let segDetsBuffer = null;
let segOutBuffer = null;
let segReadBuffer = null;
let segUniformBuffer = null;
let segBindGroup = null;
let segBufferSize = { MW: 0, MH: 0, MaskC: 0 };

function initSegPipeline(device, MW, MH, MaskC) {
    if (
        segPipeline &&
        segBufferSize.MW === MW &&
        segBufferSize.MH === MH &&
        segBufferSize.MaskC === MaskC
    ) {
        return;
    }

    if (segProtoBuffer) segProtoBuffer.destroy();
    if (segDetsBuffer) segDetsBuffer.destroy();
    if (segOutBuffer) segOutBuffer.destroy();
    if (segReadBuffer) segReadBuffer.destroy();
    if (segUniformBuffer) segUniformBuffer.destroy();

    segBufferSize = { MW, MH, MaskC };

    const protoSize = Math.ceil((MaskC * MW * MH * 4) / 16) * 16;
    segProtoBuffer = device.createBuffer({
        size: protoSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const maxDets = 100; // Hardcoded max objects
    const detStructFloats = 8 + MaskC;
    const detsSize = Math.ceil(((4 + maxDets * detStructFloats) * 4) / 16) * 16;
    segDetsBuffer = device.createBuffer({
        size: detsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const outSize = Math.ceil((MW * MH * 4) / 16) * 16;
    segOutBuffer = device.createBuffer({
        size: outSize,
        usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
    });

    segReadBuffer = device.createBuffer({
        size: outSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    segUniformBuffer = device.createBuffer({
        size: Math.ceil((8 * 4) / 16) * 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const shaderCode = `
        struct Det {
            box : vec4<f32>,
            clsID : f32,
            pad1 : f32,
            pad2 : f32,
            pad3 : f32,
            coeffs : array<f32, ${MaskC}>
        };

        struct Dets {
            count : u32,
            pad1 : u32,
            pad2 : u32,
            pad3 : u32,
            items : array<Det>
        };

        struct Params {
            MW : u32,
            MH : u32,
            MaskC : u32,
            pad : u32,
            scaleX : f32,
            scaleY : f32,
        };

        @group(0) @binding(0) var<storage, read> protoData : array<f32>;
        @group(0) @binding(1) var<storage, read> inDets : Dets;
        @group(0) @binding(2) var<storage, read_write> outMap : array<f32>;
        @group(0) @binding(3) var<uniform> params : Params;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let x = global_id.x;
            let y = global_id.y;
            let MW = params.MW;
            let MH = params.MH;
            let MaskC = params.MaskC;

            if (x >= MW || y >= MH) {
                return;
            }

            let i = y * MW + x;
            
            var currentVal : f32 = 0.0;
            var currentID_coded : f32 = 0.0;
            var currentAlpha : f32 = 0.0;

            for (var idx = 0u; idx < inDets.count; idx++) {
                let det = inDets.items[idx];
                let bx1 = max(0.0, floor(det.box[0] / params.scaleX));
                let by1 = max(0.0, floor(det.box[1] / params.scaleY));
                let bx2 = min(f32(MW), ceil(det.box[2] / params.scaleX));
                let bx2_safe = min(f32(MW), bx2); // Just in case
                let by2 = min(f32(MH), ceil(det.box[3] / params.scaleY));

                let fx = f32(x);
                let fy = f32(y);

                if (fx >= bx1 && fx < bx2 && fy >= by1 && fy < by2) {
                    var sum : f32 = 0.0;
                    for (var c = 0u; c < MaskC; c++) {
                        sum += det.coeffs[c] * protoData[c * MW * MH + i];
                    }
                    
                    let alpha = 1.0 / (1.0 + exp(-sum));
                    // Smooth remap to eliminate the bounding box noise floor
                    let cleanAlpha = max(0.0, (alpha - 0.01) / 0.99);
                    let incomingAlphaVal = cleanAlpha * 0.99;

                    if (incomingAlphaVal > currentAlpha) {
                        currentVal = det.clsID + 1.0 + incomingAlphaVal;
                        currentID_coded = floor(currentVal);
                        currentAlpha = incomingAlphaVal;
                    } else if (currentID_coded > 0.0) {
                        currentAlpha *= (1.0 - alpha);
                        currentVal = currentID_coded + currentAlpha;
                    }
                }
            }
            
            outMap[i] = currentVal;
        }
    `;

    segPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: device.createShaderModule({ code: shaderCode }),
            entryPoint: "main",
        },
    });

    segBindGroup = device.createBindGroup({
        layout: segPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: segProtoBuffer } },
            { binding: 1, resource: { buffer: segDetsBuffer } },
            { binding: 2, resource: { buffer: segOutBuffer } },
            { binding: 3, resource: { buffer: segUniformBuffer } },
        ],
    });
}

let lastValidSegResult = null;
let segMissFrames = 0;

export async function decodeYOLOSeg(
    outs,
    outputNames,
    scoreThr,
    topk,
    device,
    inputW = INPUT_W,
    inputH = INPUT_H,
) {
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
                const label = dData[off + 5];
                if (PERSON_SEG_ONLY && label !== 0) continue;

                const x1 = dData[off + 0];
                const y1 = dData[off + 1];
                const x2 = dData[off + 2];
                const y2 = dData[off + 3];

                dets.push({
                    box: [x1, y1, x2 - x1, y2 - y1],
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

    if (dets.length === 0) {
        if (
            SEG_DECAY_LIMIT > 0 &&
            lastValidSegResult &&
            segMissFrames < SEG_DECAY_LIMIT
        ) {
            segMissFrames++;
            const len = lastValidSegResult.data.length;
            const src = lastValidSegResult.data;
            const decayedData = new Float32Array(len);
            for (let i = 0; i < len; i++) {
                const val = src[i];
                if (val > 0.0) {
                    const id = val | 0;
                    decayedData[i] = id + (val - id) * 0.7;
                }
            }
            lastValidSegResult = { ...lastValidSegResult, data: decayedData };
            return lastValidSegResult;
        }
        lastValidSegResult = null;
        segMissFrames = 0;
        return { width: MW, height: MH, data: new Float32Array(MW * MH) };
    }

    segMissFrames = 0;

    // --- APPLY NMS (Cleanup overlapping duplicates) ---
    // This fixes the "fighting layers" by removing duplicate boxes
    // before we even start drawing.
    // Using 0.45 IoU threshold (standard YOLO value)
    dets = nmsPerClass(dets, 0.45, topk);

    // Sort by area descending (Painter's Algorithm approximation)
    // Large objects (background) first, Small objects (foreground) last.
    dets.sort((a, b) => b.box[2] * b.box[3] - a.box[2] * a.box[3]);

    const scaleX = inputW / MW;
    const scaleY = inputH / MH;

    if (!device) {
        // Fallback to CPU calculation if WebGPU is unavailable
        const outMap = new Float32Array(MW * MH);
        const MH_MW = MH * MW;

        for (let k = 0; k < dets.length; k++) {
            const clsID = dets[k].label;
            const coeffs = dets[k].coeffs;
            const box = dets[k].box;

            const pad = 0;
            const mx1 = Math.max(0, Math.floor(box[0] / scaleX) - pad);
            const my1 = Math.max(0, Math.floor(box[1] / scaleY) - 0);
            const mx2 = Math.min(
                MW,
                Math.ceil((box[0] + box[2]) / scaleX) + pad,
            );
            const my2 = Math.min(
                MH,
                Math.ceil((box[1] + box[3]) / scaleY) + pad,
            );

            for (let y = my1; y < my2; y++) {
                for (let x = mx1; x < mx2; x++) {
                    const i = y * MW + x;

                    let sum = 0;
                    for (let c = 0; c < MaskC; c++) {
                        sum += coeffs[c] * protoData[c * MH_MW + i];
                    }

                    const alpha = 1.0 / (1.0 + Math.exp(-sum));
                    // Smooth remap to eliminate the bounding box noise floor
                    const cleanAlpha = Math.max(0.0, (alpha - 0.01) / 0.99);

                    const currentVal = outMap[i];
                    const currentID_coded = Math.floor(currentVal);
                    let currentAlpha = currentVal - currentID_coded;

                    const incomingAlphaVal = cleanAlpha * 0.99;

                    if (incomingAlphaVal > currentAlpha) {
                        outMap[i] = clsID + 1.0 + incomingAlphaVal;
                    } else if (currentID_coded > 0) {
                        currentAlpha *= 1.0 - alpha;
                        outMap[i] = currentID_coded + currentAlpha;
                    }
                }
            }
        }
        const result = { width: MW, height: MH, data: outMap };
        lastValidSegResult = result;
        return result;
    }

    // --- WebGPU Compute Acceleration ---
    initSegPipeline(device, MW, MH, MaskC);

    device.queue.writeBuffer(segProtoBuffer, 0, protoData);

    const maxDets = 100;
    const detFloats = 8 + MaskC;
    const mappedCount = Math.min(dets.length, maxDets);
    const detsData = new Float32Array(4 + maxDets * detFloats);

    new Uint32Array(detsData.buffer, 0, 16)[0] = mappedCount;

    for (let k = 0; k < mappedCount; k++) {
        const offset = 4 + k * detFloats;
        const d = dets[k];
        detsData[offset + 0] = d.box[0];
        detsData[offset + 1] = d.box[1];
        detsData[offset + 2] = d.box[0] + d.box[2]; // convert W back to X2 for the shader bounds check
        detsData[offset + 3] = d.box[1] + d.box[3]; // convert H back to Y2 for the shader bounds check
        detsData[offset + 4] = d.label;
        detsData.set(d.coeffs, offset + 8);
    }
    device.queue.writeBuffer(segDetsBuffer, 0, detsData);

    const paramsData = new Uint32Array(8);
    paramsData[0] = MW;
    paramsData[1] = MH;
    paramsData[2] = MaskC;
    paramsData[3] = 0;
    const paramsF32 = new Float32Array(paramsData.buffer);
    paramsF32[4] = scaleX;
    paramsF32[5] = scaleY;
    device.queue.writeBuffer(segUniformBuffer, 0, paramsData);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(segPipeline);
    passEncoder.setBindGroup(0, segBindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(MW / 16), Math.ceil(MH / 16));
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(
        segOutBuffer,
        0,
        segReadBuffer,
        0,
        MW * MH * 4,
    );
    device.queue.submit([commandEncoder.finish()]);

    await segReadBuffer.mapAsync(GPUMapMode.READ);
    const gpuData = new Float32Array(segReadBuffer.getMappedRange());
    // Create copy so we can unmap
    const outMap = new Float32Array(gpuData);
    segReadBuffer.unmap();

    const result = { width: MW, height: MH, data: outMap };
    lastValidSegResult = result;
    return result;
}

export { decodeYOLO_or_OBB };
