// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import * as ort from "onnxruntime-web/webgpu";
import { INPUT_W, INPUT_H, FLIP_HORIZONTAL } from "../config.js";

const NUM = 3 * INPUT_H * INPUT_W;
export const f32InputBuffer = new Float32Array(NUM);
export const inputTensor = new ort.Tensor("float32", f32InputBuffer, [
    1,
    3,
    INPUT_H,
    INPUT_W,
]);

// Debug Canvas
let _dbgCtx = null;
if (typeof document !== "undefined") {
    const DEBUG_RAW = new URLSearchParams(location.search).get("debug");
    if (DEBUG_RAW) {
        const c = document.createElement("canvas");
        c.id = "debug-canvas";
        c.width = INPUT_W;
        c.height = INPUT_H;
        document.body.appendChild(c);
        _dbgCtx = c.getContext("2d");
    }
}

function _debugDrawCHW_fromU8CHW(payloadU8, H = INPUT_H, W = INPUT_W) {
    if (!_dbgCtx) return;
    const plane = W * H;
    const out = new Uint8ClampedArray(4 * plane);
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const idx = y * W + x; // Normal orientation for debug
            const p = idx * 4;
            const r = payloadU8[0 * plane + idx];
            const g = payloadU8[1 * plane + idx];
            const b = payloadU8[2 * plane + idx];
            out[p + 0] = r;
            out[p + 1] = g;
            out[p + 2] = b;
            out[p + 3] = 255;
        }
    }
    _dbgCtx.putImageData(new ImageData(out, W, H), 0, 0);
}

export function toInputTensorFromU8CHW(payload, H = INPUT_H, W = INPUT_W) {
    // payload is U8 CHW [R...][G...][B...]
    for (let i = 0; i < NUM; i++) f32InputBuffer[i] = payload[i] * (1 / 255);
    if (_dbgCtx) _debugDrawCHW_fromU8CHW(payload, H, W);
    return inputTensor;
}

export function toInputTensorFromImageData(imgData, flipH = false) {
    const src = imgData.data;
    const W = imgData.width,
        H = imgData.height;
    const plane = W * H;

    if (!flipH) {
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                const p = y * W + x;
                const s = p * 4;
                f32InputBuffer[0 * plane + p] = src[s] / 255;
                f32InputBuffer[1 * plane + p] = src[s + 1] / 255;
                f32InputBuffer[2 * plane + p] = src[s + 2] / 255;
            }
        }
    } else {
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                const x2 = W - 1 - x;
                const p = y * W + x;
                const s = (y * W + x2) * 4;
                f32InputBuffer[0 * plane + p] = src[s] / 255;
                f32InputBuffer[1 * plane + p] = src[s + 1] / 255;
                f32InputBuffer[2 * plane + p] = src[s + 2] / 255;
            }
        }
    }
    return inputTensor;
}
