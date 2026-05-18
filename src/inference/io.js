// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import * as ort from "onnxruntime-web/webgpu";
import { INPUT_W, INPUT_H, FLIP_HORIZONTAL } from "../config.js";
import { device } from "./onnx.js";

const NUM = 3 * INPUT_H * INPUT_W;
export const f32InputBuffer = new Float32Array(NUM);
export let inputTensor = new ort.Tensor("float32", f32InputBuffer, [
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

let u8GpuBuffer = null;
let f32GpuBuffer = null;
let computePipeline = null;
let computeBindGroup = null;
let currentGpuW = 0;
let currentGpuH = 0;

function initGpuIoIfNeeded(H = INPUT_H, W = INPUT_W) {
    if (!device) return false;

    // If dimensions match existing buffer, we are good to go
    if (u8GpuBuffer && currentGpuH === H && currentGpuW === W) return true;

    // Destroy old buffers if dimensions changed
    if (u8GpuBuffer) u8GpuBuffer.destroy();
    if (f32GpuBuffer) f32GpuBuffer.destroy();

    currentGpuH = H;
    currentGpuW = W;

    const numBytes = 3 * H * W;
    const numFloats = 3 * H * W;
    const totalU32 = Math.ceil(numBytes / 4);

    // WebGPU storage buffers must be 4-byte aligned
    u8GpuBuffer = device.createBuffer({
        size: totalU32 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    f32GpuBuffer = device.createBuffer({
        size: numFloats * 4,
        usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
    });

    const shaderCode = `
        @group(0) @binding(0) var<storage, read> u8_data: array<u32>;
        @group(0) @binding(1) var<storage, read_write> f32_data: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            let total_u32 = ${totalU32}u;
            if (idx >= total_u32) { return; }

            let packed = u8_data[idx];
            
            let b0 = f32(packed & 0xFFu) / 255.0;
            let b1 = f32((packed >> 8u) & 0xFFu) / 255.0;
            let b2 = f32((packed >> 16u) & 0xFFu) / 255.0;
            let b3 = f32((packed >> 24u) & 0xFFu) / 255.0;

            let out_idx = idx * 4u;
            let total_f32 = ${numFloats}u;
            
            if (out_idx < total_f32) { f32_data[out_idx] = b0; }
            if (out_idx + 1u < total_f32) { f32_data[out_idx + 1u] = b1; }
            if (out_idx + 2u < total_f32) { f32_data[out_idx + 2u] = b2; }
            if (out_idx + 3u < total_f32) { f32_data[out_idx + 3u] = b3; }
        }
    `;

    const module = device.createShaderModule({ code: shaderCode });
    computePipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module,
            entryPoint: "main",
        },
    });

    computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: u8GpuBuffer } },
            { binding: 1, resource: { buffer: f32GpuBuffer } },
        ],
    });

    return true;
}

export function toInputTensorFromU8CHW(payload, H = INPUT_H, W = INPUT_W) {
    if (_dbgCtx) _debugDrawCHW_fromU8CHW(payload, H, W);

    if (initGpuIoIfNeeded(H, W)) {
        // Fast path: upload to GPU and process with compute shader
        device.queue.writeBuffer(u8GpuBuffer, 0, payload);

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, computeBindGroup);
        const workgroups = Math.ceil(Math.ceil((3 * H * W) / 4) / 64);
        passEncoder.dispatchWorkgroups(workgroups);
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);

        return ort.Tensor.fromGpuBuffer(f32GpuBuffer, {
            dataType: "float32",
            dims: [1, 3, H, W],
        });
    }

    // Fallback: CPU loop
    for (let i = 0; i < NUM; i++) f32InputBuffer[i] = payload[i] * (1 / 255);
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

let videoComputePipeline = null;
let videoF32GpuBuffer = null;
let videoParamsBuffer = null;
let videoSampler = null;
let currentVideoW = 0;
let currentVideoH = 0;

export function toInputTensorFromVideo(
    video,
    W = INPUT_W,
    H = INPUT_H,
    flipH = false,
) {
    if (!device) throw new Error("WebGPU device not initialized");

    const numFloats = 3 * W * H;

    if (videoF32GpuBuffer && (currentVideoW !== W || currentVideoH !== H)) {
        videoF32GpuBuffer.destroy();
        videoF32GpuBuffer = null;
    }
    currentVideoW = W;
    currentVideoH = H;

    if (!videoF32GpuBuffer) {
        videoF32GpuBuffer = device.createBuffer({
            size: numFloats * 4,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST,
        });
        videoParamsBuffer = device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        videoSampler = device.createSampler({
            magFilter: "linear",
            minFilter: "linear",
        });
    }

    if (!videoComputePipeline) {
        const shaderCode = `
            @group(0) @binding(0) var src_tex: texture_external;
            @group(0) @binding(1) var src_sampler: sampler;
            @group(0) @binding(2) var<storage, read_write> dst_f32: array<f32>;

            struct Params {
                out_w: u32,
                out_h: u32,
                src_w: f32,
                src_h: f32,
                pad_x: f32,
                pad_y: f32,
                used_w: f32,
                used_h: f32,
                flip_h: u32,
            }
            @group(0) @binding(3) var<uniform> params: Params;

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = global_id.x;
                let y = global_id.y;
                if (x >= params.out_w || y >= params.out_h) {
                    return;
                }

                var px = f32(x);
                if (params.flip_h != 0u) {
                    px = f32(params.out_w - 1u - x);
                }

                let nx = (px - params.pad_x) / params.used_w;
                let ny = (f32(y) - params.pad_y) / params.used_h;

                var color = vec3<f32>(0.0, 0.0, 0.0);
                if (nx >= 0.0 && nx <= 1.0 && ny >= 0.0 && ny <= 1.0) {
                    let sample_rgba = textureSampleBaseClampToEdge(src_tex, src_sampler, vec2<f32>(nx, ny));
                    color = sample_rgba.rgb;
                }

                let plane = params.out_w * params.out_h;
                let p = y * params.out_w + x;
                
                // Keep output format in float32 CHW
                dst_f32[0u * plane + p] = color.r;
                dst_f32[1u * plane + p] = color.g;
                dst_f32[2u * plane + p] = color.b;
            }
        `;
        const module = device.createShaderModule({ code: shaderCode });
        videoComputePipeline = device.createComputePipeline({
            layout: "auto",
            compute: { module, entryPoint: "main" },
        });
    }

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const scl = Math.min(W / vw, H / vh);
    const dw = vw * scl;
    const dh = vh * scl;
    const ox = ((W - dw) / 2) | 0;
    const oy = ((H - dh) / 2) | 0;

    const externalTexture = device.importExternalTexture({ source: video });

    const paramsData = new ArrayBuffer(48);
    const paramsF32 = new Float32Array(paramsData);
    const paramsU32 = new Uint32Array(paramsData);
    paramsU32[0] = W;
    paramsU32[1] = H;
    paramsF32[2] = vw;
    paramsF32[3] = vh;
    paramsF32[4] = ox;
    paramsF32[5] = oy;
    paramsF32[6] = dw;
    paramsF32[7] = dh;
    paramsU32[8] = flipH ? 1 : 0;

    device.queue.writeBuffer(videoParamsBuffer, 0, paramsData);

    const bindGroup = device.createBindGroup({
        layout: videoComputePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: externalTexture },
            { binding: 1, resource: videoSampler },
            { binding: 2, resource: { buffer: videoF32GpuBuffer } },
            { binding: 3, resource: { buffer: videoParamsBuffer } },
        ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(videoComputePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(W / 16), Math.ceil(H / 16));
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    return ort.Tensor.fromGpuBuffer(videoF32GpuBuffer, {
        dataType: "float32",
        dims: [1, 3, H, W],
    });
}
