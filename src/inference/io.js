// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import * as ort from "onnxruntime-web/webgpu";
import { INPUT_W, INPUT_H, DEV_MODE, USE_BINARY } from "../config.js";
import { device } from "./onnx.js";
import { setStatus } from "../ui.js";
import {
    getOrientedSourceCoords,
    getRotatedSize,
    normalizeRotationDeg,
} from "../utils/orientation.js";

const DEFAULT_NUM = 3 * INPUT_H * INPUT_W;
export let f32InputBuffer = new Float32Array(DEFAULT_NUM);
export let inputTensor = new ort.Tensor("float32", f32InputBuffer, [
    1,
    3,
    INPUT_H,
    INPUT_W,
]);
let currentCpuW = INPUT_W;
let currentCpuH = INPUT_H;

function ensureCpuInputTensor(H = INPUT_H, W = INPUT_W) {
    if (currentCpuH === H && currentCpuW === W) return inputTensor;
    currentCpuH = H;
    currentCpuW = W;
    f32InputBuffer = new Float32Array(3 * H * W);
    inputTensor = new ort.Tensor("float32", f32InputBuffer, [1, 3, H, W]);
    return inputTensor;
}

// Debug Canvas
let _dbgCtx = null;
if (typeof document !== "undefined") {
    if (DEV_MODE && USE_BINARY) {
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
    ensureCpuInputTensor(H, W);
    const num = 3 * H * W;
    for (let i = 0; i < num; i++) f32InputBuffer[i] = payload[i] * (1 / 255);
    return inputTensor;
}

export function toInputTensorFromImageData(imgData, transform = {}) {
    const src = imgData.data;
    const srcW = imgData.width;
    const srcH = imgData.height;
    const flipH = !!transform.flipH;
    const flipV = !!transform.flipV;
    const rotationDeg = normalizeRotationDeg(transform.rotationDeg);
    const rotatedSize = getRotatedSize(srcW, srcH, rotationDeg);
    const outW = rotatedSize.width;
    const outH = rotatedSize.height;
    const plane = outW * outH;

    ensureCpuInputTensor(outH, outW);

    if (!flipH && !flipV && rotationDeg === 0) {
        for (let y = 0; y < srcH; y++) {
            for (let x = 0; x < srcW; x++) {
                const p = y * srcW + x;
                const s = p * 4;
                f32InputBuffer[0 * plane + p] = src[s] / 255;
                f32InputBuffer[1 * plane + p] = src[s + 1] / 255;
                f32InputBuffer[2 * plane + p] = src[s + 2] / 255;
            }
        }
        return inputTensor;
    }

    for (let y = 0; y < outH; y++) {
        for (let x = 0; x < outW; x++) {
            const p = y * outW + x;
            const srcPos = getOrientedSourceCoords(
                x,
                y,
                srcW,
                srcH,
                flipH,
                flipV,
                rotationDeg,
            );
            const s = (srcPos.y * srcW + srcPos.x) * 4;
            f32InputBuffer[0 * plane + p] = src[s] / 255;
            f32InputBuffer[1 * plane + p] = src[s + 1] / 255;
            f32InputBuffer[2 * plane + p] = src[s + 2] / 255;
        }
    }

    return inputTensor;
}

let videoComputePipeline = null;
let videoComputePipeline_CEF = null;
let cefVideoTexture = null;
let cefVideoCanvas = null;
let cefVideoCtx = null;
let videoF32GpuBuffer = null;
let videoParamsBuffer = null;
let videoSampler = null;
let currentVideoW = 0;
let currentVideoH = 0;

let _videoCanvas = null;
let _videoCtx = null;
let _didLogBypass = false;
let _didLogWebGPU = false;
let _asyncChecked = false;
let webgpuVideoFailed = false;

let _cpuPixelCheckDone = false;

function toInputTensorFromVideo_CPU(video, W, H, transform = {}) {
    if (!_videoCanvas) {
        _videoCanvas = document.createElement("canvas");
        _videoCanvas.width = W;
        _videoCanvas.height = H;
        _videoCtx = _videoCanvas.getContext("2d", { willReadFrequently: true });
    } else if (_videoCanvas.width !== W || _videoCanvas.height !== H) {
        _videoCanvas.width = W;
        _videoCanvas.height = H;
    }

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const scl = Math.min(W / vw, H / vh);
    const dw = vw * scl;
    const dh = vh * scl;
    const ox = ((W - dw) / 2) | 0;
    const oy = ((H - dh) / 2) | 0;

    _videoCtx.fillStyle = "black";
    _videoCtx.fillRect(0, 0, W, H);
    _videoCtx.drawImage(video, 0, 0, vw, vh, ox, oy, dw, dh);
    const imgData = _videoCtx.getImageData(0, 0, W, H);

    // One-time pixel check: sample centre pixel to see if canvas draw is returning real data
    if (!_cpuPixelCheckDone) {
        _cpuPixelCheckDone = true;
        const cx = (W / 2) | 0;
        const cy = (H / 2) | 0;
        const idx = (cy * W + cx) * 4;
        const r = imgData.data[idx];
        const g = imgData.data[idx + 1];
        const b = imgData.data[idx + 2];
        if (r === 0 && g === 0 && b === 0) {
            setStatus(
                `CPU fallback: canvas black! readyState=${video.readyState} vid=${vw}x${vh}`,
            );
        } else {
            setStatus(
                `CPU fallback OK: centre px r=${r} g=${g} b=${b} | vid=${vw}x${vh}`,
            );
        }
    }

    return toInputTensorFromImageData(imgData, transform);
}

let _bitmapCanvas = null;
let _bitmapCtx = null;

export function toInputTensorFromBitmap(
    bitmap,
    W = INPUT_W,
    H = INPUT_H,
    transform = {},
) {
    if (!_bitmapCanvas) {
        _bitmapCanvas = document.createElement("canvas");
        _bitmapCtx = _bitmapCanvas.getContext("2d", {
            willReadFrequently: true,
        });
    }
    if (_bitmapCanvas.width !== W || _bitmapCanvas.height !== H) {
        _bitmapCanvas.width = W;
        _bitmapCanvas.height = H;
    }

    const bw = bitmap.width;
    const bh = bitmap.height;
    const scl = Math.min(W / bw, H / bh);
    const dw = bw * scl;
    const dh = bh * scl;
    const ox = ((W - dw) / 2) | 0;
    const oy = ((H - dh) / 2) | 0;

    _bitmapCtx.fillStyle = "black";
    _bitmapCtx.fillRect(0, 0, W, H);
    _bitmapCtx.drawImage(bitmap, 0, 0, bw, bh, ox, oy, dw, dh);
    const imgData = _bitmapCtx.getImageData(0, 0, W, H);

    // One-time pixel check
    if (!_cpuPixelCheckDone) {
        _cpuPixelCheckDone = true;
        const cx = (W / 2) | 0;
        const cy = (H / 2) | 0;
        const i = (cy * W + cx) * 4;
        const r = imgData.data[i],
            g = imgData.data[i + 1],
            b = imgData.data[i + 2];
        if (r === 0 && g === 0 && b === 0) {
            setStatus(`ImageCapture: canvas still black! bitmap=${bw}x${bh}`);
        } else {
            setStatus(`ImageCapture OK: centre px r=${r} g=${g} b=${b}`);
        }
    }

    return toInputTensorFromImageData(imgData, transform);
}

let testReadBuffer = null;

export function toInputTensorFromVideo(
    video,
    W = INPUT_W,
    H = INPUT_H,
    transform = {},
) {
    const flipH = !!transform.flipH;
    const flipV = !!transform.flipV;
    const rotationDeg = normalizeRotationDeg(transform.rotationDeg);

    // Automatically detect TouchDesigner's CEF environment string
    const isCEF =
        typeof navigator !== "undefined" &&
        (navigator.userAgent.indexOf("TouchDesigner") !== -1 ||
            navigator.userAgent.indexOf("CEF") !== -1);

    // Automatically bypass WebGPU compute shader entirely if we are inside TouchDesigner CEF
    // or if a silent WebGPU failure has been detected.
    if (isCEF || webgpuVideoFailed || flipV || rotationDeg !== 0) {
        if (!_didLogBypass) {
            console.log(
                `CEF pipeline detected or WebGPU failed (failed=${webgpuVideoFailed}): forcing pure CPU fallback to parse video element pixels.`,
            );
            _didLogBypass = true;
        }
        return toInputTensorFromVideo_CPU(video, W, H, {
            flipH,
            flipV,
            rotationDeg,
        });
    }

    try {
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

        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;

        // For CEF, we use a custom pipeline with copyExternalImageToTexture (standard 2D texture)
        // For Chrome, we use the zero-copy importExternalTexture
        if (isCEF && !videoComputePipeline_CEF) {
            const shaderCode = `
                @group(0) @binding(0) var src_tex: texture_2d<f32>;
                @group(0) @binding(1) var src_sampler: sampler;
                @group(0) @binding(2) var<storage, read_write> dst_f32: array<f32>;

                struct Params {
                    out_w: u32, out_h: u32, src_w: f32, src_h: f32,
                    pad_x: f32, pad_y: f32, used_w: f32, used_h: f32, flip_h: u32,
                }
                @group(0) @binding(3) var<uniform> params: Params;

                @compute @workgroup_size(16, 16)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let x = global_id.x; let y = global_id.y;
                    if (x >= params.out_w || y >= params.out_h) { return; }

                    var px = f32(x);
                    if (params.flip_h != 0u) { px = f32(params.out_w - 1u - x); }

                    let nx = (px - params.pad_x) / params.used_w;
                    let ny = (f32(y) - params.pad_y) / params.used_h;

                    var color = vec3<f32>(0.0, 0.0, 0.0);
                    if (nx >= 0.0 && nx <= 1.0 && ny >= 0.0 && ny <= 1.0) {
                        color = textureSampleLevel(src_tex, src_sampler, vec2<f32>(nx, ny), 0.0).rgb;
                    }
                    let plane = params.out_w * params.out_h;
                    let p = y * params.out_w + x;
                    dst_f32[0u * plane + p] = color.r;
                    dst_f32[1u * plane + p] = color.g;
                    dst_f32[2u * plane + p] = color.b;
                }
            `;
            videoComputePipeline_CEF = device.createComputePipeline({
                layout: "auto",
                compute: {
                    module: device.createShaderModule({ code: shaderCode }),
                    entryPoint: "main",
                },
            });
        } else if (!isCEF && !videoComputePipeline) {
            const shaderCode = `
                @group(0) @binding(0) var src_tex: texture_external;
                @group(0) @binding(1) var src_sampler: sampler;
                @group(0) @binding(2) var<storage, read_write> dst_f32: array<f32>;

                struct Params {
                    out_w: u32, out_h: u32, src_w: f32, src_h: f32,
                    pad_x: f32, pad_y: f32, used_w: f32, used_h: f32, flip_h: u32,
                }
                @group(0) @binding(3) var<uniform> params: Params;

                @compute @workgroup_size(16, 16)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let x = global_id.x; let y = global_id.y;
                    if (x >= params.out_w || y >= params.out_h) { return; }

                    var px = f32(x);
                    if (params.flip_h != 0u) { px = f32(params.out_w - 1u - x); }

                    let nx = (px - params.pad_x) / params.used_w;
                    let ny = (f32(y) - params.pad_y) / params.used_h;

                    var color = vec3<f32>(0.0, 0.0, 0.0);
                    if (nx >= 0.0 && nx <= 1.0 && ny >= 0.0 && ny <= 1.0) {
                        color = textureSampleBaseClampToEdge(src_tex, src_sampler, vec2<f32>(nx, ny)).rgb;
                    }
                    let plane = params.out_w * params.out_h;
                    let p = y * params.out_w + x;
                    dst_f32[0u * plane + p] = color.r;
                    dst_f32[1u * plane + p] = color.g;
                    dst_f32[2u * plane + p] = color.b;
                }
            `;
            videoComputePipeline = device.createComputePipeline({
                layout: "auto",
                compute: {
                    module: device.createShaderModule({ code: shaderCode }),
                    entryPoint: "main",
                },
            });
        }

        const diagMsg = `WebGPU | ready:${video.readyState} | vid:${vw}x${vh} | tgt:${W}x${H}`;

        const scl = Math.min(W / vw, H / vh);
        const dw = vw * scl;
        const dh = vh * scl;
        const ox = ((W - dw) / 2) | 0;
        const oy = ((H - dh) / 2) | 0;

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

        let bindGroup;

        if (isCEF) {
            // CEF Path using copyExternalImageToTexture
            if (
                !cefVideoTexture ||
                cefVideoTexture.width !== vw ||
                cefVideoTexture.height !== vh
            ) {
                if (cefVideoTexture) cefVideoTexture.destroy();
                cefVideoTexture = device.createTexture({
                    size: [vw, vh, 1],
                    format: "rgba8unorm",
                    usage:
                        GPUTextureUsage.TEXTURE_BINDING |
                        GPUTextureUsage.COPY_DST |
                        GPUTextureUsage.RENDER_ATTACHMENT,
                });
            }

            // In CEF, direct video-to-texture fails silently.
            // Draw to a 2D canvas first, then copy the canvas to the WebGPU texture.
            if (!cefVideoCanvas) {
                cefVideoCanvas = document.createElement("canvas");
                cefVideoCtx = cefVideoCanvas.getContext("2d", {
                    willReadFrequently: true,
                });
            }
            if (cefVideoCanvas.width !== vw || cefVideoCanvas.height !== vh) {
                cefVideoCanvas.width = vw;
                cefVideoCanvas.height = vh;
            }
            // Draw video to local 2D context to materialize the frame
            cefVideoCtx.drawImage(video, 0, 0, vw, vh);

            // Copy intermediate Canvas to GPUTexture
            device.queue.copyExternalImageToTexture(
                { source: cefVideoCanvas },
                { texture: cefVideoTexture },
                [vw, vh],
            );

            bindGroup = device.createBindGroup({
                layout: videoComputePipeline_CEF.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: cefVideoTexture.createView() },
                    { binding: 1, resource: videoSampler },
                    { binding: 2, resource: { buffer: videoF32GpuBuffer } },
                    { binding: 3, resource: { buffer: videoParamsBuffer } },
                ],
            });
        } else {
            // Chrome standard zero-copy Path using importExternalTexture
            const externalTexture = device.importExternalTexture({
                source: video,
            });
            bindGroup = device.createBindGroup({
                layout: videoComputePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: externalTexture },
                    { binding: 1, resource: videoSampler },
                    { binding: 2, resource: { buffer: videoF32GpuBuffer } },
                    { binding: 3, resource: { buffer: videoParamsBuffer } },
                ],
            });
        }

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(
            isCEF ? videoComputePipeline_CEF : videoComputePipeline,
        );
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(W / 16), Math.ceil(H / 16));
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);

        // --- ASYNC DEBUG & SELF-HEALING: Check if the texture read successfully ---
        if (!_asyncChecked && video.currentTime > 0.5) {
            _asyncChecked = true;
            if (!testReadBuffer) {
                testReadBuffer = device.createBuffer({
                    size: 16, // 4 floats
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                });
            }
            const readbackEncoder = device.createCommandEncoder();
            const centerIdx = Math.floor(H / 2) * W + Math.floor(W / 2);
            readbackEncoder.copyBufferToBuffer(
                videoF32GpuBuffer,
                centerIdx * 4,
                testReadBuffer,
                0,
                16,
            );
            device.queue.submit([readbackEncoder.finish()]);

            testReadBuffer
                .mapAsync(GPUMapMode.READ)
                .then(() => {
                    const arr = new Float32Array(
                        testReadBuffer.getMappedRange(),
                    );
                    const r = arr[0];
                    const g = arr[1];
                    const b = arr[2];
                    const a = arr[3];
                    const isAllZero = r === 0 && g === 0 && b === 0 && a === 0;
                    testReadBuffer.unmap();

                    if (isAllZero) {
                        // Double check on CPU if the video is actually black, or if it's a WebGPU silent failure
                        const checkCanvas = document.createElement("canvas");
                        checkCanvas.width = 10;
                        checkCanvas.height = 10;
                        const checkCtx = checkCanvas.getContext("2d");
                        try {
                            checkCtx.drawImage(video, 0, 0, 10, 10);
                            const imgData = checkCtx.getImageData(0, 0, 10, 10);
                            let cpuIsAllZero = true;
                            for (let i = 0; i < imgData.data.length; i += 4) {
                                if (
                                    imgData.data[i] !== 0 ||
                                    imgData.data[i + 1] !== 0 ||
                                    imgData.data[i + 2] !== 0
                                ) {
                                    cpuIsAllZero = false;
                                    break;
                                }
                            }

                            if (!cpuIsAllZero) {
                                // WebGPU is silent failing! Video has content, but GPU buffer is zero.
                                webgpuVideoFailed = true;
                                _didLogBypass = false; // Allow printing bypass log
                                setStatus(
                                    `WebGPU Silent Fail => Texture is all black. Falling back to CPU frame extraction.`,
                                );
                                console.warn(
                                    "WebGPU silent fail detected: Video texture is all black, but video has real pixel content. Falling back to CPU frame parsing.",
                                );
                            } else {
                                // Video itself is black (e.g. camera covered or initializing). Reset check so we check again.
                                _asyncChecked = false;
                            }
                        } catch (drawErr) {
                            console.warn(
                                "Failed to CPU check video frame during WebGPU silent fail test:",
                                drawErr,
                            );
                            // If drawing fails, let's also fallback just in case
                            webgpuVideoFailed = true;
                            _didLogBypass = false;
                            setStatus(
                                `WebGPU Silent Fail => Canvas check failed. Falling back to CPU.`,
                            );
                        }
                    } else {
                        setStatus(
                            `WebGPU Image OK! Middle Pixel: r=${r.toFixed(2)}, g=${g.toFixed(2)}`,
                        );
                    }
                })
                .catch((e) => {
                    setStatus("WebGPU Test readback failed: " + e.message);
                    webgpuVideoFailed = true;
                    _didLogBypass = false;
                });
        }

        // We only set the regular status if we aren't in the middle of our async test check that replaces it
        if (_asyncChecked && !testReadBuffer) {
            setStatus(diagMsg + " | Dispatched OK.");
        }

        return ort.Tensor.fromGpuBuffer(videoF32GpuBuffer, {
            dataType: "float32",
            dims: [1, 3, H, W],
        });
    } catch (e) {
        const errMsg = e.message ? e.message : String(e);
        setStatus("WebGPU Fail: " + errMsg);
        return toInputTensorFromVideo_CPU(video, W, H, {
            flipH,
            flipV,
            rotationDeg,
        });
    }
}
