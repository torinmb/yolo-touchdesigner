// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import * as ort from "onnxruntime-web/webgpu";
import {
    INPUT_W,
    INPUT_H,
    DET_SCORE_T,
    DET_IOU_T,
    DET_TOPK,
    SEG_SCORE_T,
    SEG_TOPK,
    POSE_IOU_T,
    POSE_TOPK,
    POSE_SCORE_T,
    MODEL_DETECT_KEY,
    MODEL_POSE_KEY,
    MODEL_SEG_KEY,
    ENABLE_DET,
    ENABLE_POSE,
    ENABLE_SEG,
    USE_CPU,
} from "../config.js";
import {
    decodeYOLO_or_OBB,
    decodeYOLOPose,
    decodeYOLOSeg,
    decodeYOLOv26,
    decodeYOLOv26Pose,
    nmsPerClass,
} from "./postprocess.js";

// Setup ORT Environment
ort.env.wasm.wasmPaths = "./";
ort.env.allowLocalModels = true;
ort.env.allowRemoteModels = false;
ort.env.useBrowserCache = false;

// Shared State
export let detSession = null;
export let detNmsSession = null;
export let poseSession = null;
export let segSession = null;
export let device = null;
export let IS_OBB = false;
export let IS_V26 = false;

// GPU Fetches
let tBoxes = null,
    tScores = null,
    tClasses = null;
let gpuFetches = null;
let gpuFetchesReady = false;

const det_tensor_topk = new ort.Tensor("int32", new Int32Array([DET_TOPK]));
const det_tensor_iou_threshold = new ort.Tensor(
    "float32",
    new Float32Array([DET_IOU_T]),
);
const det_tensor_score_thresh = new ort.Tensor(
    "float32",
    new Float32Array([DET_SCORE_T]),
);

function makeGpuTensor(nelem, dtype, dims) {
    const bytesPerElem = dtype === "float16" ? 2 : 4;
    const bytes = nelem * bytesPerElem;
    // @ts-ignore
    const buf = device.createBuffer({
        size: Math.ceil(bytes / 16) * 16,
        usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
    });
    return ort.Tensor.fromGpuBuffer(buf, { dataType: dtype, dims });
}

function prepareGpuFetchesForNms(sess, topk) {
    try {
        if (!device || !sess) return;
        tBoxes = makeGpuTensor(topk * 4, "float32", [topk, 4]);
        tScores = makeGpuTensor(topk, "float32", [topk]);
        tClasses = makeGpuTensor(topk, "float32", [topk]);

        gpuFetches = {};
        let haveBoxes = false;
        const oneDSlots = [];
        for (const name of sess.outputNames) {
            const md = sess.outputMetadata[name];
            const dims = md?.dimensions || [];
            if (dims.length === 2 && dims[1] === 4 && !haveBoxes) {
                gpuFetches[name] = tBoxes;
                haveBoxes = true;
            } else if (dims.length === 1) {
                oneDSlots.push(name);
            }
        }
        if (oneDSlots.length >= 2) {
            gpuFetches[oneDSlots[0]] = tScores;
            gpuFetches[oneDSlots[1]] = tClasses;
        } else {
            gpuFetches = null;
            gpuFetchesReady = false;
            return;
        }
        gpuFetchesReady = true;
    } catch (e) {
        console.warn(
            "Failed to prebind GPU fetches; falling back to default:",
            e,
        );
        gpuFetches = null;
        gpuFetchesReady = false;
    }
}

export async function initSessions(baseURL) {
    const providers = USE_CPU ? ["wasm"] : ["webgpu"];

    // 1. Detect Session
    if (ENABLE_DET) {
        const path = `${baseURL}models/${MODEL_DETECT_KEY}.onnx`;
        detSession = await ort.InferenceSession.create(path, {
            executionProviders: providers,
            graphOptimizationLevel: "all",
        });

        // OBB / V26 Heuristics
        IS_OBB = /(^|[\\/])[^\\/]*obb/i.test(MODEL_DETECT_KEY);
        IS_V26 = false;

        try {
            const out0 =
                detSession &&
                detSession.outputMetadata[detSession.outputNames[0]];
            const dims = out0?.dimensions || [];

            // Check for v26 signature: [1, 300, 6]
            if (dims.length === 3 && dims[1] === 300 && dims[2] === 6) {
                IS_V26 = true;
            }

            const C =
                (dims[1] > 10 ? dims[1] : undefined) ??
                (dims[2] > 10 ? dims[2] : undefined);

            // Only try OBB heuristic if not v26
            if (!IS_V26 && !IS_OBB && typeof C === "number" && C >= 6) {
                const maybeNcOBB = C - 5;
                if (maybeNcOBB > 0) IS_OBB = true;
            }
        } catch {}
    }

    // 2. Pose Session
    if (ENABLE_POSE) {
        const path = `${baseURL}models/${MODEL_POSE_KEY}.onnx`;
        poseSession = await ort.InferenceSession.create(path, {
            executionProviders: providers,
            graphOptimizationLevel: "all",
        });
    }

    // 2b. Seg Session
    if (ENABLE_SEG) {
        const path = `${baseURL}models/${MODEL_SEG_KEY}.onnx`;
        segSession = await ort.InferenceSession.create(path, {
            executionProviders: providers,
            graphOptimizationLevel: "all",
        });
    }

    // 3. WebGPU Device
    if (!USE_CPU) {
        try {
            device =
                ort.env.webgpu?.device ||
                (navigator.gpu &&
                    (await navigator.gpu.requestAdapter()) &&
                    (await (
                        await navigator.gpu.requestAdapter()
                    ).requestDevice()));
        } catch (e) {
            console.warn("WebGPU init failed", e);
        }
    }

    // 4. Detect NMS Session
    if (detSession && !IS_V26) {
        const nmsPath = `${baseURL}yolo-decoder.onnx`;
        try {
            if (!IS_OBB) {
                detNmsSession = await ort.InferenceSession.create(nmsPath, {
                    executionProviders: providers,
                    graphOptimizationLevel: "all",
                });
                if (device) prepareGpuFetchesForNms(detNmsSession, DET_TOPK);
            }
        } catch (e) {
            console.warn("Detect decoder not available; using JS NMS.", e);
            detNmsSession = null;
        }
    }
}

export async function runDetect(input) {
    if (!detSession) return [];
    const outs = await detSession.run({ [detSession.inputNames[0]]: input });
    const head = outs[detSession.outputNames[0]];

    // v26 / DETR / NMS-included support
    if (head.dims.length === 3 && head.dims[1] === 300 && head.dims[2] === 6) {
        return decodeYOLOv26(head, DET_SCORE_T, DET_TOPK);
    }

    let keep = [];

    // Fast path via ONNX decoder/NMS
    if (!IS_OBB && detNmsSession) {
        const feeds = {
            [detNmsSession.inputNames[0]]: head,
            [detNmsSession.inputNames[1]]: det_tensor_topk,
            [detNmsSession.inputNames[2]]: det_tensor_iou_threshold,
            [detNmsSession.inputNames[3]]: det_tensor_score_thresh,
        };
        const nmsOuts = gpuFetchesReady
            ? await detNmsSession.run(feeds, gpuFetches)
            : await detNmsSession.run(feeds);

        const outsList = detNmsSession.outputNames.map((n) => nmsOuts[n]);
        let boxesT = null,
            scoresT = null,
            classesT = null;

        for (const t of outsList) {
            const dims = t.dims;
            if (dims.length === 2 && dims[1] === 4 && !boxesT) boxesT = t;
            else if (dims.length === 1 && t.type === "float32" && !scoresT)
                scoresT = t;
        }
        // find classes
        for (const t of outsList) {
            if (
                t !== boxesT &&
                t !== scoresT &&
                t.dims.length === 1 &&
                !classesT
            )
                classesT = t;
        }

        if (boxesT && scoresT && classesT) {
            const boxes = boxesT.getData ? await boxesT.getData() : boxesT.data;
            const scores = scoresT.getData
                ? await scoresT.getData()
                : scoresT.data;
            const clsRaw = classesT.getData
                ? await classesT.getData()
                : classesT.data;
            const N = scores.length;
            for (let i = 0; i < N; i++) {
                const off = i * 4;
                const x1 = boxes[off + 0],
                    y1 = boxes[off + 1];
                const x2 = boxes[off + 2],
                    y2 = boxes[off + 3];
                const w = Math.max(1, x2 - x1);
                const h = Math.max(1, y2 - y1);
                const clsIdx =
                    typeof clsRaw[i] === "bigint"
                        ? Number(clsRaw[i])
                        : clsRaw[i] | 0;
                keep.push({
                    box: [x1, y1, w, h],
                    label: clsIdx,
                    score: scores[i],
                });
            }
            return keep;
        }
    }

    // JS Decode Path (Fallback/OBB)
    const dets = decodeYOLO_or_OBB(head, DET_SCORE_T, DET_TOPK, IS_OBB);
    return nmsPerClass(dets, DET_IOU_T, DET_TOPK);
}

export async function runPose(input) {
    if (!poseSession) return [];
    const outs = await poseSession.run({
        [detSession ? detSession.inputNames[0] : poseSession.inputNames[0]]:
            input,
    });
    const head = outs[poseSession.outputNames[0]];

    // v26 / DETR / NMS-included support
    if (head.dims.length === 3 && head.dims[1] === 300 && head.dims[2] === 57) {
        return decodeYOLOv26Pose(head, POSE_SCORE_T, POSE_TOPK);
    }

    const dets = decodeYOLOPose(
        head,
        POSE_SCORE_T,
        POSE_TOPK,
        INPUT_W,
        INPUT_H,
    );
    return nmsPerClass(dets, POSE_IOU_T, POSE_TOPK);
}

export async function runSeg(input) {
    if (!segSession) return null;
    const outs = await segSession.run({
        [segSession.inputNames[0]]: input,
    });
    return decodeYOLOSeg(outs, segSession.outputNames, SEG_SCORE_T, SEG_TOPK);
}
