// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

// yolo_websocket_uint8_chw_dual.js
// Two ingestion modes:
//
//   1) Binary mode (?binary=1) — Receive CHW uint8 via WebSocket (16B header)
//      -> FP32 -> run YOLO detect/pose (WebGPU), send predictions JSON.
//
//   2) Webcam mode (default) — GetUserMedia + RAF
//      Draw visible, full-screen <video> (object-fit: cover) for the user,
//      simultaneously repack to a square 640x640 canvas for the model,
//      run YOLO detect/pose (WebGPU), send predictions JSON.
//
// In both modes, we:
//  - Send `{loaded:true}` when WS connects
//  - Send `{webcamDevices: [{label, deviceId}, ...]}` on WS open (and again after permission)
//  - Support ?webcamLabel=Exact Camera Name
//  - Respect detect/pose toggles and thresholds from query params

import * as ort from "onnxruntime-web/webgpu";

/* ================================
   ORT environment
================================ */
ort.env.wasm.wasmPaths = "./";
ort.env.allowLocalModels = true;
ort.env.allowRemoteModels = false;
ort.env.useBrowserCache = false;
// ort.env.logLevel = "verbose";

/* ================================
   Query helpers (readable + strict)
================================ */
const qs = new URLSearchParams(location.search);
const boolish = (v, def = false) =>
    v == null ? def : /^(1|true|on|yes)$/i.test(String(v));

const pick = (...names) => {
    for (const n of names) if (n && qs.has(n)) return qs.get(n);
    return null;
};
const getStr = (names, fallback) => {
    const v = pick(...names);
    return v != null ? v : fallback;
};
const getBool = (names, fallback) => {
    const v = pick(...names);
    return v != null ? boolish(v, fallback) : fallback;
};
const getNum = (primaryNames, fallback, commonFallbackName) => {
    for (const n of primaryNames) {
        if (qs.has(n)) {
            const x = parseFloat(qs.get(n));
            if (!Number.isNaN(x)) return x;
        }
    }
    if (commonFallbackName && qs.has(commonFallbackName)) {
        const x = parseFloat(qs.get(commonFallbackName));
        if (!Number.isNaN(x)) return x;
    }
    return fallback;
};
const getInt = (primaryNames, fallback, commonFallbackName) => {
    for (const n of primaryNames) {
        if (qs.has(n)) {
            const x = parseInt(qs.get(n), 10);
            if (!Number.isNaN(x)) return x;
        }
    }
    if (commonFallbackName && qs.has(commonFallbackName)) {
        const x = parseInt(qs.get(commonFallbackName), 10);
        if (!Number.isNaN(x)) return x;
    }
    return fallback;
};

/* ================================
   Global config / toggles
================================ */
const wsPort = qs.get("wsPort") || "62309";
const USE_BINARY = getBool(["binary"], false); // default to webcam mode when false

// Stream toggles + models (with aliases/typos/legacy preserved)
let ENABLE_DET = getBool(["Objecttrackingenabled"], true);
let ENABLE_POSE = getBool(["Posetrackingenabled", "pose"], true);

let modelDetectKey = getStr(["Obecttrackingmodel"], "yolo11n");
let modelPoseKey = getStr(["Posemodel"], "yolo11n-pose");

// Legacy single `model=` inference, only if no explicit toggles were provided.
const legacyModel = qs.get("model");
const anyToggleProvided = [
    "Poseenabled",
    "pose",
    "Objecttrackingenabled",
    "detect",
].some((k) => qs.has(k));
if (legacyModel && !anyToggleProvided) {
    if (/pose/i.test(legacyModel)) {
        ENABLE_POSE = true;
        ENABLE_DET = false;
        modelPoseKey = legacyModel;
    } else {
        ENABLE_DET = true;
        ENABLE_POSE = false;
        modelDetectKey = legacyModel;
    }
}

// Per-task thresholds / tuning (with back-compat fallbacks)
// Detect (objects)
const DET_SCORE_T = getNum(["Detscoret"], 0.4, "Scoret");
const DET_IOU_T = getNum(["Detiout"], 0.45, "Iout");
const DET_TOPK = getInt(["Dettopk"], 100, "Topk");

// Pose (people keypoints)
const POSE_SCORE_T = getNum(["Posescoret"], 0.35, "Scoret");
const POSE_IOU_T = getNum(["Poseiout"], 0.45, "Iout");
const POSE_TOPK = getInt(["Posetopk"], 50, "Topk");

// Tiny IoU tracker (separate controls per stream; legacy Trkiou/Trkttl fallback)
const DET_TRK_IOU = getNum(["Detrkiou"], 0.5, "Trkiou");
const DET_TRK_TTL = getInt(["Detrkttl"], 2, "Trkttl");
const POSE_TRK_IOU = getNum(["Posetrkiou"], 0.5, "Trkiou");
const POSE_TRK_TTL = getInt(["Posetrkttl"], 2, "Trkttl");

// Webcam options
const WEBCAM_LABEL = getStr(["webcamLabel"], null);
const FLIP_HORIZONTAL = getBool(["flipHorizontal"], true); // visual + model input flip

/* ================================
   Constants & shared state
================================ */
const INPUT_W = 640,
    INPUT_H = 640;

let device = null;
let ws = null;

// latest mediaTime from requestVideoFrameCallback (webcam only)
let lastVideoMediaTime = 0; // seconds

// Sessions per stream
let detSession = null,
    detNmsSession = null;
let poseSession = null;

// GPU fetch tensors for detect decoder/NMS (detect-only)
let tBoxes = null,
    tScores = null,
    tClasses = null;
let gpuFetches = null;
let gpuFetchesReady = false;

// Pre-create threshold tensors for ONNX detect decoder (detect-only)
const det_tensor_topk = new ort.Tensor("int32", new Int32Array([DET_TOPK]));
const det_tensor_iou_threshold = new ort.Tensor(
    "float32",
    new Float32Array([DET_IOU_T])
);
const det_tensor_score_thresh = new ort.Tensor(
    "float32",
    new Float32Array([DET_SCORE_T])
);

// Input buffer reused for both binary + webcam paths
const NUM = 3 * INPUT_H * INPUT_W;
const f32InputBuffer = new Float32Array(NUM);
const inputTensor = new ort.Tensor("float32", f32InputBuffer, [
    1,
    3,
    INPUT_H,
    INPUT_W,
]);

/* ================================
   UI elements (status + video + offscreen canvas)
================================ */
const statusEl =
    document.getElementById("status") ||
    (() => {
        const el = document.createElement("div");
        el.id = "status";
        el.style.cssText =
            "position:fixed;left:12px;bottom:12px;padding:6px 8px;background:#111;color:#eee;font:12px/1.3 monospace;z-index:999999;border:1px solid #333;border-radius:6px;";
        document.body.appendChild(el);
        return el;
    })();

function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg || "";
}

/** Ensure a visible, full-screen <video> that mirrors (optional) and doesn’t eat clicks. */
function ensureVisibleVideo() {
    let v = document.getElementById("video");
    if (!v) {
        v = document.createElement("video");
        v.id = "video";
        v.autoplay = true;
        v.playsInline = true;
        v.muted = true;
        document.body.appendChild(v);
    }
    v.style.cssText = [
        "position:fixed",
        "inset:0",
        "width:100vw",
        "height:100vh",
        "object-fit:cover",
        "z-index:1",
        "background:#000",
        "pointer-events:none",
        `transform:${FLIP_HORIZONTAL ? "scaleX(-1)" : "none"}`,
        "transform-origin:center center",
    ].join(";");

    return v;
}

// Offscreen processing canvas (square 640x640 for model input)
const procCanvas = document.createElement("canvas");
procCanvas.width = INPUT_W;
procCanvas.height = INPUT_H;
const procCtx = procCanvas.getContext("2d", { willReadFrequently: true });

/* ================================
   Debug canvas (optional)
================================ */
const DEBUG_RAW = boolish(qs.get("debug"));
const DEBUG_FLIPY = boolish(qs.get("flipY"));
const DEBUG_SWAPRB = boolish(qs.get("swapRB"));
const _dbgCtx = (() => {
    if (!DEBUG_RAW || typeof document === "undefined") return null;
    const c = document.createElement("canvas");
    c.width = INPUT_W;
    c.height = INPUT_H;
    c.style.cssText =
        "position:fixed;right:12px;bottom:12px;border:1px solid #444;image-rendering:pixelated;z-index:99999;background:#000;";
    document.body.appendChild(c);
    return c.getContext("2d");
})();
function _debugDrawCHW_fromU8CHW(payloadU8, H = INPUT_H, W = INPUT_W) {
    if (!_dbgCtx) return;
    const plane = W * H;
    const out = new Uint8ClampedArray(4 * plane);
    for (let y = 0; y < H; y++) {
        const ySrc = DEBUG_FLIPY ? H - 1 - y : y;
        for (let x = 0; x < W; x++) {
            const idx = ySrc * W + x;
            const p = (y * W + x) * 4;
            let r = payloadU8[0 * plane + idx];
            let g = payloadU8[1 * plane + idx];
            let b = payloadU8[2 * plane + idx];
            if (DEBUG_SWAPRB) {
                const t = r;
                r = b;
                b = t;
            }
            out[p + 0] = r;
            out[p + 1] = g;
            out[p + 2] = b;
            out[p + 3] = 255;
        }
    }
    _dbgCtx.putImageData(new ImageData(out, W, H), 0, 0);
}

/* ================================
   Minimal IoU tracker (unchanged)
================================ */
class IoUTracker {
    constructor(iouMatch = 0.5, ttl = 1) {
        this.iouMatch = iouMatch;
        this.ttl = ttl;
        this.nextId = 1;
        this.tracks = new Map();
    }
    update(dets) {
        const ids = [...this.tracks.keys()];
        for (const id of ids) this.tracks.get(id).miss++;

        const pairs = [];
        for (let di = 0; di < dets.length; di++) {
            for (const id of ids) {
                const iou = this._iou(this.tracks.get(id).box, dets[di].box);
                pairs.push([iou, id, di]);
            }
        }
        pairs.sort((a, b) => b[0] - a[0]);

        const takenTrack = new Set();
        const takenDet = new Set();
        for (const [iou, id, di] of pairs) {
            if (iou < this.iouMatch) break;
            if (takenTrack.has(id) || takenDet.has(di)) continue;
            const t = this.tracks.get(id),
                d = dets[di];
            t.box = d.box;
            t.label = d.label;
            t.score = d.score;
            t.keypoints = d.keypoints;
            t.hits++;
            t.miss = 0;
            takenTrack.add(id);
            takenDet.add(di);
        }

        for (let di = 0; di < dets.length; di++) {
            if (takenDet.has(di)) continue;
            const d = dets[di];
            const id = this.nextId++;
            this.tracks.set(id, {
                box: d.box,
                label: d.label,
                score: d.score,
                keypoints: d.keypoints,
                age: 0,
                hits: 1,
                miss: 0,
            });
        }

        for (const id of [...this.tracks.keys()]) {
            const t = this.tracks.get(id);
            t.age++;
            if (t.miss > this.ttl) this.tracks.delete(id);
        }

        return [...this.tracks.entries()]
            .filter(([_, t]) => t.miss === 0)
            .map(([id, t]) => ({
                id,
                box: t.box,
                label: t.label,
                score: t.score,
                keypoints: t.keypoints,
            }));
    }
    _iou(a, b) {
        const [ax, ay, aw, ah] = a,
            [bx, by, bw, bh] = b;
        const ax2 = ax + aw,
            ay2 = ay + ah,
            bx2 = bx + bw,
            by2 = by + bh;
        const ix = Math.max(ax, bx),
            iy = Math.max(ay, by);
        const ix2 = Math.min(ax2, bx2),
            iy2 = Math.min(ay2, by2);
        const iw = Math.max(0, ix2 - ix),
            ih = Math.max(0, iy2 - iy);
        const inter = iw * ih;
        const uni = aw * ah + bw * bh - inter + 1e-9;
        return inter / uni;
    }
}
const trackerDet = new IoUTracker(DET_TRK_IOU, DET_TRK_TTL);
const trackerPose = new IoUTracker(POSE_TRK_IOU, POSE_TRK_TTL);

/* ================================
   Binary header parsing: <BBBBHHII>
   type(10), dtype(1=u8), layout(1=CHW), pad, H, W, seq, td_frame
================================ */
function parseHeader(buf) {
    if (buf.byteLength < 16) return null;
    const dv = new DataView(buf);
    const type = dv.getUint8(0);
    const dtype = dv.getUint8(1);
    const layout = dv.getUint8(2);
    const H = dv.getUint16(4, true);
    const W = dv.getUint16(6, true);
    const seq = dv.getUint32(8, true);
    const td = dv.getUint32(12, true);
    if (type !== 10 || dtype !== 1 || layout !== 1) return null; // require u8 CHW
    const payload = new Uint8Array(buf, 16); // length = 3*H*W
    return { H, W, seq, td, payload };
}

/* ================================
   Input packers (binary vs webcam)
================================ */
function toInputTensorFromU8CHW(job) {
    // job.payload is U8 CHW [R-plane | G-plane | B-plane] length=3*H*W
    const u8 = job.payload;
    for (let i = 0; i < NUM; i++) f32InputBuffer[i] = u8[i] * (1 / 255);
    if (_dbgCtx) _debugDrawCHW_fromU8CHW(u8, INPUT_H, INPUT_W);
    return inputTensor;
}

function toInputTensorFromImageData(imgData, flipH = false) {
    // RGBA -> CHW float32 [0..1]; optional horizontal flip for model input
    const src = imgData.data;
    const W = imgData.width,
        H = imgData.height;
    const plane = W * H;
    // precompute indexes for flip vs non-flip
    if (!flipH) {
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                const p = y * W + x;
                const s = p * 4;
                const r = src[s],
                    g = src[s + 1],
                    b = src[s + 2];
                f32InputBuffer[0 * plane + p] = r / 255;
                f32InputBuffer[1 * plane + p] = g / 255;
                f32InputBuffer[2 * plane + p] = b / 255;
            }
        }
    } else {
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                const x2 = W - 1 - x;
                const p = y * W + x;
                const s = (y * W + x2) * 4;
                const r = src[s],
                    g = src[s + 1],
                    b = src[s + 2];
                f32InputBuffer[0 * plane + p] = r / 255;
                f32InputBuffer[1 * plane + p] = g / 255;
                f32InputBuffer[2 * plane + p] = b / 255;
            }
        }
    }
    return inputTensor;
}

/* ================================
   Decoders (unchanged)
================================ */
function decodeYOLO(tensor, thr, topk) {
    const d = tensor.data,
        sh = tensor.dims;
    let num,
        dim,
        T = false;
    if (sh.length === 3) {
        if (sh[1] === 84) {
            num = sh[2];
            dim = sh[1];
            T = false;
        } else if (sh[2] === 84) {
            num = sh[1];
            dim = sh[2];
            T = true;
        } else {
            num = sh[sh.length - 1];
            dim = sh[sh.length - 2];
            T = sh[2] === dim;
        }
    } else if (sh.length === 2) {
        num = sh[0];
        dim = sh[1] - 1;
        T = true;
    } else return [];
    const W = INPUT_W,
        H = INPUT_H;
    const sb = T ? dim : 1;
    const sbx = T ? 1 : num;
    const out = [];
    for (let i = 0; i < num; i++) {
        const x = d[0 * sbx + i * sb];
        const y = d[1 * sbx + i * sb];
        const w = d[2 * sbx + i * sb];
        const h = d[3 * sbx + i * sb];
        let bestC = -1,
            bestS = -1;
        for (let c = 4; c < dim; c++) {
            const s = d[c * sbx + i * sb];
            if (s > bestS) {
                bestS = s;
                bestC = c - 4;
            }
        }
        if (bestS < thr) continue;
        let bx, by, bw, bh;
        if (w > 0 && h > 0 && w <= W * 2 && h <= H * 2) {
            bw = w;
            bh = h;
            bx = x - w / 2;
            by = y - h / 2;
        } else {
            const x2 = w,
                y2 = h;
            bx = x;
            by = y;
            bw = x2 - x;
            bh = y2 - y;
        }
        bx = Math.max(0, Math.min(W - 1, bx));
        by = Math.max(0, Math.min(H - 1, by));
        bw = Math.max(1, Math.min(W - bx, bw));
        bh = Math.max(1, Math.min(H - by, bh));
        out.push({ box: [bx, by, bw, bh], label: bestC, score: bestS });
    }
    out.sort((a, b) => b.score - a.score);
    if (out.length > topk) out.length = topk;
    return out;
}

function decodeYOLOPose(
    raw_tensor,
    score_threshold = 0.45,
    topk,
    W = INPUT_W,
    H = INPUT_H
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

function nmsPerClass(dets, iouThr, topk) {
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

    function iou(a, b) {
        const [ax, ay, aw, ah] = a,
            [bx, by, bw, bh] = b;
        const ax2 = ax + aw,
            ay2 = ay + ah,
            bx2 = bx + bw,
            by2 = by + bh;
        const ix = Math.max(ax, bx),
            iy = Math.max(ay, by);
        const ix2 = Math.min(ax2, bx2),
            iy2 = Math.min(ay2, by2);
        const iw = Math.max(0, ix2 - ix),
            ih = Math.max(0, iy2 - iy);
        const inter = iw * ih;
        const uni = aw * ah + bw * bh - inter + 1e-9;
        return inter / uni;
    }
}

function nmsPerClassWithKpts(dets, iouThr, topk) {
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

    function iou(a, b) {
        const [ax, ay, aw, ah] = a,
            [bx, by, bw, bh] = b;
        const ax2 = ax + aw,
            ay2 = ay + ah,
            bx2 = bx + bw,
            by2 = by + bh;
        const ix = Math.max(ax, bx),
            iy = Math.max(ay, by);
        const ix2 = Math.min(ax2, bx2),
            iy2 = Math.min(ay2, by2);
        const iw = Math.max(0, ix2 - ix),
            ih = Math.max(0, iy2 - iy);
        const inter = iw * ih;
        const uni = aw * ah + bw * bh - inter + 1e-9;
        return inter / uni;
    }
}

/* ================================
   GPU fetch tensor helpers (detect decoder only)
================================ */
function makeGpuTensor(nelem, dtype, dims) {
    const bytesPerElem = dtype === "float16" ? 2 : 4;
    const bytes = nelem * bytesPerElem;
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
            if (dims.length === 2 && !haveBoxes) {
                gpuFetches[name] = tBoxes;
                haveBoxes = true;
            } else if (dims.length === 1) {
                oneDSlots.push(name);
            } else {
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
            e
        );
        gpuFetches = null;
        gpuFetchesReady = false;
    }
}

/* ================================
   Coord mapping to normalized [0..1] with Y-flip to bottom-left origin
================================ */
function flipYKeypointsNorm(kpts, H) {
    if (!kpts) return kpts;
    return kpts.map((k) => ({ x: k.x / H, y: (H - k.y) / H, score: k.score }));
}
function mapBoxYFlipNorm([x, y, w, h], H) {
    return [x / H, (H - (y + h)) / H, w / H, h / H];
}

/* ================================
   Inference runners
================================ */
async function runDetect(input) {
    if (!detSession) return [];
    const outs = await detSession.run({ [detSession.inputNames[0]]: input });
    const head = outs[detSession.outputNames[0]];
    let keep = [];

    if (detNmsSession) {
        const feeds = {
            [detNmsSession.inputNames[0]]: head,
            [detNmsSession.inputNames[1]]: det_tensor_topk,
            [detNmsSession.inputNames[2]]: det_tensor_iou_threshold,
            [detNmsSession.inputNames[3]]: det_tensor_score_thresh,
        };
        const nmsOuts = gpuFetchesReady
            ? await detNmsSession.run(feeds, gpuFetches)
            : await detNmsSession.run(feeds);

        // Infer which outputs are boxes/scores/classes
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
        for (const t of outsList) {
            if (t === boxesT || t === scoresT) continue;
            if (t.dims.length === 1 && !classesT) classesT = t;
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
                const w = Math.max(1, x2 - x1),
                    h = Math.max(1, y2 - y1);
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
        } else {
            // fallback JS
            keep = nmsPerClass(
                decodeYOLO(head, DET_SCORE_T, DET_TOPK),
                DET_IOU_T,
                DET_TOPK
            );
        }
    } else {
        // JS
        keep = nmsPerClass(
            decodeYOLO(head, DET_SCORE_T, DET_TOPK),
            DET_IOU_T,
            DET_TOPK
        );
    }
    return keep;
}

async function runPose(input) {
    if (!poseSession) return [];
    const outs = await poseSession.run({ [poseSession.inputNames[0]]: input });
    const head = outs[poseSession.outputNames[0]];
    const dets = decodeYOLOPose(
        head,
        POSE_SCORE_T,
        POSE_TOPK,
        INPUT_W,
        INPUT_H
    );
    return nmsPerClassWithKpts(dets, POSE_IOU_T, POSE_TOPK);
}

/* ================================
   Output packing + send
================================ */
function packAndSendPreds(frameId, seq, keepDet, keepPose, videoFrame) {
    const H = INPUT_H;
    const predsDet = keepDet.map((t) => {
        const box = mapBoxYFlipNorm(t.box, H);
        return {
            tx: box[0],
            ty: box[1],
            width: box[2],
            height: box[3],
            categoryName: [t.label],
            score: t.score,
            id: t.id,
        };
    });

    const predsPose = keepPose.map((t) => {
        const box = mapBoxYFlipNorm(t.box, H);
        const kpts = flipYKeypointsNorm(t.keypoints, H);
        return {
            tx: box[0],
            ty: box[1],
            width: box[2],
            height: box[3],
            categoryName: [t.label],
            score: t.score,
            id: t.id,
            keypoints: kpts,
        };
    });

    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    if (detSession && poseSession) {
        ws.send(
            JSON.stringify({
                type: "yolo_combined",
                frame: frameId >>> 0,
                seq: seq >>> 0,
                videoFrame,
                width: INPUT_W,
                height: INPUT_H,
                yolo: predsDet,
                yolo_pose: predsPose,
            })
        );
    } else if (detSession) {
        ws.send(
            JSON.stringify({
                type: "yolo",
                frame: frameId >>> 0,
                seq: seq >>> 0,
                videoFrame,
                width: INPUT_W,
                height: INPUT_H,
                predictions: predsDet,
            })
        );
    } else if (poseSession) {
        ws.send(
            JSON.stringify({
                type: "yolo_pose",
                frame: frameId >>> 0,
                seq: seq >>> 0,
                videoFrame,
                width: INPUT_W,
                height: INPUT_H,
                predictions: predsPose,
            })
        );
    }
}

/* ================================
   Binary mode pump (unchanged core)
================================ */
let latestBinaryJob = null;
let binaryProcessing = false;
async function pumpBinary() {
    if (binaryProcessing) return;
    binaryProcessing = true;
    try {
        const job = latestBinaryJob;
        latestBinaryJob = null;
        if (!job || (!detSession && !poseSession)) return;

        const input = toInputTensorFromU8CHW(job);

        // Run streams
        let keepDet = [];
        let keepPose = [];
        if (detSession) keepDet = await runDetect(input);
        if (poseSession) keepPose = await runPose(input);

        // Track
        const tracksDet = detSession ? trackerDet.update(keepDet) : [];
        const tracksPose = poseSession ? trackerPose.update(keepPose) : [];

        packAndSendPreds(job.td, job.seq, tracksDet, tracksPose, 0);
    } catch (e) {
        console.error(e);
        setStatus(`Error (binary): ${e?.message || e}`);
    } finally {
        binaryProcessing = false;
        if (latestBinaryJob) queueMicrotask(pumpBinary);
    }
}

/* ================================
   Webcam mode (RAF) helpers
================================ */
function drawPaddedSquareFromVideo(video, ctx, size, fill = "#000") {
    ctx.canvas.width = size;
    ctx.canvas.height = size;
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    // fit inside (letterbox) so model always sees full image
    const scl = Math.min(size / vw, size / vh);
    const dw = Math.round(vw * scl);
    const dh = Math.round(vh * scl);
    const ox = ((size - dw) / 2) | 0;
    const oy = ((size - dh) / 2) | 0;

    ctx.save();
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = fill;
    ctx.fillRect(0, 0, size, size);
    // flip horizontally if requested, draw into padded square
    if (FLIP_HORIZONTAL) {
        ctx.translate(size, 0);
        ctx.scale(-1, 1);
        // when flipped, drawImage x is mirrored around canvas width
        // but because we center with ox, we can draw at (ox, oy) in the flipped space
        ctx.drawImage(video, ox, oy, dw, dh);
    } else {
        ctx.drawImage(video, ox, oy, dw, dh);
    }
    ctx.restore();

    return ctx.getImageData(0, 0, size, size);
}

let rafProcessing = false;
let prevT = 0;
async function rafLoop(video) {
    if (!video || video.readyState < 2) {
        requestAnimationFrame(() => rafLoop(video));
        return;
    }
    if (rafProcessing) {
        requestAnimationFrame(() => rafLoop(video));
        return;
    }
    rafProcessing = true;

    try {
        const frame = lastVideoMediaTime;
        // Pack webcam -> 640x640 RGBA -> CHW float32
        const img = drawPaddedSquareFromVideo(video, procCtx, INPUT_W, "#000");
        const input = toInputTensorFromImageData(img, /*flipH*/ false); // already flipped when drawing

        // Run streams
        let keepDet = [];
        let keepPose = [];
        if (detSession) keepDet = await runDetect(input);
        if (poseSession) keepPose = await runPose(input);

        // Trackers
        const tracksDet = detSession ? trackerDet.update(keepDet) : [];
        const tracksPose = poseSession ? trackerPose.update(keepPose) : [];

        // Send predictions; webcam has no td_frame/seq — use -1
        packAndSendPreds(-1, 0, tracksDet, tracksPose, frame);

        // Minimal FPS status
        const now = performance.now();
        if (prevT) {
            const fps = 1000 / (now - prevT);
            setStatus(
                `FPS: ${fps.toFixed(2)} ${USE_BINARY ? "(binary)" : "(webcam)"}`
            );
        }
        prevT = now;
    } catch (e) {
        console.error(e);
        setStatus(`Error (webcam): ${e?.message || e}`);
    } finally {
        rafProcessing = false;
        requestAnimationFrame(() => rafLoop(video));
    }
}

/* ================================
   Webcam utilities
================================ */
async function listWebcamDevices() {
    try {
        const all = await navigator.mediaDevices.enumerateDevices();
        return all
            .filter((d) => d.kind === "videoinput")
            .map((d) => ({ label: d.label || "", deviceId: d.deviceId || "" }));
    } catch (e) {
        console.warn("enumerateDevices failed:", e);
        return [];
    }
}

function findDeviceIdByLabel(devices, wantedLabel) {
    if (!wantedLabel) return null;
    for (const d of devices) {
        if (d.label === wantedLabel) return d.deviceId;
    }
    // try loose match if exact not found
    const loose = devices.find(
        (d) =>
            d.label && d.label.toLowerCase().includes(wantedLabel.toLowerCase())
    );
    return loose ? loose.deviceId : null;
}

/* ================================
   Boot
================================ */
(async function main() {
    setStatus("Loading…");

    // 1) Create sessions
    const baseURL = new URL(".", location.href);
    if (ENABLE_DET) {
        const path = `${baseURL}models/${modelDetectKey}.onnx`;
        detSession = await ort.InferenceSession.create(path, {
            executionProviders: ["webgpu"],
            graphOptimizationLevel: "all",
        });
    }
    if (ENABLE_POSE) {
        const path = `${baseURL}models/${modelPoseKey}.onnx`;
        poseSession = await ort.InferenceSession.create(path, {
            executionProviders: ["webgpu"],
            graphOptimizationLevel: "all",
        });
    }

    // 2) WebGPU device (for optional GPU-bound NMS fetches)
    device =
        ort.env.webgpu?.device ||
        (navigator.gpu &&
            (await navigator.gpu.requestAdapter()) &&
            (await (await navigator.gpu.requestAdapter()).requestDevice()));

    // 3) Load detect decoder/NMS only if detect session present
    if (detSession) {
        const nmsPath = `${baseURL}yolo-decoder.onnx`;
        try {
            detNmsSession = await ort.InferenceSession.create(nmsPath, {
                executionProviders: ["webgpu"],
                graphOptimizationLevel: "all",
            });
            if (device) prepareGpuFetchesForNms(detNmsSession, DET_TOPK);
        } catch (e) {
            console.warn("Detect decoder not available; using JS NMS.", e);
            detNmsSession = null;
        }
    }

    // 4) WebSocket connection
    ws = new WebSocket(`ws://localhost:${wsPort}`);
    ws.binaryType = "arraybuffer";
    ws.onopen = async () => {
        console.log("WebSocket connected");
        ws.send(JSON.stringify({ loaded: true }));

        // Send initial webcam list (labels may be empty before permission)
        const devices = await listWebcamDevices();
        const deviceLabels = devices.map((d) => d.label);
        ws.send(JSON.stringify({ webcamDevices: deviceLabels }));

        setStatus(USE_BINARY ? "Ready (binary)" : "Ready (webcam)");
    };
    ws.onerror = () => {
        setStatus(
            `Error: WebSocket connection to 'ws://localhost:${wsPort}/' failed.`
        );
    };
    ws.onclose = (event) => {
        setStatus(
            `🔌 Connection closed (code ${event.code}) on port ${wsPort}`
        );
    };

    if (USE_BINARY) {
        // -------- Path A: Binary frames (no RAF) --------
        ws.onmessage = (ev) => {
            if (!(ev.data instanceof ArrayBuffer)) return;
            const job = parseHeader(ev.data);
            if (!job) return;
            if (job.H !== INPUT_H || job.W !== INPUT_W) return; // expect 640x640
            latestBinaryJob = job;
            pumpBinary();
        };
        return; // Binary path done
    }

    // -------- Path B: Webcam + RAF --------
    const video = ensureVisibleVideo();

    // Try to choose device by label if provided
    const initialDevices = await listWebcamDevices();

    const exactId = findDeviceIdByLabel(initialDevices, WEBCAM_LABEL);

    console.log("WEBCAM", WEBCAM_LABEL, exactId, initialDevices);
    const constraints = exactId
        ? {
              video: {
                  deviceId: { exact: exactId },
                  width: { ideal: window.innerWidth },
                  height: { ideal: window.innerHeight },
              },
          }
        : {
              video: {
                  width: { ideal: window.innerWidth },
                  height: { ideal: window.innerHeight },
              },
          };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        const vfc = (now, metadata) => {
            if (metadata && typeof metadata.mediaTime === "number") {
                if (ws?.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ tick: metadata.mediaTime }));
                }
                lastVideoMediaTime = metadata.mediaTime; // seconds
            }
            video.requestVideoFrameCallback(vfc);
        };
        video.requestVideoFrameCallback(vfc);

        // Kick off RAF loop once metadata is ready
        video.onloadedmetadata = () => {
            requestAnimationFrame(() => rafLoop(video));
        };

        // After permission, labels become available; re-send a nicer list
        // const devicesAfter = await listWebcamDevices();
        // if (ws && ws.readyState === WebSocket.OPEN) {
        //   ws.send(JSON.stringify({ webcamDevices: devicesAfter }));
        // }

        // Kick off RAF loop once metadata is ready
        video.onloadedmetadata = () => {
            requestAnimationFrame(() => rafLoop(video));
        };
    } catch (err) {
        console.error("getUserMedia failed:", err);
        setStatus(`Camera error: ${err?.message || err}`);
    }
})();
