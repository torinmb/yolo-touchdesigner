// yolo_websocket_uint8_chw.js (auto-detect detect/pose)
// Receive CHW uint8 via WebSocket (16B header), normalize to FP32,
// run YOLO on WebGPU, detect task type (detect vs pose), decode accordingly,
// optional ONNX decoder/NMS (GPU) for DETECT models, JS fallback for POSE,
// minimal IoU tracker, send JSON via WebSocket. Assumes 640x640 input.

import * as ort from "onnxruntime-web/webgpu";

// ---------- ORT env ----------
ort.env.wasm.wasmPaths = "./";
ort.env.allowLocalModels = true;
ort.env.allowRemoteModels = false;
ort.env.useBrowserCache = false;
// ort.env.logLevel = "verbose";

// ---------- Params ----------
const qs       = new URLSearchParams(location.search);
const wsPort   = qs.get("wsPort") || "59172";
const modelKey = qs.get("model")  || "yolo11n-pose";

const SCORE_T  = parseFloat(qs.get("Scoret") || "0.4");
const IOU_T    = parseFloat(qs.get("Iout")   || "0.45");
const TOPK     = parseInt(qs.get("Topk")     || "50", 10);

// Optional tracker tuning via query
const TRK_IOU  = parseFloat(qs.get("Trkiou") || "0.5");
const TRK_TTL  = parseInt(qs.get("Trkttl")   || "1", 10);

// ---------- Globals ----------
const INPUT_W = 640, INPUT_H = 640;
let yoloSession = null, nmsSession = null, ws = null;
let latest = null, processing = false;
let device = null;

// Probe results (autodetected model type/shapes)
let probe = { task: "detect", C: 0, N: 0, axisC: 2, poseK: 0 }; // filled after load

// Preallocated GPU fetch tensors (for DETECT decoder/NMS only)
let tBoxes = null, tScores = null, tClasses = null;
let gpuFetches = null;
let gpuFetchesReady = false;

// Pre-create threshold tensors for ONNX decoder (detect only)
const tensor_topk          = new ort.Tensor("int32",   new Int32Array([TOPK]));
const tensor_iou_threshold = new ort.Tensor("float32", new Float32Array([IOU_T]));
const tensor_score_thresh  = new ort.Tensor("float32", new Float32Array([SCORE_T]));


/** DEBUG Canvas */
const _qs = (typeof qs !== 'undefined' && qs) ? qs : new URLSearchParams(location.search);
const DEBUG_RAW    = /^(1|true)$/i.test(_qs.get('debug')  || '');
const DEBUG_FLIPY  = /^(1|true)$/i.test(_qs.get('flipY')  || '');
const DEBUG_SWAPRB = /^(1|true)$/i.test(_qs.get('swapRB') || '');

const _dbgCtx = (() => {
  if (!DEBUG_RAW || typeof document === 'undefined') return null;
  const c = document.createElement('canvas');
  c.width = 640; c.height = 640;
  c.style.cssText = 'position:fixed;right:12px;bottom:12px;border:1px solid #444;image-rendering:pixelated;z-index:99999;background:#000;';
  document.body.appendChild(c);
  return c.getContext('2d');
})();

/** Rebuilds CHW uint8 payload (R..G..B..) into an RGBA image on a tiny canvas. */
function _debugDrawCHW(job) {
  if (!_dbgCtx) return;
  const { H, W, payload } = job;
  if (_dbgCtx.canvas.width !== W || _dbgCtx.canvas.height !== H) {
    _dbgCtx.canvas.width = W; _dbgCtx.canvas.height = H;
  }
  const plane = W * H;
  const out = new Uint8ClampedArray(4 * plane);

  // Optional view-only transforms (do not affect inference)
  for (let y = 0; y < H; y++) {
    const ySrc = DEBUG_FLIPY ? (H - 1 - y) : y;
    for (let x = 0; x < W; x++) {
      const idx = ySrc * W + x;
      const p = (y * W + x) * 4;
      let r = payload[0 * plane + idx];
      let g = payload[1 * plane + idx];
      let b = payload[2 * plane + idx];
      if (DEBUG_SWAPRB) { const t = r; r = b; b = t; }
      out[p + 0] = r; out[p + 1] = g; out[p + 2] = b; out[p + 3] = 255;
    }
  }
  _dbgCtx.putImageData(new ImageData(out, W, H), 0, 0);
}

// ---------- Minimal IoU tracker (keeps matched tracks only) ----------
class IoUTracker {
  constructor(iouMatch = 0.5, ttl = 1) {
    this.iouMatch = iouMatch;
    this.ttl = ttl;
    this.nextId = 1;
    this.tracks = new Map(); // id -> {box, label, score, keypoints?, age, hits, miss}
  }
  update(dets) {
    const ids = [...this.tracks.keys()];
    for (const id of ids) this.tracks.get(id).miss++;

    // Greedy IoU matching
    const pairs = [];
    for (let di = 0; di < dets.length; di++) {
      for (const id of ids) {
        const iou = this._iou(this.tracks.get(id).box, dets[di].box);
        pairs.push([iou, id, di]);
      }
    }
    pairs.sort((a, b) => b[0] - a[0]);

    const takenTrack = new Set();
    const takenDet   = new Set();
    for (const [iou, id, di] of pairs) {
      if (iou < this.iouMatch) break;
      if (takenTrack.has(id) || takenDet.has(di)) continue;
      const t = this.tracks.get(id), d = dets[di];
      t.box = d.box; t.label = d.label; t.score = d.score; t.keypoints = d.keypoints;
      t.hits++; t.miss = 0;
      takenTrack.add(id); takenDet.add(di);
    }

    // New tracks
    for (let di = 0; di < dets.length; di++) {
      if (takenDet.has(di)) continue;
      const d = dets[di];
      const id = this.nextId++;
      this.tracks.set(id, {
        box: d.box, label: d.label, score: d.score, keypoints: d.keypoints,
        age: 0, hits: 1, miss: 0
      });
    }

    // Drop stale
    for (const id of [...this.tracks.keys()]) {
      const t = this.tracks.get(id);
      t.age++; if (t.miss > this.ttl) this.tracks.delete(id);
    }

    // Emit only matched this frame
    return [...this.tracks.entries()]
      .filter(([_, t]) => t.miss === 0)
      .map(([id, t]) => ({ id, box: t.box, label: t.label, score: t.score, keypoints: t.keypoints }));
  }
  _iou(a, b) {
    const [ax, ay, aw, ah] = a, [bx, by, bw, bh] = b;
    const ax2 = ax + aw, ay2 = ay + ah, bx2 = bx + bw, by2 = by + bh;
    const ix = Math.max(ax, bx), iy = Math.max(ay, by);
    const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
    const iw = Math.max(0, ix2 - ix), ih = Math.max(0, iy2 - iy);
    const inter = iw * ih;
    const uni = aw * ah + bw * bh - inter + 1e-9;
    return inter / uni;
  }
}
const tracker = new IoUTracker(TRK_IOU, TRK_TTL);

// ---------- Header parsing: <BBBBHHII> ----------
// type(10), dtype(1=u8), layout(1=CHW), pad, H, W, seq, td_frame
function parseHeader(buf) {
  if (buf.byteLength < 16) return null;
  const dv = new DataView(buf);
  const type   = dv.getUint8(0);
  const dtype  = dv.getUint8(1);
  const layout = dv.getUint8(2);
  const H      = dv.getUint16(4, true);
  const W      = dv.getUint16(6, true);
  const seq    = dv.getUint32(8, true);
  const td     = dv.getUint32(12, true);
  if (type !== 10 || dtype !== 1 || layout !== 1) return null; // require u8 CHW
  const payload = new Uint8Array(buf, 16); // length = 3*H*W
  return { H, W, seq, td, payload };
}

// ---------- Input tensor (reused each frame) ----------
const NUM = 3 * INPUT_H * INPUT_W;
const f32InputBuffer = new Float32Array(NUM);
const inputTensor = new ort.Tensor("float32", f32InputBuffer, [1,3,INPUT_H,INPUT_W]);

function toInputTensorInPlace(job) {
  const u8 = job.payload; // length == NUM
  for (let i = 0; i < NUM; i++) f32InputBuffer[i] = u8[i] * (1/255);
  return inputTensor;
}

// ---------- Decoders ----------
// Generic detection decoder (your original)
function decodeYOLO(tensor, thr, topk) {
  const d = tensor.data, sh = tensor.dims;
  let num, dim, T = false;
  if (sh.length === 3) {
    if (sh[1] === 84)      { num = sh[2]; dim = sh[1]; T = false; }
    else if (sh[2] === 84) { num = sh[1]; dim = sh[2]; T = true;  }
    else                   { num = sh[sh.length-1]; dim = sh[sh.length-2]; T = (sh[2] === dim); }
  } else if (sh.length === 2) { num = sh[0]; dim = sh[1] - 1; T = true; }
  else return [];

  const W = INPUT_W, H = INPUT_H;
  const sb = T ? dim : 1;
  const sbx = T ? 1 : num;

  const out = [];
  for (let i = 0; i < num; i++) {
    const x = d[0 * sbx + i * sb];
    const y = d[1 * sbx + i * sb];
    const w = d[2 * sbx + i * sb];
    const h = d[3 * sbx + i * sb];

    let bestC = -1, bestS = -1;
    for (let c = 4; c < dim; c++) {
      const s = d[c * sbx + i * sb];
      if (s > bestS) { bestS = s; bestC = c - 4; }
    }
    if (bestS < thr) continue;

    let bx, by, bw, bh;
    if (w > 0 && h > 0 && w <= W * 2 && h <= H * 2) {
      bw = w; bh = h; bx = x - w / 2; by = y - h / 2;
    } else {
      const x2 = w, y2 = h; bx = x; by = y; bw = x2 - x; bh = y2 - y;
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

// --- replace your decodeYOLOPose with this exact version ---
function decodeYOLOPose(raw_tensor, score_threshold = 0.45, topk, W = INPUT_W, H = INPUT_H) {
  const sh = raw_tensor.dims;
  const data = raw_tensor.data;

  if (sh.length !== 3) return [];

  // Prefer Ultralytics-style layout: [1, C=5+3K, N]
  let C = sh[1], N = sh[2], layout = "CN";
  const looksPose = (c) => c > 5 && ((c - 5) % 3 === 0);

  if (!looksPose(C)) {
    // Fallback: [1, N, C]
    C = sh[2];
    N = sh[1];
    if (!looksPose(C)) return []; // not a pose head
    layout = "NC";
  }

  const K = ((C - 5) / 3) | 0;

  // Accessor that treats the buffer as [C, N]
  const get = (c, n) => (layout === "CN") ? data[c * N + n] : data[n * C + c];

  const out = [];
  for (let i = 0; i < N; i++) {
    const score = get(4, i);
    if (score <= score_threshold) continue;

    // bbox center format to tlwh (same as the working example)
    const cx = get(0, i), cy = get(1, i), w = get(2, i), h = get(3, i);
    const bx = cx - 0.5 * w;
    const by = cy - 0.5 * h;

    // keypoints: base = 5 + kp*3, then x, y, conf each stride across N
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


// Class-agnostic per-class NMS (detect)
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
        if (iou(a.box, k.box) > iouThr) { ok = false; break; }
      }
      if (ok) { keep.push(a); if (keep.length >= topk) break; }
    }
    keepAll.push(...keep);
  }
  keepAll.sort((a, b) => b.score - a.score);
  if (keepAll.length > topk) keepAll.length = topk;
  return keepAll;

  function iou(a, b) {
    const [ax, ay, aw, ah] = a, [bx, by, bw, bh] = b;
    const ax2 = ax + aw, ay2 = ay + ah, bx2 = bx + bw, by2 = by + bh;
    const ix = Math.max(ax, bx), iy = Math.max(ay, by);
    const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
    const iw = Math.max(0, ix2 - ix), ih = Math.max(0, iy2 - iy);
    const inter = iw * ih;
    const uni = aw * ah + bw * bh - inter + 1e-9;
    return inter / uni;
  }
}

// Same as above but preserves `keypoints` (pose)
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
        if (iou(a.box, k.box) > iouThr) { ok = false; break; }
      }
      if (ok) { keep.push(a); if (keep.length >= topk) break; }
    }
    keepAll.push(...keep);
  }
  keepAll.sort((a, b) => b.score - a.score);
  if (keepAll.length > topk) keepAll.length = topk;
  return keepAll;

  function iou(a, b) {
    const [ax, ay, aw, ah] = a, [bx, by, bw, bh] = b;
    const ax2 = ax + aw, ay2 = ay + ah, bx2 = bx + bw, by2 = by + bh;
    const ix = Math.max(ax, bx), iy = Math.max(ay, by);
    const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
    const iw = Math.max(0, ix2 - ix), ih = Math.max(0, iy2 - iy);
    const inter = iw * ih;
    const uni = aw * ah + bw * bh - inter + 1e-9;
    return inter / uni;
  }
}

// ---------- Helper: pick decoder outputs (boxes/scores/classes) ----------
function pickDecoderOutputs(nmsOuts, nmsSession) {
  const outs = nmsSession.outputNames.map(name => nmsOuts[name]);
  let boxesT = null, scoresT = null, classesT = null;

  for (const t of outs) {
    const dims = t.dims;
    if (dims.length === 2 && dims[1] === 4 && !boxesT) boxesT = t;
    else if (dims.length === 1 && t.type === "float32" && !scoresT) scoresT = t;
  }
  // Classes: remaining 1D tensor
  for (const t of outs) {
    if (t === boxesT || t === scoresT) continue;
    if (t.dims.length === 1 && !classesT) classesT = t;
  }
  return { boxesT, scoresT, classesT };
}

// ---------- GPU fetch tensor helpers (detect decoder only) ----------
function makeGpuTensor(nelem, dtype, dims) {
  const bytesPerElem = (dtype === "float16") ? 2 : 4;
  const bytes = nelem * bytesPerElem;
  const buf = device.createBuffer({
    size: Math.ceil(bytes / 16) * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  return ort.Tensor.fromGpuBuffer(buf, { dataType: dtype, dims });
}

function prepareGpuFetchesForNms() {
  try {
    if (!device) return;
    tBoxes   = makeGpuTensor(TOPK * 4, "float32", [TOPK, 4]);
    tScores  = makeGpuTensor(TOPK,     "float32", [TOPK]);
    tClasses = makeGpuTensor(TOPK,     "float32", [TOPK]);

    gpuFetches = {};
    let haveBoxes = false;
    const oneDSlots = [];

    for (const name of nmsSession.outputNames) {
      const md = nmsSession.outputMetadata[name];
      const dims = md?.dimensions || [];
      if (dims.length === 2 && !haveBoxes) { gpuFetches[name] = tBoxes; haveBoxes = true; }
      else if (dims.length === 1) { oneDSlots.push(name); }
      else { oneDSlots.push(name); }
    }
    if (oneDSlots.length >= 2) {
      gpuFetches[oneDSlots[0]] = tScores;
      gpuFetches[oneDSlots[1]] = tClasses;
    } else {
      gpuFetches = null; gpuFetchesReady = false; return;
    }
    gpuFetchesReady = true;
  } catch (e) {
    console.warn("Failed to prebind GPU fetches; falling back to default:", e);
    gpuFetches = null; gpuFetchesReady = false;
  }
}

// ---------- Model probing (auto-detect task) ----------
async function inferTaskAndShapes(sess) {
  const outName = sess.outputNames[0];
  let dims = sess.outputMetadata[outName]?.dimensions || [];
  if (!dims || dims.length === 0 || dims.includes(-1) || dims.includes("dynamic")) {
    const dummy = new ort.Tensor("float32", new Float32Array(1*3*INPUT_H*INPUT_W).fill(0), [1,3,INPUT_H,INPUT_W]);
    const outs  = await sess.run({ [sess.inputNames[0]]: dummy });
    const t     = outs[sess.outputNames[0]];
    dims        = t.dims;
  }
  // head is typically [1, C, N] or [1, N, C]
  const candidates = dims.filter(v => v > 8 && v < 512);
  const C = candidates.find(v => v === 56 || v === 84 || v === 85) ?? candidates[0] ?? 0;
  const axisC = dims.indexOf(C);
  const N     = dims[axisC === 1 ? 2 : 1];

  let task = "detect";
  let poseK = 0;
  if (C === 56) { task = "pose"; poseK = 17; }
  else if (C > 5 && (C - 5) % 3 === 0) { task = "pose"; poseK = (C - 5) / 3; }

  return { task, C, N, axisC, poseK };
}


function flipYKeypointsNorm(kpts, H) {
  if (!kpts) return kpts;
  return kpts.map(k => ({ x: k.x/H, y: (H - k.y)/H, score: k.score }));
}

        
function mapBoxYFlipNorm([x, y, w, h], H) {

      return [x/H, (H - (y + h))/H, w/H, h/H];
}


// ---------- Main pump ----------
async function pump() {
  if (processing) return;
  processing = true;
  try {
    const job = latest; latest = null;
    if (!job || !yoloSession) return;
    _debugDrawCHW(job)

    const input = toInputTensorInPlace(job);
    const yoloOuts = await yoloSession.run({ [yoloSession.inputNames[0]]: input });
    const yoloHead = yoloOuts[yoloSession.outputNames[0]];

    let keep = [];

    if (probe.task === "detect" && nmsSession) {
      // GPU decoder/NMS path (detect only)
      const nmsFeeds = {
        [nmsSession.inputNames[0]]: yoloHead,
        [nmsSession.inputNames[1]]: tensor_topk,
        [nmsSession.inputNames[2]]: tensor_iou_threshold,
        [nmsSession.inputNames[3]]: tensor_score_thresh
      };

      let nmsOuts;
      if (gpuFetchesReady) nmsOuts = await nmsSession.run(nmsFeeds, gpuFetches);
      else                 nmsOuts = await nmsSession.run(nmsFeeds);

      const { boxesT, scoresT, classesT } = pickDecoderOutputs(nmsOuts, nmsSession);

      if (boxesT && scoresT && classesT) {
        const boxes  = (boxesT.getData)  ? await boxesT.getData()  : boxesT.data;
        const scores = (scoresT.getData) ? await scoresT.getData() : scoresT.data;
        const clsRaw = (classesT.getData)? await classesT.getData(): classesT.data;

        const N = scores.length;
        for (let i = 0; i < N; i++) {
          const off = i * 4;
          const x1 = boxes[off+0], y1 = boxes[off+1];
          const x2 = boxes[off+2], y2 = boxes[off+3];
          const w = Math.max(1, x2 - x1), h = Math.max(1, y2 - y1);
          const clsIdx = (typeof clsRaw[i] === "bigint") ? Number(clsRaw[i]) : (clsRaw[i] | 0);
          keep.push({ box: [x1, y1, w, h], label: clsIdx, score: scores[i] });
        }
      } else {
        // Fallback to JS detect
        const dets = decodeYOLO(yoloHead, SCORE_T, TOPK);
        keep = nmsPerClass(dets, IOU_T, TOPK);
      }
    } else {
      // JS path: pose or detect (if no decoder)
      if (probe.task === "pose") {
        const dets = decodeYOLOPose(yoloHead, SCORE_T, TOPK, INPUT_W, INPUT_H);
        keep = nmsPerClassWithKpts(dets, IOU_T, TOPK);
      } else {
        const dets = decodeYOLO(yoloHead, SCORE_T, TOPK);
        keep = nmsPerClass(dets, IOU_T, TOPK);
      }
    }

    const tracks = tracker.update(keep);

    if (ws && ws.readyState === WebSocket.OPEN) {
      const H = INPUT_H, W = INPUT_W;

      const preds = tracks.map(t => {
        let box = t.box;
        let kpts = t.keypoints;

        box = mapBoxYFlipNorm(box, H);
        if (probe.task === "pose" && kpts) kpts = flipYKeypointsNorm(kpts, H);


        return {
          tx: box[0], ty: box[1],
          width: box[2], height: box[3],
          categoryName: [t.label],
          score: t.score,
          id: t.id,
          ...(probe.task === "pose" ? { keypoints: kpts } : {})
        };
      });

      ws.send(JSON.stringify({
        type: probe.task === "pose" ? "yolo_pose" : "yolo",
        frame: job.td >>> 0,
        seq: job.seq >>> 0,
        width: W,
        height: H,
        predictions: preds
      }));
    }
  } catch (e) {
    console.error(e);
  } finally {
    processing = false;
  }
}

// ---------- Boot ----------
(async function main() {
  const baseURL   = new URL(".", location.href);
  const modelPath = `${baseURL}models/${modelKey}.onnx`;

  // 1) Initialize YOLO session
  yoloSession = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "all",
  });

  // 2) Ensure device available after first session
  device = ort.env.webgpu?.device || (navigator.gpu && (await navigator.gpu.requestAdapter()) && await (await navigator.gpu.requestAdapter()).requestDevice());

  // 3) Probe model to infer task/shapes
  probe = await inferTaskAndShapes(yoloSession);
  console.log("Model probe:", probe);

  // 4) If DETECT, try to load decoder/NMS (pose uses JS NMS here)
  if (probe.task === "detect") {
    const nmsPath = `${baseURL}yolo-decoder.onnx`; // your existing detect decoder
    try {
      nmsSession = await ort.InferenceSession.create(nmsPath, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
      if (device) prepareGpuFetchesForNms();
    } catch (e) {
      console.warn("Detect decoder not available; using JS NMS.", e);
      nmsSession = null;
    }
  } else {
    nmsSession = null; // pose uses JS decode + JS NMS
  }

  // 5) WebSocket ingest
  ws = new WebSocket(`ws://localhost:${wsPort}`);
  ws.onopen = () => { console.log('connected socket') };
  ws.binaryType = "arraybuffer";
  ws.onmessage = (ev) => {
    if (!(ev.data instanceof ArrayBuffer)) return;
    const job = parseHeader(ev.data);
    if (!job) return;
    if (job.H !== INPUT_H || job.W !== INPUT_W) return; // expect 640x640
    latest = job;
    pump();
  };
})();

// ---- Half-float helpers (kept if you want them later) ----
function f32ToF16Bits(val) {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = val;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let mant = x & 0x007fffff;
  let exp  = (x >>> 23) & 0xff;
  if (exp === 0xff) { // Inf/NaN
    const infNan = mant ? 0x7e00 : 0x7c00;
    return sign | infNan;
  }
  exp = exp - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x00800000) >>> (1 - exp);
    return sign | (mant + 0x00001000 + ((mant >>> 13) & 1)) >>> 13;
  }
  const rounded = mant + 0x00001000 + ((mant >>> 13) & 1);
  return sign | (exp << 10) | (rounded >>> 13);
}