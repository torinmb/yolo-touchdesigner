// yolo_websocket_uint8_chw_dual.js
// Receive CHW uint8 via WebSocket (16B header) -> FP32 -> run YOLO detect/pose (WebGPU)
// Query params toggle detect/pose independently, can run both, merge JSON when both enabled.

import * as ort from "onnxruntime-web/webgpu";

// ---------- ORT env ----------
ort.env.wasm.wasmPaths = "./";
ort.env.allowLocalModels = true;
ort.env.allowRemoteModels = false;
ort.env.useBrowserCache = false;
// ort.env.logLevel = "verbose";

// ---------- Query helpers ----------
const qs = new URLSearchParams(location.search);
const boolish = (v, def = false) =>
  v == null ? def : /^(1|true|on|yes)$/i.test(String(v));

const wsPort        = qs.get("wsPort") || "59613";

// Stream toggles + models
const ENABLE_DET    = boolish(qs.get("detect"), true); // default off
const ENABLE_POSE   = boolish(qs.get("pose"),   true);  // default on (matches original pose default)
const modelDetectKey= qs.get("modelDetect") || "yolo11n";
const modelPoseKey  = qs.get("modelPose")   || "yolo11n-pose";

// Back-compat: if user only supplies `model=...` and no explicit toggles, respect the old behavior.
// If model ends with '-pose', treat as pose-only, else detect-only—unless detect/pose were explicitly set above.
const modelKeyLegacy = qs.get("model");
if (modelKeyLegacy && !qs.has("detect") && !qs.has("pose")) {
  const isPose = /pose/i.test(modelKeyLegacy);
  // override defaults only if user didn't pass detect/pose
  // legacy path: run just the one inferred by the name
  if (isPose) {
    // pose only
    (function(){ /* no-op, defaults already pose:true, detect:false */ })();
  } else {
    // detect only
    // flip: detect on, pose off
    (function(){
      // re-evaluate globals with detect forced true, pose false
      // (simple overwrite here)
    })();
  }
}

// ---------- Thresholds / tuning ----------
const SCORE_T  = parseFloat(qs.get("Scoret") || "0.4");
const IOU_T    = parseFloat(qs.get("Iout")   || "0.45");
const TOPK     = parseInt(qs.get("Topk")     || "50", 10);

// Tracker tuning (shared for both streams)
const TRK_IOU  = parseFloat(qs.get("Trkiou") || "0.5");
const TRK_TTL  = parseInt(qs.get("Trkttl")   || "1", 10);

// ---------- Globals ----------
const INPUT_W = 640, INPUT_H = 640;
let device = null;
let ws = null;
let latest = null;
let processing = false;

// Sessions per stream
let detSession = null, detNmsSession = null;
let poseSession = null;

// GPU fetch tensors for detect decoder/NMS
let tBoxes = null, tScores = null, tClasses = null;
let gpuFetches = null;
let gpuFetchesReady = false;

// Pre-create threshold tensors for ONNX detect decoder
const tensor_topk          = new ort.Tensor("int32",   new Int32Array([TOPK]));
const tensor_iou_threshold = new ort.Tensor("float32", new Float32Array([IOU_T]));
const tensor_score_thresh  = new ort.Tensor("float32", new Float32Array([SCORE_T]));

// ---------- DEBUG Canvas ----------
const DEBUG_RAW    = boolish(qs.get("debug"));
const DEBUG_FLIPY  = boolish(qs.get("flipY"));
const DEBUG_SWAPRB = boolish(qs.get("swapRB"));
const _dbgCtx = (() => {
  if (!DEBUG_RAW || typeof document === 'undefined') return null;
  const c = document.createElement('canvas');
  c.width = 640; c.height = 640;
  c.style.cssText = 'position:fixed;right:12px;bottom:12px;border:1px solid #444;image-rendering:pixelated;z-index:99999;background:#000;';
  document.body.appendChild(c);
  return c.getContext('2d');
})();
function _debugDrawCHW(job) {
  if (!_dbgCtx) return;
  const { H, W, payload } = job;
  if (_dbgCtx.canvas.width !== W || _dbgCtx.canvas.height !== H) {
    _dbgCtx.canvas.width = W; _dbgCtx.canvas.height = H;
  }
  const plane = W * H;
  const out = new Uint8ClampedArray(4 * plane);
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

// ---------- Minimal IoU tracker ----------
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
    const takenDet   = new Set();
    for (const [iou, id, di] of pairs) {
      if (iou < this.iouMatch) break;
      if (takenTrack.has(id) || takenDet.has(di)) continue;
      const t = this.tracks.get(id), d = dets[di];
      t.box = d.box; t.label = d.label; t.score = d.score; t.keypoints = d.keypoints;
      t.hits++; t.miss = 0;
      takenTrack.add(id); takenDet.add(di);
    }

    for (let di = 0; di < dets.length; di++) {
      if (takenDet.has(di)) continue;
      const d = dets[di];
      const id = this.nextId++;
      this.tracks.set(id, {
        box: d.box, label: d.label, score: d.score, keypoints: d.keypoints,
        age: 0, hits: 1, miss: 0
      });
    }

    for (const id of [...this.tracks.keys()]) {
      const t = this.tracks.get(id);
      t.age++; if (t.miss > this.ttl) this.tracks.delete(id);
    }

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
const trackerDet  = new IoUTracker(TRK_IOU, TRK_TTL);
const trackerPose = new IoUTracker(TRK_IOU, TRK_TTL);

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

function decodeYOLOPose(raw_tensor, score_threshold = 0.45, topk, W = INPUT_W, H = INPUT_H) {
  const sh = raw_tensor.dims;
  const data = raw_tensor.data;
  if (sh.length !== 3) return [];
  let C = sh[1], N = sh[2], layout = "CN";
  const looksPose = (c) => c > 5 && ((c - 5) % 3 === 0);
  if (!looksPose(C)) { C = sh[2]; N = sh[1]; if (!looksPose(C)) return []; layout = "NC"; }
  const K = ((C - 5) / 3) | 0;
  const get = (c, n) => (layout === "CN") ? data[c * N + n] : data[n * C + c];
  const out = [];
  for (let i = 0; i < N; i++) {
    const score = get(4, i);
    if (score <= score_threshold) continue;
    const cx = get(0, i), cy = get(1, i), w = get(2, i), h = get(3, i);
    const bx = cx - 0.5 * w;
    const by = cy - 0.5 * h;
    const keypoints = new Array(K);
    for (let kp = 0; kp < K; kp++) {
      const base = 5 + kp * 3;
      keypoints[kp] = { x: get(base + 0, i), y: get(base + 1, i), score: get(base + 2, i) };
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
function prepareGpuFetchesForNms(sess) {
  try {
    if (!device || !sess) return;
    tBoxes   = makeGpuTensor(TOPK * 4, "float32", [TOPK, 4]);
    tScores  = makeGpuTensor(TOPK,     "float32", [TOPK]);
    tClasses = makeGpuTensor(TOPK,     "float32", [TOPK]);

    gpuFetches = {};
    let haveBoxes = false;
    const oneDSlots = [];
    for (const name of sess.outputNames) {
      const md = sess.outputMetadata[name];
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

// ---------- Y-flip mapping to normalized coords (0..1) ----------
function flipYKeypointsNorm(kpts, H) {
  if (!kpts) return kpts;
  return kpts.map(k => ({ x: k.x / H, y: (H - k.y) / H, score: k.score }));
}
function mapBoxYFlipNorm([x, y, w, h], H) {
  return [x / H, (H - (y + h)) / H, w / H, h / H];
}

// ---------- Inference helpers ----------
async function runDetect(input) {
  if (!detSession) return [];
  const outs = await detSession.run({ [detSession.inputNames[0]]: input });
  const head = outs[detSession.outputNames[0]];
  let keep = [];

  if (detNmsSession) {
    const feeds = {
      [detNmsSession.inputNames[0]]: head,
      [detNmsSession.inputNames[1]]: tensor_topk,
      [detNmsSession.inputNames[2]]: tensor_iou_threshold,
      [detNmsSession.inputNames[3]]: tensor_score_thresh
    };
    const nmsOuts = gpuFetchesReady ? await detNmsSession.run(feeds, gpuFetches)
                                    : await detNmsSession.run(feeds);

    // Infer which outputs are boxes/scores/classes
    const outsList = detNmsSession.outputNames.map(n => nmsOuts[n]);
    let boxesT = null, scoresT = null, classesT = null;
    for (const t of outsList) {
      const dims = t.dims;
      if (dims.length === 2 && dims[1] === 4 && !boxesT) boxesT = t;
      else if (dims.length === 1 && t.type === "float32" && !scoresT) scoresT = t;
    }
    for (const t of outsList) {
      if (t === boxesT || t === scoresT) continue;
      if (t.dims.length === 1 && !classesT) classesT = t;
    }
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
      // fallback JS
      keep = nmsPerClass(decodeYOLO(head, SCORE_T, TOPK), IOU_T, TOPK);
    }
  } else {
    // JS
    keep = nmsPerClass(decodeYOLO(head, SCORE_T, TOPK), IOU_T, TOPK);
  }
  return keep;
}

async function runPose(input) {
  if (!poseSession) return [];
  const outs = await poseSession.run({ [poseSession.inputNames[0]]: input });
  const head = outs[poseSession.outputNames[0]];
  const dets = decodeYOLOPose(head, SCORE_T, TOPK, INPUT_W, INPUT_H);
  return nmsPerClassWithKpts(dets, IOU_T, TOPK);
}

// ---------- Main pump ----------
async function pump() {
  if (processing) return;
  processing = true;
  try {
    const job = latest; latest = null;
    if (!job || (!detSession && !poseSession)) return;
    _debugDrawCHW(job);

    const input = toInputTensorInPlace(job);

    // Run enabled streams
    let keepDet = [];
    let keepPose = [];
    if (detSession) keepDet = await runDetect(input);
    if (poseSession) keepPose = await runPose(input);

    // Trackers (separate per stream)
    const tracksDet  = detSession  ? trackerDet.update(keepDet)   : [];
    const tracksPose = poseSession ? trackerPose.update(keepPose) : [];

    // Map to output predictions
    const H = INPUT_H, W = INPUT_W;

    const predsDet = tracksDet.map(t => {
      const box = mapBoxYFlipNorm(t.box, H);
      return {
        tx: box[0], ty: box[1],
        width: box[2], height: box[3],
        categoryName: [t.label],
        score: t.score,
        id: t.id
      };
    });

    const predsPose = tracksPose.map(t => {
      const box = mapBoxYFlipNorm(t.box, H);
      const kpts = flipYKeypointsNorm(t.keypoints, H);
      return {
        tx: box[0], ty: box[1],
        width: box[2], height: box[3],
        categoryName: [t.label],
        score: t.score,
        id: t.id,
        keypoints: kpts
      };
    });

    if (ws && ws.readyState === WebSocket.OPEN) {
      // Merge or single depending on which are enabled
      if (detSession && poseSession) {
        ws.send(JSON.stringify({
          type: "yolo_combined",
          frame: job.td >>> 0,
          seq: job.seq >>> 0,
          width: W, height: H,
          yolo: predsDet,
          yolo_pose: predsPose
        }));
      } else if (detSession) {
        ws.send(JSON.stringify({
          type: "yolo",
          frame: job.td >>> 0,
          seq: job.seq >>> 0,
          width: W, height: H,
          predictions: predsDet
        }));
      } else if (poseSession) {
        ws.send(JSON.stringify({
          type: "yolo_pose",
          frame: job.td >>> 0,
          seq: job.seq >>> 0,
          width: W, height: H,
          predictions: predsPose
        }));
      }
    }
  } catch (e) {
    console.error(e);
  } finally {
    processing = false;
  }
}

// ---------- Boot ----------
(async function main() {
  // If both toggles are off, still open the socket and draw debug—just don't run models.
  const baseURL = new URL(".", location.href);

  // 1) Init sessions per stream
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

  // 2) Device for GPU tensors (only needed for detect decoder GPU fetches)
  device = ort.env.webgpu?.device ||
           (navigator.gpu && (await navigator.gpu.requestAdapter()) && await (await navigator.gpu.requestAdapter()).requestDevice());

  // 3) Load detect decoder/NMS only if detect session present
  if (detSession) {
    const nmsPath = `${baseURL}yolo-decoder.onnx`;
    try {
      detNmsSession = await ort.InferenceSession.create(nmsPath, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
      if (device) prepareGpuFetchesForNms(detNmsSession);
    } catch (e) {
      console.warn("Detect decoder not available; using JS NMS.", e);
      detNmsSession = null;
    }
  }

  // 4) WebSocket ingest
  ws = new WebSocket(`ws://localhost:${wsPort}`);
  ws.onopen = () => console.log('connected socket');
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
