# Performance audit (measured with the harness)

Apple GPU / Chrome WebGPU, 640x640 frames, YoloDemo.1.mov `--start 9 --stride 4`, lockstep replay
unless noted. "page" = frame received -> last result sent, from `harness/probe.js`
(shown in every `replay` report). "rtt" = TD `webSocketSendBinary` -> JSON result back.
Every proposed change below was run through `replay --against` / `ab`; outputs are listed as
bit-identical or not.

## Where the time goes

| scenario | rtt p50 | page | notes |
| --- | --- | --- | --- |
| det yolo26n | 22 ms | 17 ms | 381 `writeBuffer` + 31 `submit` per frame (ORT-web per-kernel dispatch), 3 readbacks |
| pose yolo26n | 24 | 19 | same |
| seg yolo26n | 26 | 22 | 3.3 MB proto tensor read back to the CPU and written back (`decodeYOLOSeg`) |
| depth yolo26n | 37 | 26 | 1.6 MB result; GPU-bound (~20 ms of GPU wait) |
| det+pose+seg+depth | 81 | 73 | sequential; ~sum of the single-model times |

`rtt - page` is the transport + TD cost: 4.5 ms for det (1.2 MB in, 7 KB out), 10-13 ms for depth.

## Measured findings, ranked by payoff

### 1. Frame flow control wastes TD time and spikes latency (TD side, ~free to fix)
`onWebSocketReceiveText` clears `busy` on the first line for *any* text, including the
`sync` reply the page sends back while busy. So TD never waits: it re-uploads every cooked
frame, and the page silently drops the stale one (`latestJob` mailbox).
`live` run, det+pose:

| flow control | results/s | browser-dropped | rtt p50 / max | TD work p95 |
| --- | --- | --- | --- | --- |
| today (any text clears busy) | 28.7 | 24 / 285 | 43 / **324 ms** | 15.7 ms |
| result clears busy, 1 in flight | 19.3 | 0 | 37 / 48 | 4.6 |
| result clears busy, **2 in flight** | 31.7 (source-limited) | 0 | 44 / 68 | 13.6* |

\*includes the harness's own float32 source conversion (about 3 ms of the 4.5 ms upload time).
Fix: count in-flight frames in `upload_frames.py` (skip when >= 2) and decrement only when a
`"type":"yolo*"` result arrives. This is what MediaPipe's `timersAck`/`segAck` do. Keep the
1 s busy timeout. 2 is better than 1 because a single frame in flight idles the GPU for the
result trip plus up to one 16 ms TD tick.

### 2. Depth result is the only fat payload: send it at model resolution (page side)
The depth model ends in a `Resize` from 160x160 to 640x640. Cutting that node (output
`[1,1,160,160]`, 100 KB instead of 1.6 MB) gave rtt 35.3 -> 24.8 ms (-30 %), and `runDepth`
already handles any size. TD upsamples with a bilinear Resize TOP. Transport alone costs about
4-5 ms per MB in each direction on loopback (measured: server->page 1.2 MB 4.8 ms,
page->server 1.6 MB 8.5 ms), so bytes matter more than memcpy numbers suggest.
Seg is already 160x160 (100 KB) for the same reason.

### 3. WebGPU graph capture removes the per-frame dispatch overhead (page side)
`enableGraphCapture: true` + `preferredOutputLocation: 'gpu-buffer'` (the input is already a
fixed GPU buffer). Depth: `writeBuffer` calls 339 -> 1 per frame, JS dispatch 5.8 -> 0.8 ms,
output bit-identical; page time only 25.8 -> 24.2 ms because depth is GPU-bound. For the
YOLO26 detectors it fails today ("all compute graph nodes have not been partitioned") because
the end-to-end head's `TopK`/`GatherElements` run on the CPU. Cutting the exported graph
before the TopK (outputs: boxes `[1,8400,4]`, `ReduceMax` score `[1,8400]`, `ArgMax` class) and
doing the top-300 in JS makes capture work: det page 17.1 -> 13.6 ms (-20 %), rtt 22 -> 17 ms,
predictions byte-identical on the test clip. Caveat: the model's own head can emit two classes for
one anchor; the ArgMax head emits the best one, so that case would differ. Needs an export
change (`exportModel.py`) plus a few lines in `runDetect`/`runPose`.
fp16 via `onnxconverter-common` was worse (37 ms, 14 readbacks, outputs shifted); an Ultralytics
`half=True` export is the right way to try it.

### 4. Not worth doing (measured)
- Running models concurrently with `Promise.all`: ORT-web throws `Session already started`.
- The CPU `TopK` readbacks are not the cost (3 readbacks either way); the readback time is
  mostly waiting for GPU work.
- Seg proto staying on the GPU would save the 3.3 MB read + 1.1 MB write per frame; estimated
  2-3 ms, not prototyped.

### 5. Upgrading onnxruntime-web 1.22 -> 1.30 (tested, no gain)
The repo resolves 1.22.0 (package.json says ^1.20); latest is 1.30.0. Same frames, outputs identical
for det/pose/seg/depth (`compare`: SAME, all maps bit-identical). rtt p50, two runs each, 1.22 vs 1.30:
det 21.5/23.3 vs 21.4/22.0, pose 23.3/26.7 vs 23.3/22.7, seg 27.7/26.0 vs 27.1/27.5, depth 39.3/38.9 vs
40.1/41.0, all four models 80.2/81.4 vs 85.3/88.0 (about 6 % slower). The only difference is that the CPU TopK
readbacks shrink from 96 KB/33.6 KB to 2.4 KB; `enableGraphCapture` on the stock yolo26n no longer errors
but also does not take effect (436 `writeBuffer`/frame, no speedup). Not worth upgrading for speed.

### 6. TD `numpyArray` is always floating point
Per the TOP class docs, numpyArray "data will always be in floating point, regardless of what the texture data format is on the GPU",
so `upload_frames.py` always takes the float32 path (the zero-copy uint8 branch can't trigger). Measured CPU cost of
that path: 1.45 ms float32 -> u8, 0.42 ms `arr.copy()`, 0.04 ms packet build; the 4.9 MB float download is on top
and can't be measured outside TD. `harness --source-format packed` (float32) is the realistic default.
Idea to test in TD: have the repack shader write the bytes as float bit patterns so the CPU only reinterprets (not
convert) and the download shrinks 2-4x. NaN/denormal bit patterns may not survive the GPU read, so pack 16 bits into
the mantissa of a float in [1,2).

## Does MediaPipe's lossless compression apply?
Measured on real frames/maps (browser `CompressionStream`, Python zlib, level 1):

| payload | raw | zlib | ratio | cost | verdict |
| --- | --- | --- | --- | --- | --- |
| TD -> page frame (u8 CHW, real video) | 1.2 MB | 7.4x | 4.6 ms TD + 1.9 ms page inflate | saves ~4 ms wire | net **+2 ms worse** on loopback; wins only over a real network or if pipelined |
| page -> TD depth fp32 | 1.6 MB | 1.4x | **24-30 ms** deflate | | useless: float mantissas are noise |
| depth fp32, byte-shuffled | 1.6 MB | 2.3x | 16 ms | | still too slow |
| depth as uint16 millimetres | 0.8 MB | 10x | 2 ms deflate, 0.7 ms inflate | saves ~8 ms wire | good, but **lossy** (0.5 mm), so use the 160x160 option first |
| seg 160x160 fp32 | 100 KB | 200x+ | 0.1 ms | | already tiny; not needed |

Conclusion: lossless compression is a poor fit here. The data that is big is either
incompressible (float depth) or crosses loopback where the wire costs less than the CPU spent
compressing. MediaPipe needed it because its masks were large and mostly flat. The wins are
sending fewer bytes (finding 2) and not over-sending (finding 1). If TD runs on another
machine, add MediaPipe-style capability negotiation (`segCapabilities` style) for frames.

## What MediaPipe did to keep a high-volume socket from throttling TD
From `5829a7f`, `c3f4939`, `b2c4b84`, `0660bc7`:
1. A **dedicated `/segmentation` socket in a Web Worker**, so a multi-MB mask never queues
   behind small control messages or behind inference on the page's main thread.
2. **Ack-based flow control**: `segAck` per committed mask with at most 2 packets in flight and
   an ack timeout; control results capped at 3 frames in flight via `timersAck`, with stale
   frames dropped instead of queued, and an unthrottled fallback if acks stop. The ack is
   keyed on a browser sequence number (`timersSeq`) because a running count in a callback
   global reset when the DAT recompiled and starved results to ~1 fps (fixed in `b2c4b84`).
3. **Optional zlib** negotiated per connection, skipped when incompressible and retried
   after a back-off.
4. **Commit on frame match**: the mask is held until its exact video frame is displayed, with a
   900 ms pending timeout, and the ack is sent after the commit.
5. **Send less**: timers only for new video frames or while a mask is in flight; output canvases
   resized only on change; a single `json.loads` per results DAT downstream.

YOLO has item 5's spirit but not 1-4: one socket carries 1.2 MB frames in and 1.6 MB maps out
(a result can sit behind a 1.6 MB depth packet), the only ack is the `busy` flag that sync
replies defeat (finding 1), and the harness shows a 324 ms stall under that policy. The ack-on-result
fix with a count is the YOLO analogue of `timersAck`/`segAck`; a second socket for depth/seg is
the analogue of `/segmentation` if maps ever grow again. Key the ack on `seq` rather than a
module global so recompiling `webserver1_callbacks.py` cannot desync it (the harness can test that:
`--reload-callbacks-at` exists in the MediaPipe harness and is a port candidate here).

## Reproduce
```
PY=harness/.venv/bin/python
$PY harness/run.py replay --features det --start 9 --stride 4 --frames 60 --warmup 5      # page timeline, readbacks
$PY harness/run.py live   --features det,pose --seconds 10 --start 9                      # flow control
$PY harness/run.py ab --base HEAD --root <checkout-with-change> --features depth ...      # equivalence
```
