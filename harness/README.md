# YOLO TD harness

Runs the whole pipeline (web app -> WebSocket -> TD callbacks -> DAT/TOP outputs)
without TouchDesigner, so an agent (or you) can change `src/` or `td_scripts/`
and know within seconds whether the output changed and how fast it is.

```
headless Chrome (WebGPU/Metal) ── page from Vite dev server (src/, no build)
      │  ws://localhost:PORT   binary u8 CHW frames  ──►  page    results (JSON + float32 maps) ◄──
      ▼
fake_td.py   network thread (aiohttp) ──queue──► "TD main thread"
                                                  ├─ td_scripts/webserver1_callbacks.py   (unmodified)
                                                  ├─ td_scripts/upload_frames.py          (unmodified, drives the frames)
                                                  └─ fake op('predictions'), op('segmentation_data'), op('depth_data'), ...
```

`td_stubs.py` provides `op`, `absTime`, `parent`, `run`, DAT/TOP fakes; the real
callback files are `exec`'d against them, so edits to them are tested directly.
`op('source')` is a fake TOP that hands `upload_frames.py` decoded video frames
in the layout `repack.glsl` produces (`--source-format packed|packed-u8|rgb`).

## Setup (once)

```bash
python3 -m venv harness/.venv
harness/.venv/bin/pip install aiohttp numpy pillow playwright onnxruntime
brew install ffmpeg          # decodes clips; uses your installed Google Chrome (no browser download)
npm i                        # the repo's own deps (vite, onnxruntime-web)
```

No clip ships with the repo. `--video` takes anything ffmpeg reads; by default the
first `YoloDemo.*.mov` in the repo root. Detections are sparse in some clips, so
pick a window with something in it: `--start 9 --stride 4` on `YoloDemo.1.mov`
has people/animals in about a third of the frames.

All commands: `harness/.venv/bin/python harness/run.py <command> ...`

## Which command when

| you want to know | command | typical time |
| --- | --- | --- |
| does everything still run (every model family) | `check` | ~16 s |
| did my change alter any output | `ab --base <git ref>` | ~10 s |
| did a refactor change *nothing* (bit-exact) | `replay --against X.jsonl --exact` | ~6 s |
| is the browser still right vs the model | `replay --vs-reference` | ~8 s |
| how fast is it, what does TD pay | `replay` (per frame) / `live` (realtime) | ~6 s / 15 s |
| does the packaged `.tox` path (VFS + `onHTTPRequest`) work | `replay --source dist --build` | ~10 s |

```bash
PY=harness/.venv/bin/python

# 1. smoke every model family: det/pose/det+pose/seg/depth/obb on v26, det/pose on v11, wasm
$PY harness/run.py check --start 9 --stride 4
$PY harness/run.py check --start 9 --stride 4 --out harness/out/base          # save a baseline...
$PY harness/run.py check --start 9 --stride 4 --against harness/out/base      # ...verify against it later

# 2. A/B a git ref against the working tree (worktree + shared node_modules, same frames, both compared)
$PY harness/run.py ab --base master --features det,pose,seg --start 9 --stride 4 --frames 40
$PY harness/run.py ab --base HEAD~5 --head HEAD --features depth

# 3. deterministic replay with a record, then verify later changes against it
$PY harness/run.py replay --features det,pose,seg,depth --frames 40 --start 9 --stride 4 --label base
#    ...edit src/ or td_scripts/...
$PY harness/run.py replay --features det,pose,seg,depth --frames 40 --start 9 --stride 4 --label new \
    --against harness/out/records/base.jsonl            # tolerant (cross-backend safe)
#    add --exact for refactors that must not change a single byte

# 4. ground truth: same frames through onnxruntime-python + a port of the tracker/formatting
$PY harness/run.py replay --features det,pose --vs-reference --start 9 --stride 4 --frames 40

# 5. performance
$PY harness/run.py live --features det,pose --seconds 30 --interval 5          # upload_frames.py at TD's pace
$PY harness/run.py live --mode webcam --features det,pose --seconds 30          # page-driven fake camera
$PY harness/run.py replay --features seg --frames 100 --source-format packed-u8 # zero-copy upload path

# other models / params (any page URL param works)
$PY harness/run.py replay --features det --model det=yolo26n-obb
$PY harness/run.py replay --features det --param Detscoret=0.3 --cpu            # cpu=1 -> onnxruntime wasm
```

Exit status is non-zero on a page error, no results, a TD callback exception, a
script error, or (with `--against`/`--vs-reference`/`ab`/`check --against`) a diff
beyond tolerance, so these slot straight into an agent loop.

## Two ways frames are driven

- **replay (lockstep)** one frame in flight; the next is uploaded only after the
  previous frame's JSON result arrives. Every video frame is processed exactly
  once, in order, so two runs produce the same records regardless of speed. This
  is what comparisons use. The main loop is event-driven, so it runs as fast as
  the browser answers (no 60 fps tick).
- **live** `upload_frames.py` is called every cooked frame, exactly like the
  Execute DAT in TD (busy flag, sync ticks, 60 fps), against a clip that plays
  in real time (`--frames` long, looped). `browser-dropped` counts frames the
  page discarded because it was still busy (it keeps only the latest). Note that
  the page's sync reply clears the `busy` flag in `webserver1_callbacks.py`, so in
  free-running mode the browser drops frames rather than TD pacing itself. This
  is real behaviour of the shipped scripts, not a harness artifact.
  `--mode webcam` skips TD uploads and lets the page read Chrome's fake webcam.

## Reading the output

One line per scenario, then details on failure or in the single-scenario commands:

```
OK   det+pose  load 0.9s | 40 frames @ 20.2/s | rtt p50 34.5 p95 43.3 ms | cb p95 0.33 ms | TD work p95 14.0 ms | browser-dropped 0 | maps -
```

| field | meaning |
| --- | --- |
| `load` | page open -> `{"loaded": true}` (sessions created) |
| `rtt` | `webSocketSendBinary` -> JSON result received. This is the number to optimise: GPU preprocess + inference + decode + tracker + send |
| `cb` | time the real `onWebSocketReceive*` callbacks hold the TD main thread. Seg/depth binary callbacks show here (depth is 1.6 MB/frame) |
| `TD work` | total main-thread time per cooked frame including `upload_frames.py` |
| `segmentation arrives` / `depth arrives` | when each binary map lands relative to the upload; the JSON result is always last |
| DAT `json.loads` | what a downstream consumer pays to parse `predictions` once per changed frame |
| `maps` | commits into `segmentation_data` / `depth_data` Script TOPs |

The page's `warmup` first frames (default 3) are excluded from the stats but kept in the records.

## What `compare` checks

Records are JSONL keyed by video frame (`f00012`), so runs line up by frame.
Per frame: the set of outputs, predictions schema, detection counts, greedy
box/label matching, track ids, max box / score / keypoint / angle error (normalized
units), and for float maps a 64x64 strided sample plus an exact sha1. Segmentation
is judged by class-id agreement and alpha error (the map encodes `classId + 1 + alpha`),
depth by mean/max absolute error in meters. The TD frame counters (`frame`, `seq`)
are ignored since they depend on timing. `--tol-box`, `--tol-score`,
`--tol-seg-alpha`, `--tol-depth-abs` adjust limits; `--exact` demands byte/bit
identity (same backend, same machine). `first divergence` tells you which frame to
look at; `--save-maps DIR` also saves every full map as `.npy`.

Observed here (Apple GPU, Chrome WebGPU): two identical runs are bit-identical;
WebGPU vs onnxruntime-python reference differ by ~1e-7 on boxes/scores/keypoints and
~1e-5 m on depth, track ids identical.

## Failure modes it can emulate

`--max-msgs-per-frame N` (TD ingesting few messages per frame), `--hitch-every S
--hitch-ms MS` (a TD main-thread hitch), `--fps N` (cook rate), `--td-scripts DIR`
(callbacks from an older release against the current web bundle),
`--source-format` (which array layout reaches `upload_frames.py`).

## Files

| file | role |
| --- | --- |
| `run.py` | CLI: `replay`, `live`, `check`, `ab`, `compare` |
| `fake_td.py` | network thread, TD main-thread loop, per-frame metrics, recording |
| `td_stubs.py` | `op`/`absTime`/DAT/TOP/VFS fakes, callback loader |
| `frames.py` | ffmpeg decode + cache, fake `source` TOP |
| `compare.py` | frame-by-frame diff of two recordings |
| `reference.py` | onnxruntime-python golden (YOLO26 det / pose / depth) |

Outputs land in `harness/out/` (gitignored): `records/*.jsonl`, `report-*.json`,
`console-*.log` (browser console), `textport-*.log` (everything the callbacks `print`/`debug`).

## Limits

- Frames are fed as 640x640 letterboxed RGB, which is what `repack.glsl` hands the
  tox; the harness does not run TD's GLSL or CEF. The webcam path in `webcam.js`
  (importExternalTexture GPU path) is exercised by `live --mode webcam`, but its
  inputs are not frame-aligned, so use `replay` for anything you need to diff.
- Chrome's WebGPU here is the real Metal GPU in headless mode; numbers differ from
  CEF inside TD, so use rtt/cb for relative comparisons.
- TD's Web Server DAT internals (ingestion limits, thread hand-off) are modelled
  by the knobs above, not emulated; calibrate them against real TD.
- `ab` against a ref old enough to predate the vertical-packed upload format needs
  `--source-format rgb`.
