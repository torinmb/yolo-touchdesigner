#!/usr/bin/env python3
"""YOLO TouchDesigner test harness: headless Chrome <-> fake TD, no TouchDesigner.

  replay   deterministic: frames are fed to the page one at a time over the same
           binary protocol the tox uses; records every result for comparing
  live     realtime: the clip plays at 30fps (binary like TD, or a fake webcam)
           and you read throughput / latency / TD main-thread cost
  check    smoke suite over every model family (det, pose, seg, depth, obb, v11 ...)
  ab       build a baseline git ref in a worktree, replay both, compare
  compare  compare two recordings frame by frame

Examples:
  harness/.venv/bin/python harness/run.py replay --video YoloDemo.1.mov --features det --frames 30
  harness/.venv/bin/python harness/run.py replay --features seg --against harness/out/records/seg.jsonl
  harness/.venv/bin/python harness/run.py ab --base HEAD~1 --features det,pose,seg --frames 40
  harness/.venv/bin/python harness/run.py check
"""

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

from compare import compare as compareRecordings  # noqa: E402
from fake_td import REPO, FakeTD, summarize  # noqa: E402
from frames import SourceTOP, loadFrames  # noqa: E402

OUT = HARNESS / 'out'
RECORDS = OUT / 'records'
DEFAULT_VIDEOS = ('YoloDemo.1.mov', 'YoloDemo.0.mov', 'YoloDemo.2.mov')

FEATURE_PARAMS = {
	'det': 'Objecttrackingenabled',
	'pose': 'Posetrackingenabled',
	'seg': 'Segmentationenabled',
	'depth': 'Depthenabled',
}
MODEL_PARAMS = {
	'det': 'Objecttrackingmodel',
	'pose': 'Posemodel',
	'seg': 'Segmentationmodel',
	'depth': 'Depthmodel',
}
DEFAULT_MODELS = {'det': 'yolo26n', 'pose': 'yolo26n-pose', 'seg': 'yolo26n-seg', 'depth': 'yolo26n-depth'}

# name -> (features, model overrides, extra url params)
CHECK_SUITE = {
	'det-v26': (['det'], {'det': 'yolo26n'}, {}),
	'pose-v26': (['pose'], {'pose': 'yolo26n-pose'}, {}),
	'det+pose-v26': (['det', 'pose'], {}, {}),
	'seg-v26': (['seg'], {}, {}),
	'depth-v26': (['depth'], {}, {}),
	'obb-v26': (['det'], {'det': 'yolo26n-obb'}, {}),
	'det-v11': (['det'], {'det': 'yolo11n'}, {}),
	'pose-v11': (['pose'], {'pose': 'yolo11n-pose'}, {}),
	'det-v26-wasm': (['det'], {'det': 'yolo26n'}, {'cpu': '1'}),
}

NOISE = ('Failed to load resource', 'DevTools', 'GPU stall', '[vite]', '[W:onnxruntime', 'VerifyEachNodeIsAssignedToAnEp')


# ------------------------------------------------------------------ plumbing

def freePort():
	with socket.socket() as sock:
		sock.bind(('127.0.0.1', 0))
		return sock.getsockname()[1]


def startVite(root):
	port = freePort()
	config = ['--config', 'vite.harness.config.mjs'] if (Path(root) / 'vite.harness.config.mjs').exists() else []
	process = subprocess.Popen(
		[str(Path(root) / 'node_modules' / '.bin' / 'vite'), '--port', str(port), '--strictPort',
			'--host', '127.0.0.1', '--logLevel', 'warn'] + config,
		cwd=root, stdout=subprocess.DEVNULL)
	base = 'http://localhost:{}'.format(port)
	deadline = time.time() + 40
	while time.time() < deadline:
		if process.poll() is not None:
			sys.exit('vite dev server exited (root {})'.format(root))
		try:
			urllib.request.urlopen(base + '/favicon.ico', timeout=1)
			return process, base
		except OSError:
			time.sleep(0.2)
	process.terminate()
	sys.exit('vite dev server did not start')


def buildDist(root, outDir):
	print('vite build -> {} ...'.format(outDir), flush=True)
	began = time.time()
	result = subprocess.run(
		[str(Path(root) / 'node_modules' / '.bin' / 'vite'), 'build', '--logLevel', 'warn',
			'--outDir', str(outDir), '--emptyOutDir']
		+ (['--config', 'vite.harness.config.mjs'] if (Path(root) / 'vite.harness.config.mjs').exists() else []),
		cwd=root)
	if result.returncode != 0:
		sys.exit('vite build failed')
	print('built in {:.1f}s'.format(time.time() - began))


def resolveVideo(name):
	if name:
		return name
	for candidate in DEFAULT_VIDEOS:
		if (REPO / candidate).exists():
			return str(REPO / candidate)
	sys.exit('pass --video <clip> (any format ffmpeg reads; a person in frame gives the models something to find)')


def resolveMjpeg(video):
	"""Chrome's fake webcam only plays MJPEG/Y4M: convert once with ffmpeg, cached."""
	import hashlib
	path = Path(video).expanduser().resolve()
	if path.suffix.lower() in ('.mjpeg', '.mjpg', '.y4m'):
		return path
	stat = path.stat()
	key = hashlib.sha1('{}|{}|{}'.format(path, stat.st_size, stat.st_mtime_ns).encode()).hexdigest()[:10]
	cached = OUT / 'videos' / '{}-{}.mjpeg'.format(path.stem, key)
	if cached.exists():
		return cached
	cached.parent.mkdir(parents=True, exist_ok=True)
	print('converting {} for the fake webcam ...'.format(path.name), flush=True)
	partial = cached.with_suffix('.partial')
	subprocess.run([
		'ffmpeg', '-y', '-loglevel', 'error', '-i', str(path), '-an',
		'-vf', 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuvj420p',
		'-r', '30', '-q:v', '4', '-f', 'mjpeg', str(partial)], check=True)
	partial.rename(cached)
	return cached


class Session:
	"""One Chrome + one Vite dev server per checkout, reused across scenarios."""

	def __init__(self, args, webcamVideo=None):
		self.args = args
		self.webcamVideo = webcamVideo
		self.vite = {}
		self.dist = None
		self.playwright = None
		self.browser = None

	async def __aenter__(self):
		from playwright.async_api import async_playwright
		self.playwright = await async_playwright().start()
		flags = [
			'--use-fake-ui-for-media-stream',
			'--autoplay-policy=no-user-gesture-required',
			'--disable-background-timer-throttling',
			'--disable-renderer-backgrounding',
			'--ignore-gpu-blocklist',
			'--enable-gpu',
			'--enable-unsafe-webgpu',
		]
		if self.webcamVideo:
			flags += ['--use-fake-device-for-media-stream',
				'--use-file-for-fake-video-capture={}'.format(self.webcamVideo)]
		self.browser = await self.playwright.chromium.launch(
			channel='chrome', headless=not self.args.headed, args=flags)
		return self

	async def __aexit__(self, *exc):
		await self.browser.close()
		await self.playwright.stop()
		for process, _ in self.vite.values():
			process.terminate()

	def viteBase(self, root):
		root = str(root)
		if root not in self.vite:
			self.vite[root] = startVite(root)
			print('vite dev server for {} on {}'.format(root, self.vite[root][1]), flush=True)
		return self.vite[root][1]


# ------------------------------------------------------------------ scenario

PROBE = """async () => {
  const out = {webgpu: false};
  try {
    const adapter = navigator.gpu && await navigator.gpu.requestAdapter();
    if (adapter) {
      const info = adapter.info || (adapter.requestAdapterInfo && await adapter.requestAdapterInfo()) || {};
      out.webgpu = true;
      out.adapter = [info.vendor, info.architecture, info.description].filter(Boolean).join(' / ');
    }
  } catch (e) { out.error = String(e); }
  return out;
}"""

HEAP = "() => performance.memory ? performance.memory.usedJSHeapSize / 1048576 : 0"


def scenarioUrl(base, scenario, port, mode):
	params = {'wsPort': port, 'binary': '1' if mode == 'binary' else '0'}
	for feature, param in FEATURE_PARAMS.items():
		params[param] = '1' if feature in scenario['features'] else '0'
	for feature in scenario['features']:
		params[MODEL_PARAMS[feature]] = scenario['models'].get(feature, DEFAULT_MODELS[feature])
	params.update(scenario['params'])
	return '{}/?{}'.format(base, urllib.parse.urlencode(params))


def makeScenario(name, features, models=None, params=None):
	return {'name': name, 'features': list(features), 'models': dict(models or {}), 'params': dict(params or {})}


def scenarioFromArgs(args):
	features = [f for f in args.features.split(',') if f]
	for feature in features:
		if feature not in FEATURE_PARAMS:
			sys.exit('unknown feature {} (choose from {})'.format(feature, ', '.join(FEATURE_PARAMS)))
	models = dict(m.split('=', 1) for m in args.model)
	params = dict(p.split('=', 1) for p in args.param)
	if args.cpu:
		params['cpu'] = '1'
	return makeScenario(args.label or '+'.join(features), features, models, params)


async def runScenario(session, scenario, args, mode='binary', lockstep=True, root=REPO, tdScripts=None,
		recordPath=None, frames=None, seconds=None, source='dev', interval=0):
	"""Run one scenario against a fresh fake TD; returns the report dict."""
	root = Path(root).resolve()
	page = None
	consoleLines = []
	OUT.mkdir(exist_ok=True)
	sourceTop = None
	if mode == 'binary':
		count = frames or args.frames
		clip = loadFrames(args.video, count, start=args.start, stride=args.stride,
			fit=args.fit)
		sourceTop = SourceTOP(clip, None, fmt=args.source_format,
			pace='asap' if lockstep else 'realtime')
	distDir = session.dist if source == 'dist' else None
	td = FakeTD(
		fps=args.fps, lockstep=lockstep, source=sourceTop,
		scriptsDir=tdScripts or Path(root) / 'td_scripts', distDir=distDir,
		recordPath=recordPath, logPath=OUT / 'textport-{}.log'.format(scenario['name']),
		verbose=args.verbose, mapStride=args.map_stride,
		saveMapsDir=(Path(args.save_maps) / scenario['name']) if args.save_maps else None,
		maxMsgsPerFrame=args.max_msgs_per_frame, hitchEverySec=args.hitch_every, hitchMs=args.hitch_ms,
		frameTimeoutSec=args.frame_timeout,
	).start()
	if sourceTop is not None:
		sourceTop.absTime = td.env.absTime
	base = 'http://localhost:{}'.format(td.port) if source == 'dist' else session.viteBase(root)

	context = await session.browser.new_context(
		viewport={'width': 1280, 'height': 720}, permissions=['camera'])
	page = await context.new_page()
	await page.add_init_script(path=str(HARNESS / 'probe.js'))
	page.on('console', lambda msg: consoleLines.append('[{}] {}'.format(msg.type, msg.text)))
	page.on('pageerror', lambda err: consoleLines.append('[pageerror] {}'.format(err)))
	page.on('crash', lambda: consoleLines.append('[pageerror] page crashed'))
	page.on('response', lambda r: r.status >= 400 and consoleLines.append('[http {}] {}'.format(r.status, r.url)))

	url = scenarioUrl(base, scenario, td.port, mode)
	openedAt = time.time()
	await page.goto(url)
	probe = await page.evaluate(PROBE)

	failure = None
	deadline = time.time() + args.load_timeout
	while td.loadedAt is None:
		if time.time() > deadline:
			failure = 'page never reported {{"loaded": true}} within {:.0f}s (models failed to load?)'.format(
				args.load_timeout)
			break
		if any(line.startswith('[pageerror]') for line in consoleLines):
			failure = 'page error while loading'
			break
		await asyncio.sleep(0.1)
	loadSec = (td.loadedAt / 1000.0 - openedAt) if td.loadedAt else None

	timeline = []
	if not failure:
		if mode == 'binary' and lockstep:
			limit = time.time() + args.run_timeout
			while not td.done and time.time() < limit:
				await asyncio.sleep(0.05)
			if not td.done:
				failure = 'run did not finish within {:.0f}s'.format(args.run_timeout)
		else:
			timeline = await sampleTimeline(page, td, seconds or args.seconds, interval)

	snapshot = td.snapshot(warmup=args.warmup)
	heap = await page.evaluate(HEAP)
	probeData = await page.evaluate('window.__probe') if mode == 'binary' else None
	td.stop()
	await context.close()

	errors = [
		line for line in consoleLines
		if line.startswith(('[error]', '[pageerror]', '[http'))
		and not any(noise in line for noise in NOISE)
	]
	if snapshot['aborted']:
		failure = failure or snapshot['aborted']
	if not failure and mode == 'binary' and snapshot['sent'] == 0:
		failure = 'upload_frames.py never uploaded a frame (client/busy/source problem; see textport log)'
	elif not failure and snapshot['answered'] == 0:
		failure = 'frames were uploaded but the page produced no results (protocol mismatch?)'
	(OUT / 'console-{}.log'.format(scenario['name'])).write_text('\n'.join(consoleLines))
	report = {
		'scenario': scenario, 'mode': mode, 'lockstep': lockstep, 'url': url,
		'gpu': probe, 'loadSec': loadSec, 'heapMB': heap, 'td': snapshot,
		'pageProbe': analyzeProbe(probeData, args.warmup), 'timeline': timeline, 'consoleErrors': errors, 'failure': failure,
		'record': str(recordPath) if recordPath else None,
		'ok': not (failure or errors or snapshot['scriptErrors'] or snapshot['exceptions']),
	}
	return report


def analyzeProbe(probe, warmup):
	"""Per-frame page-side stages: frame arrives -> every send of that frame's outputs.
	Frame boundaries are the big binary frame receives (>= 1 MB)."""
	if not probe:
		return None
	frames = [r for r in probe['recv'] if r[1] >= 500000]
	if len(frames) <= warmup + 2:
		return None
	rows = []
	for index, (t0, size) in enumerate(frames[:-1]):
		t1 = frames[index + 1][0]
		sends = [x for x in probe['send'] if t0 <= x[0] < t1 and x[2] != 'keepalive']
		maps = [m for m in probe['map'] if t0 <= m[0] < t1]
		submits = [t for t in probe['submits'] if t0 <= t < t1]
		writes = [w for w in probe['writes'] if t0 <= w[0] < t1]
		if not sends:
			continue
		lastSubmit = max(submits) if submits else t0
		lastMapEnd = max((m[0] + m[1] for m in maps), default=lastSubmit)
		rows.append({
			'recvToLastSubmitMs': lastSubmit - t0,
			'lastSubmitToMapEndMs': max(0.0, lastMapEnd - lastSubmit),
			'mapEndToSendMs': sends[-1][0] - lastMapEnd,
			'pageMs': sends[-1][0] - t0,
			'firstSendMs': sends[0][0] - t0,
			'sends': len(sends),
			'sendKB': sum(x[1] for x in sends) / 1024.0,
			'mapCount': len(maps),
			'mapMs': sum(m[1] for m in maps),
			'mapMB': sum(m[2] for m in maps) / 1048576.0,
			'mapSizes': sorted((m[2] for m in maps), reverse=True)[:6],
			'submits': len(submits),
			'writeMB': sum(w[1] for w in writes) / 1048576.0,
			'writes': len(writes),
		})
	rows = rows[warmup:]
	if not rows:
		return None
	out = {key: summarize([r[key] for r in rows]) for key in rows[0] if key != 'mapSizes'}
	out['mapSizes'] = rows[len(rows) // 2]['mapSizes']
	return out


async def sampleTimeline(page, td, seconds, interval):
	"""live mode: wait `seconds`, optionally printing a row every `interval`."""
	timeline = []
	started = time.time()
	last = {'answered': 0, 'at': started}
	while time.time() - started < seconds:
		await asyncio.sleep(min(interval or seconds, max(0.01, seconds - (time.time() - started))))
		if not interval:
			continue
		now = time.time()
		answered = td.answered
		frames = [f for f in td.frameLog.values() if f.get('doneAt', 0) / 1000.0 >= last['at']]
		row = {
			't': round(now - started, 1),
			'resultsPerSec': (answered - last['answered']) / max(1e-6, now - last['at']),
			'rttMs': summarize([f['rtt'] for f in frames if 'rtt' in f]),
			'tdWorkMsP95': summarize(td.frameWorkMs[-int(td.fps * interval):]).get('p95', 0),
			'heapMB': await page.evaluate(HEAP),
		}
		timeline.append(row)
		print('t={:>5.0f}s  {:5.1f} results/s | rtt p50 {:.1f} p95 {:.1f} ms | TD work p95 {:.2f} ms | heap {:.0f} MB'.format(
			row['t'], row['resultsPerSec'], row['rttMs'].get('p50', 0), row['rttMs'].get('p95', 0),
			row['tdWorkMsP95'], row['heapMB']), flush=True)
		last = {'answered': answered, 'at': now}
	return timeline


# ------------------------------------------------------------------- output

def f(stat, key='avg', digits=1):
	return '{:.{}f}'.format(stat[key], digits) if stat.get('n') else '-'


def summaryLine(report):
	td = report['td']
	s = report['scenario']
	status = 'OK  ' if report['ok'] else 'FAIL'
	maps = ' '.join('{}={}'.format(k.replace('_data', ''), v['commits']) for k, v in td['maps'].items())
	return ('{} {:<20} load {:>5}s | {} frames @ {:5.1f}/s | rtt p50 {} p95 {} ms | '
		'cb p95 {} ms | TD work p95 {} ms | browser-dropped {} | maps {}').format(
		status, s['name'][:20],
		'{:.1f}'.format(report['loadSec']) if report['loadSec'] else '-',
		td['answered'], td['throughputFps'], f(td['rttMs'], 'p50'), f(td['rttMs'], 'p95'),
		f(td['kinds'].get('yolo', td['kinds'].get('yolo_combined', td['kinds'].get('yolo_pose', {})))
			.get('callbackMs', {}), 'p95', 2),
		f(td['tdFrames']['workMs'], 'p95', 2), td['droppedByBrowser'], maps or '-')


def printReport(report, verbose=True):
	td = report['td']
	print(summaryLine(report))
	if not verbose:
		if not report['ok']:
			printProblems(report)
		return
	gpu = report['gpu']
	print('  webgpu: {}'.format(gpu.get('adapter') if gpu.get('webgpu') else 'UNAVAILABLE ' + str(gpu.get('error', ''))))
	print('  stages (steady state, ms):  rtt avg {} p95 {} max {}'.format(
		f(td['rttMs']), f(td['rttMs'], 'p95'), f(td['rttMs'], 'max')), end='')
	print('  result interval ms: avg {} p50 {} p95 {} max {}'.format(
		f(td['resultIntervalMs']), f(td['resultIntervalMs'], 'p50'), f(td['resultIntervalMs'], 'p95'),
		f(td['resultIntervalMs'], 'max')), end='')
	for name, stat in td['stageMs'].items():
		if stat.get('n'):
			print('  | {} arrives avg {} p95 {}'.format(name.replace('Ms', ''), f(stat), f(stat, 'p95')), end='')
	print()
	probe = report.get('pageProbe')
	if probe:
		print('  page: recv->last send avg {} ms | GPU->CPU readbacks {} / {} ms / {} MB per frame | '
			'queue.submit {} | writeBuffer {} / {} MB per frame'.format(
			f(probe['pageMs']), f(probe['mapCount'], 'avg', 1), f(probe['mapMs']), f(probe['mapMB'], 'avg', 2),
			f(probe['submits'], 'avg', 1), f(probe['writes'], 'avg', 1), f(probe['writeMB'], 'avg', 2)))
		print('  page timeline: frame recv -> last queue.submit {} ms | -> last readback done {} ms | -> last send {} ms'.format(
			f(probe['recvToLastSubmitMs']), f(probe['lastSubmitToMapEndMs']), f(probe['mapEndToSendMs'])))
		print('  page readback sizes (bytes, one frame): {}'.format(probe['mapSizes']))
	for label, kind in td['kinds'].items():
		if label in ('keepalive',):
			continue
		print('  msg {:<14} n={:<5} avg {:7.1f} KB  callback avg {} p95 {} ms  net->callback p95 {} ms'.format(
			label, kind['count'], kind['avgKB'], f(kind['callbackMs'], 'avg', 2),
			f(kind['callbackMs'], 'p95', 2), f(kind['queueMs'], 'p95')))
	for name, dat in td['dats'].items():
		print('  DAT {:<18} writes {:<5} downstream json.loads avg {} ms'.format(
			name, dat['writes'], f(dat['downstreamParseMs'], 'avg', 3)))
	for name, info in td['maps'].items():
		print('  TOP {:<18} {} commits shape {} {}'.format(name, info['commits'], info['shape'], info['dtype']))
	print('  upload_frames.py: avg {} p95 {} ms per call | page heap {:.0f} MB'.format(
		f(td['tdFrames']['uploadMs'], 'avg', 2), f(td['tdFrames']['uploadMs'], 'p95', 2), report['heapMB']))
	printProblems(report)
	if report['record']:
		print('  record: {}'.format(report['record']))


def printProblems(report):
	td = report['td']
	if report['failure']:
		print('  FAILURE: {}'.format(report['failure']))
	for line in report['consoleErrors'][:8]:
		print('  console: {}'.format(line[:200]))
	for line in td['scriptErrors'][:5] + td['exceptions'][:5]:
		print('  TD error: {}'.format(line[:200]))


def saveReport(report, name):
	OUT.mkdir(exist_ok=True)
	path = OUT / 'report-{}.json'.format(name)
	path.write_text(json.dumps(report, indent=2, default=str))
	return path


# ------------------------------------------------------------------ commands

def commonTD(p):
	p.add_argument('--video', help='clip to use (any format; default: a YoloDemo.*.mov in the repo root)')
	p.add_argument('--features', default='det', help='comma list of: det,pose,seg,depth')
	p.add_argument('--model', action='append', default=[], help='feature=name, e.g. det=yolo26n-obb')
	p.add_argument('--param', action='append', default=[], help='extra page URL param K=V, e.g. Detscoret=0.3')
	p.add_argument('--cpu', action='store_true', help='cpu=1: onnxruntime wasm instead of WebGPU')
	p.add_argument('--label', help='name for the report/record files (default: the features)')
	p.add_argument('--source', choices=('dev', 'dist'), default='dev',
		help='dev: Vite dev server on src/ (no build); dist: vite build, served by the real onHTTPRequest')
	p.add_argument('--root', default=str(REPO), help='checkout to serve (src/, public/, td_scripts/)')
	p.add_argument('--td-scripts', help='td_scripts directory to exec (e.g. an older release)')
	p.add_argument('--build', action='store_true', help='(re)build before --source dist')
	p.add_argument('--fps', type=int, default=60, help='TD cook rate for non-lockstep runs')
	p.add_argument('--start', type=float, default=0.0, help='skip this many seconds of the clip')
	p.add_argument('--stride', type=int, default=1, help='use every Nth video frame')
	p.add_argument('--fit', choices=('letterbox', 'stretch'), default='letterbox')
	p.add_argument('--source-format', choices=('packed', 'packed-u8', 'rgb'), default='packed',
		help='array layout the fake source TOP hands upload_frames.py')
	p.add_argument('--warmup', type=int, default=3, help='frames excluded from perf stats')
	p.add_argument('--map-stride', type=int, default=10, help='pixel stride of the map samples kept in records')
	p.add_argument('--save-maps', help='also np.save every full seg/depth map under this directory')
	p.add_argument('--max-msgs-per-frame', type=int, default=0, help='emulate TD ingesting N messages per frame')
	p.add_argument('--hitch-every', type=float, default=0)
	p.add_argument('--hitch-ms', type=float, default=0)
	p.add_argument('--load-timeout', type=float, default=120)
	p.add_argument('--run-timeout', type=float, default=300)
	p.add_argument('--frame-timeout', type=float, default=20)
	p.add_argument('--headed', action='store_true')
	p.add_argument('--verbose', action='store_true', help='echo the TD textport')
	p.add_argument('--json', action='store_true', help='print the full report as JSON')


def prepareDist(session, args, root):
	if args.source != 'dist':
		return
	session.dist = OUT / 'dist-{}'.format(Path(root).name)
	if args.build or not session.dist.exists():
		buildDist(root, session.dist)


def tolerances(args):
	return {key: getattr(args, 'tol_' + key) for key in ('box', 'score', 'seg_alpha', 'depth_abs')
		if getattr(args, 'tol_' + key, None) is not None}


async def replayCommand(args):
	args.video = resolveVideo(args.video)
	scenario = scenarioFromArgs(args)
	RECORDS.mkdir(parents=True, exist_ok=True)
	recordPath = Path(args.record) if args.record else RECORDS / '{}.jsonl'.format(scenario['name'])
	async with Session(args) as session:
		prepareDist(session, args, args.root)
		report = await runScenario(session, scenario, args, mode='binary', lockstep=True, root=args.root,
			tdScripts=args.td_scripts, recordPath=recordPath, source=args.source)
	saveReport(report, scenario['name'])
	printReport(report)
	if args.json:
		print(json.dumps(report, indent=2, default=str))
	code = 0 if report['ok'] else 1
	if args.vs_reference:
		print()
		refPath = RECORDS / '{}-reference.jsonl'.format(scenario['name'])
		buildReference(args, scenario, refPath)
		result = compareRecordings(refPath, recordPath, tol=tolerances(args), exact=False)
		code = code or (0 if result['ok'] else 1)
	if args.against:
		print()
		result = compareRecordings(args.against, recordPath, tol=tolerances(args), exact=args.exact)
		code = code or (0 if result['ok'] else 1)
	return code


def buildReference(args, scenario, path):
	from reference import writeReference
	scoreT = {}
	for feature, param in (('det', 'Detscoret'), ('pose', 'Posescoret')):
		if param in scenario['params']:
			scoreT[feature] = float(scenario['params'][param])
	began = time.time()
	count = writeReference(args.video, args.frames, scenario['features'], scenario['models'], path,
		start=args.start, stride=args.stride, fit=args.fit, scoreT=scoreT, mapStride=args.map_stride)
	print('reference: {} frames through onnxruntime CPU in {:.1f}s -> {}'.format(count, time.time() - began, path))


async def liveCommand(args):
	args.video = resolveVideo(args.video)
	scenario = scenarioFromArgs(args)
	webcamVideo = resolveMjpeg(args.video) if args.mode == 'webcam' else None
	RECORDS.mkdir(parents=True, exist_ok=True)
	recordPath = Path(args.record) if args.record else RECORDS / '{}-live.jsonl'.format(scenario['name'])
	async with Session(args, webcamVideo) as session:
		prepareDist(session, args, args.root)
		report = await runScenario(session, scenario, args, mode=args.mode, lockstep=False, root=args.root,
			tdScripts=args.td_scripts, recordPath=recordPath, frames=args.frames, seconds=args.seconds,
			source=args.source, interval=args.interval)
	saveReport(report, scenario['name'] + '-live')
	printReport(report)
	if args.json:
		print(json.dumps(report, indent=2, default=str))
	return 0 if report['ok'] else 1


async def checkCommand(args):
	args.video = resolveVideo(args.video)
	only = set(args.only.split(',')) if args.only else set(CHECK_SUITE)
	outDir = Path(args.out) if args.out else OUT / 'check'
	outDir.mkdir(parents=True, exist_ok=True)
	code = 0
	results = []
	async with Session(args) as session:
		prepareDist(session, args, args.root)
		for name, (features, models, params) in CHECK_SUITE.items():
			if name not in only:
				continue
			scenario = makeScenario(name, features, models, params)
			recordPath = outDir / '{}.jsonl'.format(name)
			report = await runScenario(session, scenario, args, mode='binary', lockstep=True, root=args.root,
				tdScripts=args.td_scripts, recordPath=recordPath, source=args.source)
			printReport(report, verbose=not report['ok'])
			ok = report['ok']
			if args.against:
				baseline = Path(args.against) / '{}.jsonl'.format(name)
				if baseline.exists():
					result = compareRecordings(baseline, recordPath, tol=tolerances(args), exact=args.exact, quiet=ok)
					ok = ok and result['ok']
				else:
					print('  (no baseline for {})'.format(name))
			results.append((name, ok))
			code = code or (0 if ok else 1)
	print('\n{}: {}'.format('CHECK PASSED' if code == 0 else 'CHECK FAILED',
		' '.join('{}={}'.format(n, 'ok' if ok else 'FAIL') for n, ok in results)))
	if args.against is None:
		print('baseline recorded in {} (use --against {} next time)'.format(outDir, outDir))
	return code


def makeWorktree(ref):
	"""Check `ref` out into harness/out/worktrees/<ref> with the shared node_modules."""
	safe = ref.replace('/', '_').replace('~', '_').replace('^', '_')
	path = OUT / 'worktrees' / safe
	if path.exists():
		subprocess.run(['git', 'worktree', 'remove', '--force', str(path)], cwd=REPO, check=False,
			stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		shutil.rmtree(path, ignore_errors=True)
	path.parent.mkdir(parents=True, exist_ok=True)
	subprocess.run(['git', 'worktree', 'add', '--detach', str(path), ref], cwd=REPO, check=True,
		stdout=subprocess.DEVNULL)
	os.symlink(REPO / 'node_modules', path / 'node_modules')
	# node_modules is a symlink to the main checkout: let Vite serve through it
	(path / 'vite.harness.config.mjs').write_text(
		"import {{ mergeConfig }} from 'vite'\n"
		"import base from './vite.config.js'\n"
		"export default mergeConfig(base, {{ server: {{ fs: {{ allow: ['{}', '{}'] }} }} }})\n".format(path, REPO))
	# model files are large and may be untracked/LFS: share the working copy's
	models = path / 'public' / 'models'
	if not models.exists() or not any(models.glob('*.onnx')):
		shutil.rmtree(models, ignore_errors=True)
		os.symlink(REPO / 'public' / 'models', models)
	return path


async def abCommand(args):
	args.video = resolveVideo(args.video)
	scenario = scenarioFromArgs(args)
	base = makeWorktree(args.base)
	head = Path(args.root)
	if args.head:
		head = makeWorktree(args.head)
	print('A (base) {} -> {}\nB (head) {}'.format(args.base, base, args.head or 'working tree'), flush=True)
	RECORDS.mkdir(parents=True, exist_ok=True)
	reports = {}
	async with Session(args) as session:
		for label, root in (('base', base), ('head', head)):
			recordPath = RECORDS / '{}-{}.jsonl'.format(scenario['name'], label)
			prepareDist(session, args, root)
			reports[label] = await runScenario(
				session, {**scenario, 'name': '{}-{}'.format(scenario['name'], label)}, args, mode='binary',
				lockstep=True, root=root, recordPath=recordPath, source=args.source)
			printReport(reports[label], verbose=False)
	print()
	result = compareRecordings(RECORDS / '{}-base.jsonl'.format(scenario['name']),
		RECORDS / '{}-head.jsonl'.format(scenario['name']), tol=tolerances(args), exact=args.exact)
	a, b = reports['base']['td'], reports['head']['td']
	print('  speed: base {:.1f} fps rtt p50 {} ms  ->  head {:.1f} fps rtt p50 {} ms'.format(
		a['throughputFps'], f(a['rttMs'], 'p50'), b['throughputFps'], f(b['rttMs'], 'p50')))
	return 0 if result['ok'] and all(r['ok'] for r in reports.values()) else 1


def compareCommand(args):
	result = compareRecordings(args.a, args.b, tol=tolerances(args), exact=args.exact)
	return 0 if result['ok'] else 1


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = parser.add_subparsers(dest='command', required=True)

	def tols(p):
		p.add_argument('--exact', action='store_true', help='require byte/bit-identical output (pure refactors)')
		p.add_argument('--tol-box', type=float, help='max normalized box diff (default 1e-3)')
		p.add_argument('--tol-score', type=float, help='max score diff (default 1e-3)')
		p.add_argument('--tol-seg-alpha', type=float, help='max mean seg alpha diff (default 0.02)')
		p.add_argument('--tol-depth-abs', type=float, help='max mean depth diff in meters (default 0.02)')

	replay = sub.add_parser('replay', help='deterministic frame-by-frame run (use this to verify changes)')
	commonTD(replay)
	tols(replay)
	replay.add_argument('--frames', type=int, default=30)
	replay.add_argument('--record', help='record file (default harness/out/records/<label>.jsonl)')
	replay.add_argument('--against', help='baseline recording to compare with afterwards')
	replay.add_argument('--vs-reference', action='store_true',
		help='also run the same frames through onnxruntime-python (det/pose/depth of YOLO26) and compare')

	live = sub.add_parser('live', help='realtime run for throughput/latency/soak')
	commonTD(live)
	live.add_argument('--mode', choices=('binary', 'webcam'), default='binary',
		help='binary: TD uploads frames (upload_frames.py, unlocked); webcam: the page reads a fake camera')
	live.add_argument('--seconds', type=float, default=15)
	live.add_argument('--frames', type=int, default=150, help='clip length to loop in binary mode')
	live.add_argument('--interval', type=float, default=0, help='print a timeline row every N seconds')
	live.add_argument('--record')

	check = sub.add_parser('check', help='smoke every model family; with --against, also compare to a baseline')
	commonTD(check)
	tols(check)
	check.add_argument('--frames', type=int, default=15)
	check.add_argument('--only', help='comma list of: ' + ','.join(CHECK_SUITE))
	check.add_argument('--out', help='where to write records (default harness/out/check)')
	check.add_argument('--against', help='directory of baseline records from an earlier check --out')

	ab = sub.add_parser('ab', help='replay a baseline git ref and the working tree, then compare')
	commonTD(ab)
	tols(ab)
	ab.add_argument('--base', required=True, help='git ref for the baseline, e.g. HEAD~1 or master')
	ab.add_argument('--head', help='git ref for the candidate (default: the working tree)')
	ab.add_argument('--frames', type=int, default=30)

	cmp_ = sub.add_parser('compare', help='compare two recordings')
	cmp_.add_argument('a')
	cmp_.add_argument('b')
	tols(cmp_)

	args = parser.parse_args()
	if args.command == 'compare':
		sys.exit(compareCommand(args))
	runner = {'replay': replayCommand, 'live': liveCommand, 'check': checkCommand, 'ab': abCommand}[args.command]
	if hasattr(args, 'features') and args.command == 'check':
		args.label = None
	sys.exit(asyncio.run(runner(args)))


if __name__ == '__main__':
	main()
