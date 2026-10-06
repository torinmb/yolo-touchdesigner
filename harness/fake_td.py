"""A headless stand-in for the YOLO TouchDesigner component.

Network I/O runs on an asyncio thread (like TD's Web Server DAT network thread).
Every callback runs on one "TD main thread" that cooks at a fixed rate and
drains the queued events each frame. td_scripts/webserver1_callbacks.py and
td_scripts/upload_frames.py are loaded as-is.

Two ways of driving frames into the browser (as the tox does in binary mode):

  lockstep   one frame in flight: the next frame is uploaded only after the
             previous frame's JSON result arrives. Every video frame is
             processed exactly once, in order -> reproducible, comparable
             output. The main loop wakes on each message instead of a 60fps tick.
  free       upload_frames.py is called every cooked frame, exactly like the
             Execute DAT in TD. Its busy flag, sync ticks and the browser's
             drop-old-frames behaviour decide what gets processed.
"""

import asyncio
import base64
import collections
import json
import re
import struct
import threading
import time
from pathlib import Path

import numpy as np
from aiohttp import web

from td_stubs import TDEnvironment, arrayDigest, guessType

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / 'td_scripts'
RESULT_KINDS = ('yolo', 'yolo_pose', 'yolo_combined')
KIND = re.compile(r'"type"\s*:\s*"([A-Za-z_]+)"')
FRAME = re.compile(r'"frame"\s*:\s*(\d+)')
VIDEO_FRAME = re.compile(r'"videoFrame"\s*:\s*(-?[\d.]+)')


def pct(values, p):
	if not values:
		return 0.0
	values = sorted(values)
	return values[min(len(values) - 1, max(0, int(round(p / 100.0 * (len(values) - 1)))))]


def summarize(values):
	if not len(values):
		return {'n': 0}
	return {
		'n': len(values),
		'avg': sum(values) / len(values),
		'p50': pct(values, 50),
		'p95': pct(values, 95),
		'max': max(values),
	}


def classifyText(data):
	match = KIND.search(data)
	if match:
		return match.group(1)
	if '"keepalive"' in data[:60]:
		return 'keepalive'
	for key in ('tick', 'loaded', 'webcamDevices', 'lastFrameTime'):
		if '"{}"'.format(key) in data[:60]:
			return key
	return 'text'


def classifyBinary(data):
	if len(data) >= 16:
		kind = data[0]
		return {11: 'segmentation', 12: 'depth'}.get(kind, 'binary')
	return 'binary'


def mapSummary(array, stride):
	"""Compact record of a float map: exact digest + stats + a strided sample for diffing."""
	flat = array.reshape(array.shape[0], array.shape[1])
	sample = np.ascontiguousarray(flat[::stride, ::stride], dtype='<f4')
	return {
		'shape': list(array.shape),
		'dtype': str(array.dtype),
		'sha1': arrayDigest(array),
		'min': float(flat.min()),
		'max': float(flat.max()),
		'mean': float(flat.mean()),
		'nonzero': float(np.count_nonzero(flat)) / flat.size,
		'stride': stride,
		'subshape': list(sample.shape),
		'sub': base64.b64encode(sample.tobytes()).decode('ascii'),
	}


class WebServerDATStub:
	"""The `webServerDAT` argument handed to the callbacks (also op('webserver1'))."""

	def __init__(self, td):
		self._td = td
		self._store = {}

	def store(self, key, value):
		self._store[key] = value

	def fetch(self, key, default=None):
		return self._store.get(key, default)

	def addWarning(self, message):
		self._td.log('[warning] ' + str(message))

	def webSocketSendText(self, client, data):
		self._td.sendToClient(client, str(data), text=True)

	def webSocketSendBinary(self, client, data):
		self._td.onFrameUploaded(data)
		self._td.sendToClient(client, bytes(data), text=False)

	def webSocketSendPing(self, client):
		return

	def webSocketSendPong(self, client, data=None):
		return


class FakeTD:
	def __init__(
		self,
		port=0,
		fps=60,
		lockstep=True,
		source=None,
		scriptsDir=SCRIPTS,
		distDir=None,
		recordPath=None,
		logPath=None,
		verbose=False,
		mapStride=10,
		saveMapsDir=None,
		maxMsgsPerFrame=0,
		hitchEverySec=0,
		hitchMs=0,
		frameTimeoutSec=20.0,
	):
		self.port = port
		self.fps = fps
		self.lockstep = lockstep
		self.source = source
		self.verbose = verbose
		self.mapStride = mapStride
		self.maxMsgsPerFrame = maxMsgsPerFrame
		self.hitchEverySec = hitchEverySec
		self.hitchMs = hitchMs
		self._nextHitch = time.perf_counter() + hitchEverySec if hitchEverySec else None
		self.frameTimeoutSec = frameTimeoutSec
		self._logFile = open(logPath, 'w') if logPath else None
		self._recordFile = open(recordPath, 'w') if recordPath else None
		self.saveMapsDir = Path(saveMapsDir) if saveMapsDir else None
		if self.saveMapsDir:
			self.saveMapsDir.mkdir(parents=True, exist_ok=True)

		self._lock = threading.Lock()
		self._events = collections.deque()
		self._wake = threading.Event()
		self._running = False
		self.loop = None
		self.connections = {}
		self.clientsSeen = 0
		self.loadedAt = None  # wall-clock ms the page said {"loaded": true}

		# Per-frame bookkeeping, keyed by the TD frame number in the packet header.
		self.frameLog = {}
		self.inflight = None
		self.inflightSince = 0.0
		self.answered = 0
		self.resultTimes = []
		self.aborted = None
		self.pendingMaps = []
		self._ctx = {'td': None, 'key': None}
		self._recordSec = 0.0
		self.kinds = collections.defaultdict(lambda: {'count': 0, 'bytes': 0, 'cb': [], 'q': []})
		self.frameWorkMs = []
		self.uploadMs = []
		self.cooked = 0
		self.msgsPerFrame = []
		self.droppedFrames = 0
		self.datWrites = collections.Counter()
		self.mapCommits = collections.Counter()
		self.mapShapes = {}
		self.downstreamParseMs = collections.defaultdict(list)
		self._dirtyDats = set()
		self.exceptions = []
		self.startedAt = time.time()

		self.env = TDEnvironment(
			log=self.log, onDatWrite=self._onDatWrite, onTopCopy=self._onTopCopy, distDir=distDir)
		self.webServerDAT = WebServerDATStub(self)
		self.env.ops['webserver1'] = self.webServerDAT
		scriptsDir = Path(scriptsDir)
		self.server = self.env.loadCallbacks(scriptsDir / 'webserver1_callbacks.py')
		self.uploader = None
		if source is not None:
			self.env.ops['source'] = source
			self.uploader = self.env.loadCallbacks(scriptsDir / 'upload_frames.py')

	# ------------------------------------------------------------------ logging

	def log(self, message):
		if self._logFile:
			self._logFile.write(message + '\n')
			self._logFile.flush()
		if self.verbose:
			print('[td] ' + message, flush=True)

	def _record(self, entry):
		if self._recordFile:
			entry['td'] = self._ctx['td']
			entry['key'] = self._ctx['key']
			self._recordFile.write(json.dumps(entry) + '\n')

	def _onDatWrite(self, name, text):
		self.datWrites[name] += 1
		self._dirtyDats.add(name)
		if name == 'predictions':
			self._record({'op': name, 'text': text})

	def _onTopCopy(self, name, array):
		began = time.perf_counter()
		try:
			self._onTopCopyInner(name, array)
		finally:
			self._recordSec += time.perf_counter() - began  # harness bookkeeping, not TD cost

	def _onTopCopyInner(self, name, array):
		self.mapCommits[name] += 1
		self.mapShapes[name] = (list(array.shape), str(array.dtype))
		summary = mapSummary(array, self.mapStride)
		if self.saveMapsDir:
			np.save(self.saveMapsDir / '{}_{:05d}.npy'.format(name, self.mapCommits[name]), array)
		if self._ctx['key'] is None and self.source is None:
			self.pendingMaps.append((name, summary))
			return
		self._record({'op': name, **summary})

	# --------------------------------------------------------------- lifecycle

	def start(self):
		ready = threading.Event()
		self._running = True
		threading.Thread(target=self._runNetwork, args=(ready,), daemon=True, name='td-network').start()
		ready.wait()
		threading.Thread(target=self._runFrames, daemon=True, name='td-main').start()
		self._enqueue(('serverStart',))
		return self

	def stop(self):
		self._running = False
		self._wake.set()
		if self._recordFile:
			self._recordFile.close()
		if self.loop and self.loop.is_running():
			future = asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop)
			try:
				future.result(timeout=3)
			except Exception:
				pass
			self.loop.call_soon_threadsafe(self.loop.stop)

	async def _shutdown(self):
		for ws in list(self.connections.values()):
			await ws.close()
		await self._runner.cleanup()

	@property
	def done(self):
		if self.aborted:
			return True
		return (
			self.source is not None
			and self.source.pace == 'asap'
			and self.source.exhausted
			and self.inflight is None
		)

	# ---------------------------------------------------------------- network

	def _runNetwork(self, ready):
		self.loop = asyncio.new_event_loop()
		asyncio.set_event_loop(self.loop)
		app = web.Application(client_max_size=1024 ** 3)
		app.router.add_route('*', '/{tail:.*}', self._handle)
		runner = self._runner = web.AppRunner(app, access_log=None)
		self.loop.run_until_complete(runner.setup())
		site = web.TCPSite(runner, '127.0.0.1', self.port)
		self.loop.run_until_complete(site.start())
		self.port = site._server.sockets[0].getsockname()[1]
		ready.set()
		self.loop.run_forever()

	async def _handle(self, request):
		if request.headers.get('Upgrade', '').lower() == 'websocket':
			return await self._handleWebSocket(request)
		future = self.loop.create_future()
		self._enqueue(('http', request.method, request.path, dict(request.query), future))
		status, contentType, body = await future
		return web.Response(
			status=status, body=body,
			headers={'Content-Type': contentType or 'application/octet-stream'})

	async def _handleWebSocket(self, request):
		ws = web.WebSocketResponse(max_msg_size=0, compress=False)
		await ws.prepare(request)
		peer = request.transport.get_extra_info('peername') or ('127.0.0.1', 0)
		clientId = '{}:{}'.format(peer[0], peer[1])
		with self._lock:
			self.connections[clientId] = ws
			self.clientsSeen += 1
		self._enqueue(('open', clientId, request.path))
		try:
			async for message in ws:
				recvMs = time.time() * 1000.0
				if message.type == web.WSMsgType.TEXT:
					self._enqueue(('text', clientId, message.data, recvMs))
				elif message.type == web.WSMsgType.BINARY:
					self._enqueue(('binary', clientId, message.data, recvMs))
		finally:
			self._enqueue(('close', clientId))
		return ws

	def sendToClient(self, clientId, data, text):
		ws = self.connections.get(clientId)
		if ws is None or ws.closed:
			return
		send = ws.send_str(data) if text else ws.send_bytes(data)
		asyncio.run_coroutine_threadsafe(send, self.loop)

	def _enqueue(self, event):
		with self._lock:
			self._events.append(event)
		self._wake.set()

	def onFrameUploaded(self, data):
		"""Called from webSocketSendBinary: the TD frame number is in the header."""
		if len(data) < 16:
			return
		td = struct.unpack_from('<I', data, 12)[0]
		now = time.time() * 1000.0
		vi = self.source.sent.get(td) if self.source is not None else None
		self.frameLog[td] = {'td': td, 'vi': vi, 'sentAt': now, 'bytes': len(data)}
		self.inflight = td
		self.inflightSince = time.perf_counter()

	# ------------------------------------------------------------- main thread

	def _runFrames(self):
		period = 1.0 / self.fps
		origin = time.perf_counter()
		frameIndex = 0
		lastIndex = 0
		eventDriven = self.lockstep
		while self._running:
			if eventDriven:
				self._wake.wait(timeout=0.02)
				self._wake.clear()
				frameIndex = int((time.perf_counter() - origin) / period) + 1
			else:
				frameIndex = int((time.perf_counter() - origin) / period) + 1
				self.droppedFrames += max(0, frameIndex - lastIndex - 1)
				lastIndex = frameIndex
			self.env.absTime.frame = frameIndex
			self.env.absTime.seconds = frameIndex * period

			workStart = time.perf_counter()
			if self._nextHitch is not None and workStart >= self._nextHitch:
				self._nextHitch = workStart + self.hitchEverySec
				time.sleep(self.hitchMs / 1000.0)
			messages = self._cookFrame()
			self._feed()
			self._cookDownstream()
			self.frameWorkMs.append((time.perf_counter() - workStart) * 1000.0)
			self.msgsPerFrame.append(messages)
			self.cooked += 1

			if not eventDriven:
				delay = origin + frameIndex * period - time.perf_counter()
				if delay > 0:
					time.sleep(delay)

	def _feed(self):
		"""What the Execute DAT driving upload_frames.send_frame_u8_chw does each frame."""
		if self.uploader is None or self.aborted:
			return
		if self.lockstep:
			if self.inflight is not None:
				if time.perf_counter() - self.inflightSince > self.frameTimeoutSec:
					self.aborted = 'no result for TD frame {} after {:.0f}s'.format(
						self.inflight, self.frameTimeoutSec)
				return
			if self.source.exhausted:
				return
		start = time.perf_counter()
		try:
			self.uploader.send_frame_u8_chw()
		except Exception as error:
			self.exceptions.append('upload_frames: {!r}'.format(error))
			self.log('[exception] upload_frames: {!r}'.format(error))
		self.uploadMs.append((time.perf_counter() - start) * 1000.0)

	def _cookFrame(self):
		messages = 0
		while True:
			with self._lock:
				if not self._events:
					break
				event = self._events[0]
				if event[0] in ('text', 'binary'):
					if self.maxMsgsPerFrame and messages >= self.maxMsgsPerFrame:
						break
					messages += 1
				self._events.popleft()
			try:
				self._dispatch(event)
			except Exception as error:
				self.exceptions.append('{}: {!r}'.format(event[0], error))
				self.log('[exception] {}: {!r}'.format(event[0], error))
		return messages

	def _dispatch(self, event):
		kind = event[0]
		if kind == 'text':
			_, clientId, data, recvMs = event
			label = classifyText(data)
			if label == 'loaded':
				self.loadedAt = recvMs
			if label == 'text':
				self.log('[unclassified text from page] ' + data[:600])
			td = None
			key = None
			if label in RESULT_KINDS:
				match = FRAME.search(data[:200])
				td = int(match.group(1)) if match else None
				if td == 0xFFFFFFFF:  # webcam mode: the page has no TD frame id
					td = None
					video = VIDEO_FRAME.search(data[:300])
					key = 't{:08d}'.format(int(round(float(video.group(1)) * 1000))) if video else None
				elif td is not None:
					vi = self.frameLog.get(td, {}).get('vi')
					key = 'f{:05d}'.format(vi) if vi is not None else 'td{}'.format(td)
			self._ctx = {'td': td, 'key': key}
			start = time.perf_counter()
			self.server.onWebSocketReceiveText(self.webServerDAT, clientId, data)
			self._tally(label, len(data), (time.perf_counter() - start) * 1000.0,
				time.time() * 1000.0 - recvMs)
			if label in RESULT_KINDS:
				for name, summary in self.pendingMaps:
					self._record({'op': name, **summary})
				self.pendingMaps = []
				self._resultArrived(td, recvMs)
		elif kind == 'binary':
			_, clientId, data, recvMs = event
			label = classifyBinary(data)
			td = struct.unpack_from('<I', data, 12)[0] if len(data) >= 16 else None
			if td == 0xFFFFFFFF:
				key = None
			else:
				vi = self.frameLog.get(td, {}).get('vi')
				key = 'f{:05d}'.format(vi) if vi is not None else 'td{}'.format(td)
			self._ctx = {'td': td, 'key': key}
			start = time.perf_counter()
			self._recordSec = 0.0
			self.server.onWebSocketReceiveBinary(self.webServerDAT, clientId, data)
			self._tally(label, len(data), (time.perf_counter() - start - self._recordSec) * 1000.0,
				time.time() * 1000.0 - recvMs)
			entry = self.frameLog.get(td)
			if entry is not None:
				entry.setdefault(label + 'At', recvMs)
		elif kind == 'open':
			self.server.onWebSocketOpen(self.webServerDAT, event[1], event[2])
		elif kind == 'close':
			self.server.onWebSocketClose(self.webServerDAT, event[1])
			with self._lock:
				self.connections.pop(event[1], None)
		elif kind == 'http':
			_, method, path, query, future = event
			request = {'method': method, 'uri': path, 'pars': query, 'data': b''}
			response = {'statusCode': 404, 'statusReason': 'Not Found', 'data': b''}
			result = self.server.onHTTPRequest(self.webServerDAT, request, response) or response
			body = result.get('data') or b''
			if isinstance(body, str):
				body = body.encode()
			# the Web Server DAT serves a DAT's text as HTML when no type is set
			contentType = result.get('Content-Type') or guessType(path) or ('text/html' if path == '/' else None)
			self.loop.call_soon_threadsafe(
				future.set_result, (int(result.get('statusCode', 404)), contentType, bytes(body)))
		elif kind == 'serverStart':
			self.server.onServerStart(self.webServerDAT)

	def _tally(self, label, size, callbackMs, queueMs):
		stats = self.kinds[label]
		stats['count'] += 1
		stats['bytes'] += size
		stats['cb'].append(callbackMs)
		stats['q'].append(queueMs)

	def _resultArrived(self, td, recvMs):
		self.answered += 1
		self.resultTimes.append(recvMs)
		entry = self.frameLog.get(td)
		if entry is not None:
			entry['doneAt'] = recvMs
			entry['rtt'] = recvMs - entry['sentAt']
		if td is not None and td == self.inflight:
			self.inflight = None

	def _cookDownstream(self):
		"""Downstream DATs json.loads the predictions DAT once per cooked frame."""
		dirty, self._dirtyDats = self._dirtyDats, set()
		for name in dirty:
			start = time.perf_counter()
			try:
				json.loads(self.env.ops[name].text)
			except ValueError:
				pass
			self.downstreamParseMs[name].append((time.perf_counter() - start) * 1000.0)

	# ----------------------------------------------------------------- report

	def snapshot(self, warmup=3):
		"""Stats over answered frames after the first `warmup` (shader compile, JIT)."""
		elapsed = max(1e-6, time.time() - self.startedAt)
		frames = sorted(
			(f for f in self.frameLog.values() if 'doneAt' in f), key=lambda f: f['sentAt'])
		steady = frames[warmup:] if len(frames) > warmup + 2 else frames
		span = (steady[-1]['doneAt'] - steady[0]['sentAt']) / 1000.0 if len(steady) > 1 else 0
		sent = len(self.frameLog)
		times = self.resultTimes[warmup:] if len(self.resultTimes) > warmup + 2 else self.resultTimes
		intervals = [b - a for a, b in zip(times, times[1:])]
		resultFps = (len(times) - 1) * 1000.0 / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
		kinds = {}
		for label, stats in sorted(self.kinds.items()):
			kinds[label] = {
				'count': stats['count'],
				'avgKB': stats['bytes'] / max(1, stats['count']) / 1024.0,
				'callbackMs': summarize(stats['cb']),
				'queueMs': summarize(stats['q']),
			}
		stages = {}
		for label in ('segmentation', 'depth'):
			stages[label + 'Ms'] = summarize(
				[f[label + 'At'] - f['sentAt'] for f in steady if label + 'At' in f])
		return {
			'elapsedSec': elapsed,
			'sent': sent,
			'answered': self.answered,
			'droppedByBrowser': max(0, sent - self.answered - (1 if self.inflight is not None else 0)),
			'steadyFrames': len(steady),
			'throughputFps': (len(steady) - 1) / span if span > 0 else resultFps,
			'resultFps': resultFps,
			'resultIntervalMs': summarize(intervals),
			'rttMs': summarize([f['rtt'] for f in steady]),
			'stageMs': stages,
			'kinds': kinds,
			'tdFrames': {
				'cooked': self.cooked,
				'workMs': summarize(self.frameWorkMs),
				'overBudget': sum(1 for ms in self.frameWorkMs if ms > 1000.0 / self.fps),
				'uploadMs': summarize(self.uploadMs),
				'skipped': self.droppedFrames,
			},
			'dats': {
				name: {
					'writes': count,
					'downstreamParseMs': summarize(self.downstreamParseMs.get(name, [])),
				}
				for name, count in sorted(self.datWrites.items())
			},
			'maps': {
				name: {'commits': count, 'shape': self.mapShapes[name][0], 'dtype': self.mapShapes[name][1]}
				for name, count in sorted(self.mapCommits.items())
			},
			'scriptErrors': list(self.env.parentOp.errors),
			'exceptions': list(self.exceptions),
			'aborted': self.aborted,
		}
