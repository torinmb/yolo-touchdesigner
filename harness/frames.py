"""Decoded video frames, and the fake `source` TOP that upload_frames.py reads."""

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parent
CACHE = HARNESS / 'out' / 'frames'


def loadFrames(video, count, size=640, start=0.0, stride=1, fit='letterbox'):
	"""First `count` frames (every `stride`th) of any clip, as (N, size, size, 3) uint8 RGB.
	Decoded once with ffmpeg and cached as .npy, keyed on source path/size/mtime + params."""
	path = Path(video).expanduser().resolve()
	if not path.is_file():
		sys.exit('video not found: {}'.format(video))
	stat = path.stat()
	key = hashlib.sha1('{}|{}|{}|{}|{}|{}|{}|{}'.format(
		path, stat.st_size, stat.st_mtime_ns, count, size, start, stride, fit).encode()).hexdigest()[:10]
	cached = CACHE / '{}-{}.npy'.format(path.stem, key)
	if cached.exists():
		return np.load(cached, mmap_mode='r')

	scale = (
		'scale={0}:{0}:force_original_aspect_ratio=decrease,pad={0}:{0}:(ow-iw)/2:(oh-ih)/2'.format(size)
		if fit == 'letterbox' else 'scale={0}:{0}'.format(size))
	vf = scale + (',select=not(mod(n\\,{}))'.format(stride) if stride > 1 else '')
	command = ['ffmpeg', '-v', 'error', '-ss', str(start), '-i', str(path), '-an', '-vf', vf,
		'-vsync', '0', '-frames:v', str(count), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
	began = time.time()
	raw = subprocess.run(command, stdout=subprocess.PIPE, check=False)
	if raw.returncode != 0 or not raw.stdout:
		sys.exit('ffmpeg could not decode {} (is ffmpeg installed?)'.format(path.name))
	frames = np.frombuffer(raw.stdout, dtype=np.uint8)
	frames = frames[:frames.size // (size * size * 3) * size * size * 3].reshape(-1, size, size, 3)
	CACHE.mkdir(parents=True, exist_ok=True)
	np.save(cached, frames)
	print('decoded {} frames of {} in {:.1f}s (cached)'.format(len(frames), path.name, time.time() - began))
	return np.load(cached, mmap_mode='r')


class SourceTOP:
	"""Stands in for op('source'). numpyArray() hands upload_frames.py the next
	frame in the layout a real network would produce:

	  packed     (3H, W, 1) float32, planes stacked vertically (td_scripts/repack.glsl)
	  packed-u8  same, uint8 (zero-copy path in upload_frames.py)
	  rgb        (H, W, 3) float32, interleaved (CPU fallback path)

	pace='asap' walks the clip in order, one frame per call, and stops after the
	last. pace='realtime' returns whichever frame the wall clock says is current
	(looping), so a slow pipeline skips frames like TD does.
	"""

	def __init__(self, frames, absTime, fmt='packed', pace='asap', videoFps=30.0):
		self.frames = frames
		self.absTime = absTime
		self.fmt = fmt
		self.pace = pace
		self.videoFps = videoFps
		self.cursor = 0
		self.sent = {}  # TD frame number -> video frame index
		self.exhausted = False
		self._t0 = None

	def numpyArray(self, delayed=False):
		count = len(self.frames)
		if self.pace == 'asap':
			if self.cursor >= count:
				self.exhausted = True
				return None
			vi = self.cursor
			self.cursor += 1
		else:
			if self._t0 is None:
				self._t0 = time.perf_counter()
			vi = int((time.perf_counter() - self._t0) * self.videoFps) % count
		self.sent[int(self.absTime.frame)] = vi
		return self._convert(self.frames[vi])

	def _convert(self, rgb):
		if self.fmt == 'rgb':
			return rgb.astype(np.float32) / 255.0
		height, width = rgb.shape[:2]
		planar = np.ascontiguousarray(rgb.transpose(2, 0, 1)).reshape(3 * height, width, 1)
		return planar if self.fmt == 'packed-u8' else planar.astype(np.float32) / 255.0
