"""Minimal stand-ins for the TouchDesigner objects the YOLO callbacks use.

The real files in td_scripts/ (webserver1_callbacks.py, upload_frames.py) are
exec'd unchanged against these stubs, so the harness exercises exactly the code
TouchDesigner runs.
"""

import hashlib
import mimetypes
import time
import types
from pathlib import Path


class Sink:
	"""Accepts any attribute access, assignment or call (parent().par, ...)."""

	def __getattr__(self, name):
		return Sink()

	def __call__(self, *args, **kwargs):
		return Sink()


class FakeTextDAT:
	def __init__(self, name, onWrite=None):
		self.name = name
		self._text = ''
		self._onWrite = onWrite

	@property
	def text(self):
		return self._text

	@text.setter
	def text(self, value):
		self._text = value if isinstance(value, str) else str(value)
		if self._onWrite:
			self._onWrite(self.name, self._text)


class FakeScriptTOP:
	"""A Script TOP / Touch In. copyNumpyArray is what the callbacks call."""

	def __init__(self, name, onCopy=None):
		self.name = name
		self.array = None
		self._onCopy = onCopy

	def copyNumpyArray(self, array):
		self.array = array
		if self._onCopy:
			self._onCopy(self.name, array)


class FakeFile:
	def __init__(self, name, path):
		self.name = name
		self._path = path

	@property
	def byteArray(self):
		return self._path.read_bytes()


class FakeVFS:
	"""Virtual File System keyed by bare filename, like the tox's op('virtualFile').vfs.
	Built from a vite build directory so the real onHTTPRequest serves the
	packaged app."""

	def __init__(self, root=None):
		self._files = {}
		if root:
			for path in Path(root).rglob('*'):
				if path.is_file() and path.name != 'index.html':
					self._files.setdefault(path.name, path)

	def __getitem__(self, name):
		path = self._files.get(name)
		return FakeFile(name, path) if path else None


class FakeParent:
	def __init__(self, log):
		self.errors = []
		self._log = log

	def addScriptError(self, message):
		self.errors.append(message)
		self._log('[scriptError] ' + message)

	def clearScriptErrors(self, *args, **kwargs):
		self.errors = []

	def __getattr__(self, name):
		return Sink()


class OpResolver:
	"""op('a/b/c') resolves by its last path component, so op('yolo_server/webserver1'),
	op('../segmentation_data') and op('webserver1') all find the one fake operator.
	op.TDResources deliberately raises, as when the Thread Manager is missing,
	which makes upload_frames.py take its synchronous packing path."""

	def __init__(self, ops):
		self._ops = ops

	def __call__(self, path):
		return self._ops.get(str(path).rstrip('/').split('/')[-1])

	def __getattr__(self, name):
		raise AttributeError(name)


class TDEnvironment:
	"""One TD component's worth of operators, shared by its callback DATs."""

	TEXT_DATS = ('predictions', 'webcam_list', 'tick', 'status', 'active_client', 'frame')
	TOPS = ('segmentation_data', 'depth_data')

	def __init__(self, log=print, onDatWrite=None, onTopCopy=None, distDir=None):
		self.log = log
		self.absTime = types.SimpleNamespace(frame=1, seconds=0.0)
		self.parentOp = FakeParent(log)
		self.ops = {}
		for name in self.TEXT_DATS:
			self.ops[name] = FakeTextDAT(name, onDatWrite)
		for name in self.TOPS:
			self.ops[name] = FakeScriptTOP(name, onTopCopy)
		self.ops['virtualFile'] = types.SimpleNamespace(vfs=FakeVFS(distDir))
		index = Path(distDir) / 'index.html' if distDir else None
		self.ops['index'] = types.SimpleNamespace(
			text=index.read_text() if index and index.exists() else '')
		self.op = OpResolver(self.ops)

	def debug(self, *args):
		self.log('[debug] ' + ' '.join(str(arg) for arg in args))

	def loadCallbacks(self, path, extraGlobals=None):
		path = Path(path)
		module = types.ModuleType(path.stem)
		module.__file__ = str(path)
		module.__dict__.update(
			op=self.op,
			me=Sink(),
			absTime=self.absTime,
			debug=self.debug,
			parent=lambda *args: self.parentOp,
			run=lambda *args, **kwargs: Sink(),
			mod=lambda name: Sink(),
			print=lambda *args, **kwargs: self.log(' '.join(str(a) for a in args)),
		)
		if extraGlobals:
			module.__dict__.update(extraGlobals)
		# utf-8-sig: upload_frames.py starts with a BOM
		source = path.read_text(encoding='utf-8-sig')
		exec(compile(source, str(path), 'exec'), module.__dict__)
		return module


def arrayDigest(array):
	return hashlib.sha1(array.tobytes()).hexdigest()[:12]


def guessType(name):
	return mimetypes.guess_type(name, strict=False)[0]


def now_ms():
	return time.time() * 1000.0
