"""Independent reference: the same frames through onnxruntime (CPU, Python) and a
line-for-line port of the page's post-processing, written as a recording that
`compare` understands. It answers "is the browser pipeline (WebGPU preprocessing,
decode, tracker, protocol) still producing what the model says?" without trusting
any browser code.

Covers the NMS-free YOLO26 models: det (yolo26n), pose (yolo26n-pose) and depth.
(yolo11 needs the yolo-decoder.onnx NMS graph and seg needs the proto matmul; use
`ab` / `check --against` for those.)
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from fake_td import REPO
from frames import loadFrames
from fake_td import mapSummary

INPUT = 640
DEFAULTS = {'det': 0.4, 'pose': 0.35}
TOPK = {'det': 100, 'pose': 50}
TRACK_IOU = 0.5
TRACK_TTL = 2


def iou(a, b):
	ax, ay, aw, ah = a
	bx, by, bw, bh = b
	iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
	ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
	inter = iw * ih
	return inter / (aw * ah + bw * bh - inter + 1e-9)


class IoUTracker:
	"""Port of src/utils/tracker.js."""

	def __init__(self, iouMatch=TRACK_IOU, ttl=TRACK_TTL):
		self.iouMatch, self.ttl, self.nextId, self.tracks = iouMatch, ttl, 1, {}

	def update(self, dets):
		ids = list(self.tracks)
		for i in ids:
			self.tracks[i]['miss'] += 1
		pairs = sorted(
			((iou(self.tracks[i]['box'], d['box']), i, di) for di, d in enumerate(dets) for i in ids),
			key=lambda p: -p[0])  # stable like Array.sort
		takenTrack, takenDet = set(), set()
		for conf, i, di in pairs:
			if conf < self.iouMatch:
				break
			if i in takenTrack or di in takenDet:
				continue
			t = self.tracks[i]
			t.update(box=dets[di]['box'], label=dets[di]['label'], score=dets[di]['score'],
				keypoints=dets[di].get('keypoints'), miss=0)
			t['hits'] += 1
			takenTrack.add(i)
			takenDet.add(di)
		for di, d in enumerate(dets):
			if di not in takenDet:
				self.tracks[self.nextId] = dict(box=d['box'], label=d['label'], score=d['score'],
					keypoints=d.get('keypoints'), age=0, hits=1, miss=0)
				self.nextId += 1
		for i in list(self.tracks):
			self.tracks[i]['age'] += 1
			if self.tracks[i]['miss'] > self.ttl:
				del self.tracks[i]
		return [dict(id=i, **t) for i, t in self.tracks.items() if t['miss'] == 0]


def decode(output, threshold, topk, pose):
	rows = output[0]
	out = []
	for row in rows:
		score = float(row[4])
		if score < threshold:
			continue
		x1, y1, x2, y2 = (float(v) for v in row[:4])
		det = {'box': [x1, y1, x2 - x1, y2 - y1], 'label': float(row[5]), 'score': score}
		if pose:
			det['keypoints'] = [{'x': float(row[6 + k * 3]), 'y': float(row[7 + k * 3]),
				'score': float(row[8 + k * 3])} for k in range((row.shape[0] - 6) // 3)]
		out.append(det)
	out.sort(key=lambda d: -d['score'])
	return out[:topk]


def formatTracks(tracks, pose, size=INPUT):
	out = []
	for t in tracks:
		x, y, w, h = t['box']
		box = [x / size, (size - (y + h)) / size, w / size, h / size]
		entry = {
			'tx': box[0] + box[2] * 0.5 if not pose else box[0],
			'ty': box[1] + box[3] * 0.5 if not pose else box[1],
			'width': box[2], 'height': box[3],
			'categoryName': [t['label']], 'score': t['score'], 'id': t['id'],
		}
		if pose:
			entry['keypoints'] = [{'x': k['x'] / size, 'y': (size - k['y']) / size, 'score': k['score']}
				for k in t['keypoints']]
		out.append(entry)
	return out


def session(name):
	path = REPO / 'public' / 'models' / '{}.onnx'.format(name)
	return ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])


def writeReference(video, frames, features, models, out, start=0.0, stride=1, fit='letterbox',
		scoreT=None, mapStride=10):
	clip = loadFrames(video, frames, start=start, stride=stride, fit=fit)
	sessions = {f: session(models.get(f, {'det': 'yolo26n', 'pose': 'yolo26n-pose', 'depth': 'yolo26n-depth'}[f]))
		for f in features if f in ('det', 'pose', 'depth')}
	unsupported = [f for f in features if f not in sessions]
	if unsupported:
		raise SystemExit('reference supports det, pose, depth (not {})'.format(', '.join(unsupported)))
	trackers = {f: IoUTracker() for f in ('det', 'pose')}
	lines = []
	for vi in range(len(clip)):
		x = (np.ascontiguousarray(clip[vi].transpose(2, 0, 1), dtype=np.float32) / 255.0)[None]
		key = 'f{:05d}'.format(vi)
		result = {'frame': vi, 'seq': vi, 'videoFrame': 0, 'width': INPUT, 'height': INPUT}
		tracks = {}
		for feature in ('det', 'pose'):
			if feature in sessions:
				s = sessions[feature]
				head = s.run(None, {s.get_inputs()[0].name: x})[0]
				dets = decode(head, (scoreT or {}).get(feature, DEFAULTS[feature]), TOPK[feature], feature == 'pose')
				tracks[feature] = formatTracks(trackers[feature].update(dets), feature == 'pose')
			else:
				tracks[feature] = None
		if 'det' in sessions and 'pose' in sessions:
			result.update(yolo=tracks['det'], yolo_pose=tracks['pose'], type='yolo_combined')
		elif 'det' in sessions:
			result.update(yolo=tracks['det'], type='yolo', predictions=tracks['det'])
		elif 'pose' in sessions:
			result.update(yolo_pose=tracks['pose'], type='yolo_pose', predictions=tracks['pose'])
		else:
			result.update(yolo=[], type='yolo', predictions=[])
		lines.append({'op': 'predictions', 'td': vi, 'key': key, 'text': json.dumps(result)})
		if 'depth' in sessions:
			s = sessions['depth']
			depth = s.run(None, {s.get_inputs()[0].name: x})[0].reshape(INPUT, INPUT, 1).astype(np.float32)
			lines.append({'op': 'depth_data', 'td': vi, 'key': key, **mapSummary(depth, mapStride)})
	Path(out).parent.mkdir(parents=True, exist_ok=True)
	Path(out).write_text('\n'.join(json.dumps(line) for line in lines) + '\n')
	return len(clip)
