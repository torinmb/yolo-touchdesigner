"""Compare two recordings (the JSONL files `replay` writes) frame by frame.

Records are keyed by video frame ('f00012'), so two runs of the same clip line up
exactly no matter how fast each run was. Exit status 1 means "different".
"""

import base64
import json
from pathlib import Path

import numpy as np

DEFAULT_TOL = {
	'box': 1e-3,       # normalized units (fraction of the 640px frame)
	'score': 1e-3,
	'keypoint': 2e-3,
	'angle': 1e-3,
	'seg_ids': 0.995,  # min share of sampled pixels with the same class id
	'seg_alpha': 0.02,  # max mean |alpha diff| over sampled pixels
	'depth_abs': 0.02,  # max mean |diff| in meters over sampled pixels
	'depth_max': 0.25,  # max single-pixel |diff| in meters
	'count_frames': 0.02,  # share of frames allowed to have a different detection count
}


def load(path):
	frames = {}
	for line in Path(path).read_text().splitlines():
		entry = json.loads(line)
		key = entry.get('key')
		if key is None:
			continue
		frames.setdefault(key, {})[entry['op']] = entry
	return frames


def normalized(text):
	"""Predictions text without the TD frame counters, which depend on run timing."""
	data = json.loads(text)
	data.pop('frame', None)
	data.pop('seq', None)
	return json.dumps(data, sort_keys=True)


def decodeSample(entry):
	data = np.frombuffer(base64.b64decode(entry['sub']), dtype='<f4')
	return data.reshape(entry['subshape'])


def keypointList(det):
	return [(k['x'], k['y'], k['score']) for k in det.get('keypoints') or []]


def matchDetections(a, b):
	"""Greedy match by box distance and label; returns (pairs, unmatchedA, unmatchedB)."""
	remaining = list(range(len(b)))
	pairs, lonely = [], []
	order = sorted(range(len(a)), key=lambda i: -a[i].get('score', 0))
	for i in order:
		best, bestDist = None, None
		for j in remaining:
			if a[i].get('categoryName') != b[j].get('categoryName'):
				continue
			dist = sum(abs(a[i][k] - b[j][k]) for k in ('tx', 'ty', 'width', 'height'))
			if bestDist is None or dist < bestDist:
				best, bestDist = j, dist
		if best is None or bestDist > 0.25:
			lonely.append(i)
		else:
			remaining.remove(best)
			pairs.append((i, best))
	return pairs, lonely, remaining


def diffPredictions(ta, tb):
	pa, pb = json.loads(ta), json.loads(tb)
	out = {'schema': 0, 'count': 0, 'unmatched': 0, 'id': 0, 'box': 0.0, 'score': 0.0,
		'keypoint': 0.0, 'angle': 0.0, 'where': ''}
	if sorted(pa) != sorted(pb) or pa.get('type') != pb.get('type'):
		out['schema'] = 1
		out['where'] = 'keys {} vs {}'.format(sorted(pa), sorted(pb))
		return out
	if (pa.get('width'), pa.get('height')) != (pb.get('width'), pb.get('height')):
		out['schema'] = 1
		out['where'] = 'frame size differs'
	for name in ('yolo', 'yolo_pose'):
		la, lb = pa.get(name) or [], pb.get(name) or []
		if len(la) != len(lb):
			out['count'] += 1
			out['where'] = out['where'] or '{} count {} vs {}'.format(name, len(la), len(lb))
		pairs, lonelyA, lonelyB = matchDetections(la, lb)
		out['unmatched'] += len(lonelyA) + len(lonelyB)
		for i, j in pairs:
			x, y = la[i], lb[j]
			if x.get('id') != y.get('id'):
				out['id'] += 1
			out['box'] = max(out['box'], max(abs(x[k] - y[k]) for k in ('tx', 'ty', 'width', 'height')))
			out['score'] = max(out['score'], abs(x['score'] - y['score']))
			if 'angleRad' in x and 'angleRad' in y:
				out['angle'] = max(out['angle'], abs(x['angleRad'] - y['angleRad']))
			kx, ky = keypointList(x), keypointList(y)
			if len(kx) != len(ky):
				out['schema'] = 1
			for p, q in zip(kx, ky):
				out['keypoint'] = max(out['keypoint'], abs(p[0] - q[0]), abs(p[1] - q[1]))
	return out


def diffMap(op, ea, eb):
	if ea['shape'] != eb['shape'] or ea['dtype'] != eb['dtype']:
		return {'schema': 1, 'where': 'shape/dtype {} {} vs {} {}'.format(
			ea['shape'], ea['dtype'], eb['shape'], eb['dtype'])}
	a, b = decodeSample(ea), decodeSample(eb)
	delta = np.abs(a - b)
	out = {'schema': 0, 'meanAbs': float(delta.mean()), 'maxAbs': float(delta.max()),
		'identical': ea['sha1'] == eb['sha1']}
	if op == 'segmentation_data':
		ida, idb = np.floor(a), np.floor(b)
		out['idAgree'] = float((ida == idb).mean())
		same = (ida == idb) & (ida > 0)
		out['alphaDiff'] = float(np.abs((a - ida) - (b - idb))[same].mean()) if same.any() else 0.0
	return out


def compare(pathA, pathB, tol=None, exact=False, quiet=False):
	tol = {**DEFAULT_TOL, **(tol or {})}
	A, B = load(pathA), load(pathB)
	common = sorted(set(A) & set(B))
	result = {'a': str(pathA), 'b': str(pathB), 'framesA': len(A), 'framesB': len(B),
		'common': len(common), 'failures': [], 'worst': {}}
	fails = result['failures']
	if not common:
		fails.append('no frames in common ({} vs {})'.format(len(A), len(B)))
	if len(A) != len(B):
		fails.append('frame count differs: {} vs {} ({} only in A, {} only in B)'.format(
			len(A), len(B), len(set(A) - set(B)), len(set(B) - set(A))))

	worst = result['worst']
	countFrames = idFrames = unmatchedFrames = 0
	identicalPred = identicalMaps = mapFrames = 0
	firstDiff = None
	opsMismatch = []
	for key in common:
		ra, rb = A[key], B[key]
		if set(ra) != set(rb):
			opsMismatch.append((key, sorted(ra), sorted(rb)))
			continue
		frameDiff = None
		for op in ra:
			if op == 'predictions':
				same = normalized(ra[op]['text']) == normalized(rb[op]['text'])
				identicalPred += same
				d = diffPredictions(ra[op]['text'], rb[op]['text'])
				countFrames += d['count'] > 0
				idFrames += d['id'] > 0
				unmatchedFrames += d['unmatched'] > 0
				if d['schema']:
					fails.append('{}: predictions schema differs: {}'.format(key, d['where']))
				for field in ('box', 'score', 'keypoint', 'angle'):
					worst[field] = max(worst.get(field, 0.0), d[field])
				if not same and frameDiff is None and (
						d['count'] or d['unmatched'] or d['id'] or d['box'] > 0 or d['score'] > 0):
					frameDiff = '{} predictions: {}'.format(key, {k: v for k, v in d.items() if v})
			else:
				mapFrames += 1
				d = diffMap(op, ra[op], rb[op])
				if d['schema']:
					fails.append('{}: {} {}'.format(key, op, d['where']))
					continue
				identicalMaps += d['identical']
				if op == 'segmentation_data':
					worst['seg_ids_min'] = min(worst.get('seg_ids_min', 1.0), d['idAgree'])
					worst['seg_alpha'] = max(worst.get('seg_alpha', 0.0), d['alphaDiff'])
				else:
					worst['depth_abs'] = max(worst.get('depth_abs', 0.0), d['meanAbs'])
					worst['depth_max'] = max(worst.get('depth_max', 0.0), d['maxAbs'])
				if not d['identical'] and frameDiff is None:
					frameDiff = '{} {}: {}'.format(key, op, {k: round(v, 5) for k, v in d.items()
						if isinstance(v, float)})
		if frameDiff and firstDiff is None:
			firstDiff = frameDiff

	if opsMismatch:
		key, opsA, opsB = opsMismatch[0]
		fails.append('{} frames have different output sets, e.g. {}: A {} vs B {}'.format(
			len(opsMismatch), key, opsA, opsB))
	n = max(1, len(common))
	result.update({
		'identicalPredictionFrames': identicalPred,
		'identicalMapFrames': identicalMaps,
		'mapFrames': mapFrames,
		'countMismatchFrames': countFrames,
		'idMismatchFrames': idFrames,
		'unmatchedFrames': unmatchedFrames,
		'firstDivergence': firstDiff,
	})
	if exact:
		if identicalPred != n - sum(1 for k in common if 'predictions' not in A[k]):
			fails.append('predictions not byte-identical in {} frames'.format(
				n - identicalPred))
		if identicalMaps != mapFrames:
			fails.append('maps not bit-identical in {} of {} map frames'.format(
				mapFrames - identicalMaps, mapFrames))
	else:
		if countFrames / n > tol['count_frames']:
			fails.append('detection count differs in {}/{} frames'.format(countFrames, n))
		if unmatchedFrames / n > tol['count_frames']:
			fails.append('unmatched detections in {}/{} frames'.format(unmatchedFrames, n))
		if idFrames / n > tol['count_frames']:
			fails.append('track ids differ in {}/{} frames'.format(idFrames, n))
		for field, limit in (('box', tol['box']), ('score', tol['score']),
				('keypoint', tol['keypoint']), ('angle', tol['angle']),
				('seg_alpha', tol['seg_alpha']), ('depth_abs', tol['depth_abs']),
				('depth_max', tol['depth_max'])):
			if worst.get(field, 0.0) > limit:
				fails.append('worst {} diff {:.5f} > {}'.format(field, worst[field], limit))
		if worst.get('seg_ids_min', 1.0) < tol['seg_ids']:
			fails.append('class ids agree only {:.4f} < {}'.format(worst['seg_ids_min'], tol['seg_ids']))
	result['ok'] = not fails
	if not quiet:
		printCompare(result, exact)
	return result


def printCompare(result, exact):
	print('compare {} vs {} ({})'.format(
		Path(result['a']).name, Path(result['b']).name, 'exact' if exact else 'tolerant'))
	print('  frames: A {} B {} common {} | byte-identical predictions {} | bit-identical maps {}/{}'.format(
		result['framesA'], result['framesB'], result['common'],
		result['identicalPredictionFrames'], result['identicalMapFrames'], result['mapFrames']))
	worst = result['worst']
	if worst:
		print('  worst diffs: ' + ' '.join('{}={:.5f}'.format(k, v) for k, v in sorted(worst.items())))
	if result['firstDivergence']:
		print('  first divergence: ' + result['firstDivergence'][:300])
	for line in result['failures'][:12]:
		print('  FAIL ' + line)
	print('  RESULT: {}'.format('SAME' if result['ok'] else 'DIFFERENT'))
