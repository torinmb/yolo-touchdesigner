// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import { iou } from "./math.js";

export class IoUTracker {
    constructor(iouMatch = 0.5, ttl = 1) {
        this.iouMatch = iouMatch;
        this.ttl = ttl;
        this.nextId = 1;
        this.tracks = new Map();
    }

    _copyExtras(dst, src) {
        for (const k in src) {
            if (
                k === "box" ||
                k === "label" ||
                k === "score" ||
                k === "keypoints"
            )
                continue;
            dst[k] = src[k]; // e.g., angle, polygon, custom fields
        }
    }

    update(dets) {
        const ids = [...this.tracks.keys()];
        for (const id of ids) this.tracks.get(id).miss++;

        // Pairwise IoU
        const pairs = [];
        for (let di = 0; di < dets.length; di++) {
            for (const id of ids) {
                const conf = iou(this.tracks.get(id).box, dets[di].box);
                pairs.push([conf, id, di]);
            }
        }
        pairs.sort((a, b) => b[0] - a[0]);

        const takenTrack = new Set();
        const takenDet = new Set();
        for (const [conf, id, di] of pairs) {
            if (conf < this.iouMatch) break;
            if (takenTrack.has(id) || takenDet.has(di)) continue;
            const t = this.tracks.get(id),
                d = dets[di];
            t.box = d.box;
            t.label = d.label;
            t.score = d.score;
            t.keypoints = d.keypoints;
            this._copyExtras(t, d); // preserve angle & extras
            t.hits++;
            t.miss = 0;
            takenTrack.add(id);
            takenDet.add(di);
        }

        // New tracks
        for (let di = 0; di < dets.length; di++) {
            if (takenDet.has(di)) continue;
            const d = dets[di];
            const id = this.nextId++;
            const t = {
                box: d.box,
                label: d.label,
                score: d.score,
                keypoints: d.keypoints,
                age: 0,
                hits: 1,
                miss: 0,
            };
            this._copyExtras(t, d);
            this.tracks.set(id, t);
        }

        // Age & prune
        for (const id of [...this.tracks.keys()]) {
            const t = this.tracks.get(id);
            t.age++;
            if (t.miss > this.ttl) this.tracks.delete(id);
        }

        // Output (preserve extras)
        return [...this.tracks.entries()]
            .filter(([_, t]) => t.miss === 0)
            .map(([id, t]) => {
                const out = {
                    id,
                    box: t.box,
                    label: t.label,
                    score: t.score,
                    keypoints: t.keypoints,
                };
                this._copyExtras(out, t);
                return out;
            });
    }
}
