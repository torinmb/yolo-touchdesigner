// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import { INPUT_W, INPUT_H } from "../config.js";
import {
    mapBoxYFlipNorm,
    mapAngleToBottomLeft,
    polygonFromXYWHR,
    normPolyYFlip,
    flipYKeypointsNorm,
} from "./math.js";

export function formatPredictions(frameId, seq, keepDet, keepPose, videoFrame) {
    const H = INPUT_H;

    const predsDet = keepDet.map((t) => {
        const box = mapBoxYFlipNorm(t.box, H);
        const out = {
            tx: box[0],
            ty: box[1],
            width: box[2],
            height: box[3],
            categoryName: [t.label],
            score: t.score,
            id: t.id,
        };

        if (typeof t.angle === "number" && Number.isFinite(t.angle)) {
            out.angleRadImage = t.angle;
            out.angleRad = mapAngleToBottomLeft(t.angle);
            out.angleDeg = out.angleRad * (180 / Math.PI);
            const polyImg = polygonFromXYWHR(t.box, t.angle);
            out.polygon = normPolyYFlip(polyImg, H);
        }
        return out;
    });

    const predsPose = keepPose.map((t) => {
        const box = mapBoxYFlipNorm(t.box, H);
        const kpts = flipYKeypointsNorm(t.keypoints, H);
        return {
            tx: box[0],
            ty: box[1],
            width: box[2],
            height: box[3],
            categoryName: [t.label],
            score: t.score,
            id: t.id,
            keypoints: kpts,
        };
    });

    return {
        frame: frameId >>> 0,
        seq: seq >>> 0,
        videoFrame,
        width: INPUT_W,
        height: INPUT_H,
        yolo: predsDet,
        yolo_pose: predsPose,
    };
}
