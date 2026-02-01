// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import {
    runDetect,
    runPose,
    detSession,
    poseSession,
} from "./inference/onnx.js";
import { trackerDet, trackerPose } from "./state.js";
import { formatPredictions } from "./utils/protocol.js";

export async function runInferencePipeline(
    inputTensor,
    frameId,
    seq,
    videoFrame,
    sender,
) {
    // 1. Run Inference
    // usage of sessions is guarded by checks in runDetect/runPose,
    // but we check existence here to determine "active" streams for tracking/sending

    let keepDet = [];
    let keepPose = [];

    if (detSession) keepDet = await runDetect(inputTensor);
    if (poseSession) keepPose = await runPose(inputTensor);

    // 2. Update Trackers
    // Only update trackers if the corresponding model is active.
    // Use empty array if active but no detections found.
    const tracksDet = detSession ? trackerDet.update(keepDet) : [];
    const tracksPose = poseSession ? trackerPose.update(keepPose) : [];

    // 3. Format Output
    const msg = formatPredictions(
        frameId,
        seq,
        tracksDet,
        tracksPose,
        videoFrame,
    );

    // 4. Send Message via Callback
    if (sender) {
        if (detSession && poseSession) {
            sender({ ...msg, type: "yolo_combined" });
        } else if (detSession) {
            // Trim unused fields for bandwidth
            delete msg.yolo_pose;
            sender({ ...msg, type: "yolo", predictions: msg.yolo });
        } else if (poseSession) {
            delete msg.yolo;
            sender({ ...msg, type: "yolo_pose", predictions: msg.yolo_pose });
        }
    }
}
