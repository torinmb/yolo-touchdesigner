// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import { IoUTracker } from "./utils/tracker.js";
import {
    DET_TRK_IOU,
    DET_TRK_TTL,
    POSE_TRK_IOU,
    POSE_TRK_TTL,
} from "./config.js";

export const trackerDet = new IoUTracker(DET_TRK_IOU, DET_TRK_TTL);
export const trackerPose = new IoUTracker(POSE_TRK_IOU, POSE_TRK_TTL);
