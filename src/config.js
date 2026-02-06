// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

export const qs = new URLSearchParams(location.search);

const boolish = (v, def = false) =>
    v == null ? def : /^(1|true|on|yes)$/i.test(String(v));

const pick = (...names) => {
    for (const n of names) if (n && qs.has(n)) return qs.get(n);
    return null;
};

export const getStr = (names, fallback) => {
    const v = pick(...names);
    return v != null ? v : fallback;
};

export const getBool = (names, fallback) => {
    const v = pick(...names);
    return v != null ? boolish(v, fallback) : fallback;
};

export const getNum = (primaryNames, fallback, commonFallbackName) => {
    for (const n of primaryNames) {
        if (qs.has(n)) {
            const x = parseFloat(qs.get(n));
            if (!Number.isNaN(x)) return x;
        }
    }
    if (commonFallbackName && qs.has(commonFallbackName)) {
        const x = parseFloat(qs.get(commonFallbackName));
        if (!Number.isNaN(x)) return x;
    }
    return fallback;
};

export const getInt = (primaryNames, fallback, commonFallbackName) => {
    for (const n of primaryNames) {
        if (qs.has(n)) {
            const x = parseInt(qs.get(n), 10);
            if (!Number.isNaN(x)) return x;
        }
    }
    if (commonFallbackName && qs.has(commonFallbackName)) {
        const x = parseInt(qs.get(commonFallbackName), 10);
        if (!Number.isNaN(x)) return x;
    }
    return fallback;
};

/* ================================
   Global config / toggles
================================ */
export const WS_PORT = qs.get("wsPort") || "62309";
export const USE_BINARY = getBool(["binary"], false);
export const USE_CPU = getBool(["cpu", "CPU"], false);

// Stream toggles + models
export let ENABLE_DET = getBool(
    ["Objecttrackingenabled", "ObjectTrackingEnabled"],
    true,
);
export let ENABLE_POSE = getBool(
    ["Posetrackingenabled", "pose", "PoseTrackingEnabled"],
    true,
);
export let ENABLE_SEG = getBool(
    ["Segmentationenabled", "segmentation", "SegmentationEnabled"],
    false,
);

export let MODEL_DETECT_KEY = getStr(
    ["Objecttrackingmodel", "Obecttrackingmodel", "ObjectTrackingModel"],
    "yolo11n",
);
export let MODEL_POSE_KEY = getStr(["Posemodel", "PoseModel"], "yolo11n-pose");
export let MODEL_SEG_KEY = getStr(
    ["Segmentationmodel", "SegmentationModel"],
    "yolo26n-seg",
);

// Legacy single `model=` inference logic
const legacyModel = qs.get("model");
const anyToggleProvided = [
    "Poseenabled",
    "pose",
    "Objecttrackingenabled",
    "detect",
].some((k) => qs.has(k));

if (legacyModel && !anyToggleProvided) {
    if (/pose/i.test(legacyModel)) {
        ENABLE_POSE = true;
        ENABLE_DET = false;
        MODEL_POSE_KEY = legacyModel;
    } else {
        ENABLE_DET = true;
        ENABLE_POSE = false;
        MODEL_DETECT_KEY = legacyModel;
    }
}

// Per-task thresholds
export const DET_SCORE_T = getNum(["Detscoret"], 0.4, "Scoret");
export const DET_IOU_T = getNum(["Detiout"], 0.45, "Iout");
export const DET_TOPK = getInt(["Dettopk"], 100, "Topk");

export const SEG_SCORE_T = getNum(["Segscoret"], 0.2, "Scoret");
export const SEG_TOPK = getInt(["Segtopk"], 100, "Topk");

export const POSE_SCORE_T = getNum(["Posescoret"], 0.35, "Scoret");
export const POSE_IOU_T = getNum(["Poseiout"], 0.45, "Iout");
export const POSE_TOPK = getInt(["Posetopk"], 50, "Topk");

// Tracker settings
export const DET_TRK_IOU = getNum(["Detrkiou"], 0.5, "Trkiou");
export const DET_TRK_TTL = getInt(["Detrkttl"], 2, "Trkttl");
export const POSE_TRK_IOU = getNum(["Posetrkiou"], 0.5, "Trkiou");
export const POSE_TRK_TTL = getInt(["Posetrkttl"], 2, "Trkttl");

// Webcam options
export const WEBCAM_LABEL = getStr(["webcamLabel"], null);
export const FLIP_HORIZONTAL = getBool(["flipHorizontal"], true);

// Constants
export const INPUT_W = 640;
export const INPUT_H = 640;
