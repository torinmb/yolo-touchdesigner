// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import {
    DEV_MODE,
    FLIP_HORIZONTAL,
    FLIP_VERTICAL,
    WEBCAM_ROTATION_DEG,
} from "./config.js";
import { formatCssTransform, isQuarterTurn } from "./utils/orientation.js";

const statusEl =
    document.getElementById("status") ||
    (() => {
        const el = document.createElement("div");
        el.id = "status";
        document.body.appendChild(el);
        return el;
    })();

export function setStatus(msg) {
    if (statusEl) {
        if (!DEV_MODE) {
            statusEl.style.display = "none";
            return;
        }
        statusEl.textContent = msg || "";
        statusEl.style.display = msg ? "" : "none";
    }
}

export function ensureVisibleVideo() {
    let v = document.getElementById("video");
    if (!v) {
        v = document.createElement("video");
        v.id = "video";
        v.autoplay = true;
        v.playsInline = true;
        v.muted = true;
        document.body.appendChild(v);
    }

    const orientationTransform = formatCssTransform(
        FLIP_HORIZONTAL,
        FLIP_VERTICAL,
        WEBCAM_ROTATION_DEG,
    );
    const quarterTurn = isQuarterTurn(WEBCAM_ROTATION_DEG);

    v.style.inset = "auto";
    v.style.left = "50%";
    v.style.top = "50%";
    v.style.width = quarterTurn ? "100vh" : "100vw";
    v.style.height = quarterTurn ? "100vw" : "100vh";
    v.style.transform =
        orientationTransform === "none"
            ? "translate(-50%, -50%)"
            : `translate(-50%, -50%) ${orientationTransform}`;

    return v;
}
