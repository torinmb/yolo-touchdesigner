// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import { FLIP_HORIZONTAL } from "./config.js";

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

    // Use CSS class for flipping instead of inline styles
    if (FLIP_HORIZONTAL) {
        v.classList.add("flipped");
    } else {
        v.classList.remove("flipped");
    }

    return v;
}
