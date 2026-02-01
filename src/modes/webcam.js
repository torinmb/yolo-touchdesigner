// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import { INPUT_W, INPUT_H, FLIP_HORIZONTAL, WEBCAM_LABEL } from "../config.js";
import { detSession, poseSession } from "../inference/onnx.js";
import { toInputTensorFromImageData } from "../inference/io.js";
import { runInferencePipeline } from "../pipeline.js";
import { setStatus, ensureVisibleVideo } from "../ui.js";

// Offscreen buffer
const procCanvas = document.createElement("canvas");
procCanvas.width = INPUT_W;
procCanvas.height = INPUT_H;
const procCtx = procCanvas.getContext("2d", { willReadFrequently: true });

let rafProcessing = false;
let prevT = 0;
let lastVideoMediaTime = 0;
let _sender = null;
let _videoEl = null;

export function setWebSocketSender(senderFn) {
    _sender = senderFn;
}

export async function listWebcamDevices() {
    try {
        const all = await navigator.mediaDevices.enumerateDevices();
        return all
            .filter((d) => d.kind === "videoinput")
            .map((d) => ({ label: d.label || "", deviceId: d.deviceId || "" }));
    } catch (e) {
        console.warn("enumerateDevices failed:", e);
        return [];
    }
}

function findDeviceIdByLabel(devices, wantedLabel) {
    if (!wantedLabel) return null;
    for (const d of devices) {
        if (d.label === wantedLabel) return d.deviceId;
    }
    const loose = devices.find(
        (d) =>
            d.label &&
            d.label.toLowerCase().includes(wantedLabel.toLowerCase()),
    );
    return loose ? loose.deviceId : null;
}

function drawPaddedSquareFromVideo(video, ctx, size, fill = "#000") {
    ctx.canvas.width = size;
    ctx.canvas.height = size;
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const scl = Math.min(size / vw, size / vh);
    const dw = Math.round(vw * scl);
    const dh = Math.round(vh * scl);
    const ox = ((size - dw) / 2) | 0;
    const oy = ((size - dh) / 2) | 0;

    ctx.save();
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = fill;
    ctx.fillRect(0, 0, size, size);

    if (FLIP_HORIZONTAL) {
        ctx.translate(size, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, ox, oy, dw, dh);
    } else {
        ctx.drawImage(video, ox, oy, dw, dh);
    }
    ctx.restore();

    return ctx.getImageData(0, 0, size, size);
}

async function rafLoop() {
    if (!_videoEl || _videoEl.readyState < 2) {
        requestAnimationFrame(rafLoop);
        return;
    }
    if (rafProcessing) {
        requestAnimationFrame(rafLoop);
        return;
    }
    rafProcessing = true;

    try {
        const frame = lastVideoMediaTime;
        const img = drawPaddedSquareFromVideo(
            _videoEl,
            procCtx,
            INPUT_W,
            "#000",
        );
        // flipH false because we already flipped in drawPaddedSquareFromVideo
        const input = toInputTensorFromImageData(img, false);

        await runInferencePipeline(
            input,
            -1 /* frameId */,
            0 /* seq */,
            frame,
            _sender,
        );

        const now = performance.now();
        // FPS calculation preserved but not displayed
        if (prevT) {
            const fps = 1000 / (now - prevT);
            // setStatus(`FPS: ${fps.toFixed(2)} (webcam)`);
        }
        prevT = now;
    } catch (e) {
        console.error(e);
        setStatus(`Error (webcam): ${e?.message || e}`);
    } finally {
        rafProcessing = false;
        requestAnimationFrame(rafLoop);
    }
}

export async function startWebcam() {
    _videoEl = ensureVisibleVideo();

    const initialDevices = await listWebcamDevices();
    const exactId = findDeviceIdByLabel(initialDevices, WEBCAM_LABEL);

    const constraints = exactId
        ? {
              video: {
                  deviceId: { exact: exactId },
                  width: { ideal: window.innerWidth },
                  height: { ideal: window.innerHeight },
              },
          }
        : {
              video: {
                  width: { ideal: window.innerWidth },
                  height: { ideal: window.innerHeight },
              },
          };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        _videoEl.srcObject = stream;

        const vfc = (now, metadata) => {
            if (metadata && typeof metadata.mediaTime === "number") {
                if (_sender) {
                    _sender({ tick: metadata.mediaTime });
                }
                lastVideoMediaTime = metadata.mediaTime;
            }
            _videoEl.requestVideoFrameCallback(vfc);
        };
        _videoEl.requestVideoFrameCallback(vfc);

        _videoEl.onloadedmetadata = () => {
            setStatus("");
            requestAnimationFrame(rafLoop);
        };
    } catch (err) {
        console.error("getUserMedia failed:", err);
        setStatus(`Camera error: ${err?.message || err}`);
    }

    return initialDevices;
}
