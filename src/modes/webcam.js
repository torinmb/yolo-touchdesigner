// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import { INPUT_W, INPUT_H, FLIP_HORIZONTAL, WEBCAM_LABEL } from "../config.js";
import { detSession, poseSession } from "../inference/onnx.js";
import { toInputTensorFromVideo, toInputTensorFromBitmap } from "../inference/io.js";
import { runInferencePipeline } from "../pipeline.js";
import { setStatus, ensureVisibleVideo } from "../ui.js";

const isCEF = typeof navigator !== "undefined" &&
    (navigator.userAgent.indexOf("TouchDesigner") !== -1 || navigator.userAgent.indexOf("CEF") !== -1);

let rafProcessing = false;
let prevT = 0;
let lastVideoMediaTime = 0;
let _sender = null;
let _videoEl = null;
let _imageCapture = null;

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
        let input;

        if (isCEF && _imageCapture) {
            // CEF: grab frame directly from MediaStreamTrack, bypassing the broken video compositor
            try {
                const bitmap = await _imageCapture.grabFrame();
                input = toInputTensorFromBitmap(bitmap, INPUT_W, INPUT_H, FLIP_HORIZONTAL);
                bitmap.close();
            } catch (grabErr) {
                setStatus("ImageCapture.grabFrame failed: " + grabErr.message);
                rafProcessing = false;
                requestAnimationFrame(rafLoop);
                return;
            }
        } else {
            input = toInputTensorFromVideo(_videoEl, INPUT_W, INPUT_H, FLIP_HORIZONTAL);
        }

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

        // Set up ImageCapture for CEF where video pixel readback is broken
        if (isCEF && typeof ImageCapture !== "undefined") {
            const track = stream.getVideoTracks()[0];
            _imageCapture = new ImageCapture(track);
            setStatus("CEF: ImageCapture ready");
        } else if (isCEF) {
            setStatus("CEF: ImageCapture API not available");
        }

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
