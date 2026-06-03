// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import {
    FLIP_HORIZONTAL,
    FLIP_VERTICAL,
    qs,
    WEBCAM_INPUT_H,
    WEBCAM_INPUT_W,
    WEBCAM_LABEL,
    WEBCAM_ROTATION_DEG,
} from "../config.js";
import { detSession, poseSession, segSession } from "../inference/onnx.js";
import {
    toInputTensorFromVideo,
    toInputTensorFromBitmap,
} from "../inference/io.js";
import { runInferencePipeline } from "../pipeline.js";
import { setStatus, ensureVisibleVideo } from "../ui.js";
import { getRotatedSize } from "../utils/orientation.js";

const isCEF =
    typeof navigator !== "undefined" &&
    (navigator.userAgent.indexOf("TouchDesigner") !== -1 ||
        navigator.userAgent.indexOf("CEF") !== -1);

let rafProcessing = false;
let prevT = 0;
let lastVideoMediaTime = 0;
let _sender = null;
let _videoEl = null;
let _imageCapture = null;

const webcamTransform = {
    flipH: FLIP_HORIZONTAL,
    flipV: FLIP_VERTICAL,
    rotationDeg: WEBCAM_ROTATION_DEG,
};

function getWebcamProcessingSize() {
    return getRotatedSize(WEBCAM_INPUT_W, WEBCAM_INPUT_H, WEBCAM_ROTATION_DEG);
}

function getWebcamCaptureConstraints(exactId) {
    const widthWasExplicit = qs.has("webcamInputW") || qs.has("inputW");
    const heightWasExplicit = qs.has("webcamInputH") || qs.has("inputH");
    const defaultCaptureSize = getRotatedSize(
        window.innerWidth,
        window.innerHeight,
        WEBCAM_ROTATION_DEG,
    );

    const video = exactId ? { deviceId: { exact: exactId } } : {};

    video.width = {
        ideal: widthWasExplicit ? WEBCAM_INPUT_W : defaultCaptureSize.width,
    };
    video.height = {
        ideal: heightWasExplicit ? WEBCAM_INPUT_H : defaultCaptureSize.height,
    };

    if (widthWasExplicit && heightWasExplicit && WEBCAM_INPUT_H > 0) {
        video.aspectRatio = { ideal: WEBCAM_INPUT_W / WEBCAM_INPUT_H };
    } else if (defaultCaptureSize.height > 0) {
        video.aspectRatio = {
            ideal: defaultCaptureSize.width / defaultCaptureSize.height,
        };
    }

    return { video };
}

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
        const frameSize = getWebcamProcessingSize();

        if (isCEF && _imageCapture) {
            // CEF: grab frame directly from MediaStreamTrack, bypassing the broken video compositor
            try {
                const bitmap = await _imageCapture.grabFrame();
                input = toInputTensorFromBitmap(
                    bitmap,
                    WEBCAM_INPUT_W,
                    WEBCAM_INPUT_H,
                    webcamTransform,
                );
                bitmap.close();
            } catch (grabErr) {
                setStatus("ImageCapture.grabFrame failed: " + grabErr.message);
                rafProcessing = false;
                requestAnimationFrame(rafLoop);
                return;
            }
        } else {
            input = toInputTensorFromVideo(
                _videoEl,
                WEBCAM_INPUT_W,
                WEBCAM_INPUT_H,
                webcamTransform,
            );
        }

        await runInferencePipeline(
            input,
            -1 /* frameId */,
            0 /* seq */,
            frame,
            _sender,
            frameSize,
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

    const constraints = getWebcamCaptureConstraints(exactId);

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
