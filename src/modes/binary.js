// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import { INPUT_W, INPUT_H } from "../config.js";
import { detSession, poseSession } from "../inference/onnx.js";
import { toInputTensorFromU8CHW } from "../inference/io.js";
import { runInferencePipeline } from "../pipeline.js";
import { setStatus } from "../ui.js";

let latestJob = null;
let isProcessing = false;

function parseHeader(buf) {
    if (buf.byteLength < 16) return null;
    const dv = new DataView(buf);
    const type = dv.getUint8(0);
    const dtype = dv.getUint8(1);
    const layout = dv.getUint8(2);
    const H = dv.getUint16(4, true);
    const W = dv.getUint16(6, true);
    const seq = dv.getUint32(8, true);
    const td = dv.getUint32(12, true);
    if (type !== 10 || dtype !== 1 || layout !== 1) return null; // require u8 CHW
    // payload is a view
    const payload = new Uint8Array(buf, 16);
    return { H, W, seq, td, payload };
}

export function handleBinaryMessage(data) {
    if (data.byteLength < 16) return;

    // Only support Legacy Header U8 CHW
    const job = parseHeader(data);
    if (job && job.H === INPUT_H && job.W === INPUT_W) {
        latestJob = job;
        pumpBinary();
    }
}

export function setWebSocketSender(senderFn) {
    _sender = senderFn;
}

let _sender = null;

async function pumpBinary() {
    if (isProcessing) return;
    isProcessing = true;

    try {
        const job = latestJob;
        latestJob = null;

        if (!job || (!detSession && !poseSession)) return;

        const input = toInputTensorFromU8CHW(job.payload, INPUT_H, INPUT_W);

        await runInferencePipeline(
            input,
            job.td,
            job.seq,
            0 /* videoFrame */,
            _sender,
        );
    } catch (e) {
        console.error(e);
        setStatus(`Error (binary): ${e?.message || e}`);
    } finally {
        isProcessing = false;
        if (latestJob) queueMicrotask(pumpBinary);
    }
}
