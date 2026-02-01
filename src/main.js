// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

import "./style.css";
import { WS_PORT, USE_BINARY } from "./config.js";
import { setStatus } from "./ui.js";
import { initSessions } from "./inference/onnx.js";
import {
    handleBinaryMessage,
    setWebSocketSender as setBinarySender,
} from "./modes/binary.js";
import {
    startWebcam,
    listWebcamDevices,
    setWebSocketSender as setWebcamSender,
} from "./modes/webcam.js";

(async function main() {
    setStatus("Loading…");

    const baseURL = new URL(".", location.href);
    await initSessions(baseURL);

    // WebSocket Setup
    const ws = new WebSocket(`ws://localhost:${WS_PORT}`);
    ws.binaryType = "arraybuffer";

    const sender = (msg) => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(msg));
        }
    };

    setBinarySender(sender);
    setWebcamSender(sender);

    ws.onopen = async () => {
        console.log("WebSocket connected");
        ws.send(JSON.stringify({ loaded: true }));

        const devices = await listWebcamDevices();
        ws.send(JSON.stringify({ webcamDevices: devices.map((d) => d.label) }));

        setStatus(USE_BINARY ? "Ready (binary)" : "Ready (webcam)");
    };

    ws.onerror = () => {
        setStatus(
            `Error: WebSocket connection to 'ws://localhost:${WS_PORT}/' failed.`,
        );
    };

    ws.onclose = (event) => {
        setStatus(
            `🔌 Connection closed (code ${event.code}) on port ${WS_PORT}`,
        );
    };

    if (USE_BINARY) {
        ws.onmessage = (ev) => {
            const data = ev.data;
            // Ignore text, process binary
            if (typeof data === "string") return;
            if (data instanceof ArrayBuffer) {
                handleBinaryMessage(data);
            }
        };
    } else {
        // Webcam Mode
        await startWebcam();
    }
})();
