# Yolo TouchDesigner Plugin

![YoloTN3](https://github.com/user-attachments/assets/2f1e7e71-cca4-4ef6-a4ab-efabedf99d07)

Welcome! This project brings [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) into [TouchDesigner](https://derivative.ca/) with **no extra install required**.

### Features

- **Object Tracking** – unique IDs that persist across frames
- **Multi-Person Pose Tracking** – body keypoints, joints, & persistent person IDs across frames
- **VisDrone Support** – trained for aerial and security footage
- **Face-Tracking Nano** – unique face IDs tracked across frames
- **Drop-in Component** – packaged as a `.tox` file you can load directly in TouchDesigner
- **Custom Models** – Bring your own custom Yolov11 Models in by exporting them with ONNX
- **Webcam + TOP Support** – Use any webcam, or connect any TOP to start processing

### Get Started

1. Download the `.tox` file from [Patreon](https://www.patreon.com/posts/yolo-plugin-pose-139729511).
2. Place the .tox file next you project .toe file so TouchDesigner can reference it locally. ⚠️NOTE if you don't do this saving your project will take an eternity.
3. Drop it into your TouchDesigner project.

## Using This Plugin in Your Projects

TLDR: This plugin is designed so your TouchDesigner project communicates with a separate AGPL-licensed YOLO component over WebSockets. My intent is that your TouchDesigner networks remain your own work and do not need to be open-sourced just because they receive detection data from the plugin.

This package includes an open-source YOLO model and model-running code licensed under the GNU Affero General Public License v3.0. To support AGPL compliance, the web app, WebSocket server, model-running code, and related source needed to run and modify those parts are released under AGPL-3.0.

What this means in practice:

✅ You can use the detection output, such as IDs, labels, positions, bounding boxes, and confidence values, in personal or commercial TouchDesigner projects.

✅ Your TouchDesigner project is designed to communicate with the AGPL component over WebSockets, using detection data as the interface between them.

✅ You may use, study, modify, and redistribute the AGPL-licensed parts under the terms of the AGPL-3.0 license.

⚠️ If you modify the AGPL-licensed parts, distribute a modified version, or run a modified version as a network service for others, AGPL-3.0 may require you to provide the corresponding source code for those modifications.

⚠️ If you combine this plugin with other code in a way that creates a single derivative work of the AGPL-licensed component, additional AGPL obligations may apply.

⚠️ This project includes third-party software and/or model files from Ultralytics YOLO under AGPL-3.0. For proprietary deployments that avoid AGPL obligations, Ultralytics offers a separate Enterprise License.

This plugin is structured to keep the AGPL YOLO component separate from your TouchDesigner networks, but license compliance depends on how you use, modify, distribute, or deploy the software. If you are building a commercial product around this, you should review the AGPL-3.0 license and consult a qualified attorney if needed.

That said, I’d highly encourage you to open-source your work to support the broader open-source community.

In the plugin UI, links to this source code can be found under:

`About → Help / View Source Code`

### Dev and Build Instructions

`npm i`

`npm run dev`

`npm run build`
