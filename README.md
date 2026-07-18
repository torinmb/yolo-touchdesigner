# Yolo26 TouchDesigner Plugin

![YoloTN3](https://github.com/user-attachments/assets/2f1e7e71-cca4-4ef6-a4ab-efabedf99d07)

Welcome! This project brings [Ultralytics YOLO26 and YOLO11](https://github.com/ultralytics/ultralytics) into [TouchDesigner](https://derivative.ca/) with **no extra install required**.

### Features

- **YOLO26 Object Detection** – fast general object detection with support for newer YOLO26 model variants
- **YOLO26 Multi Person Pose Tracking** – supports both YOLO26 Pose and YOLO11 Pose for multi-person body keypoints, joints, and persistent IDs
- **YOLO26 OBB + VisDrone** – oriented bounding boxes plus aerial/security-focused detection for rotated objects and drone footage
- **YOLO26 Segmentation** – YOLO26 segmentation support for mask-based output workflows
- **YOLO11 Face + Small Models** – lightweight YOLO11 face tracking and YOLO11 small model support for faster deployments
- **Backwards Compatible** – detection and pose workflows remain compatible with existing YOLO11 models; segmentation is the new addition
- **Drop-in Component** – packaged as a `.tox` file you can load directly in TouchDesigner
- **Custom Models** – bring your own compatible YOLO11 and YOLO26 ONNX exports
- **Webcam + TOP Support** – use any webcam, or connect any TOP to start processing

### Get Started

For the easiest setup, download the current toolkit from Patreon. It includes the ready-to-use plugin, visualization and control helpers, example networks, and setup resources.

The core `yolo.tox` and its corresponding source code are also available in this repository under AGPL-3.0. The open-source version provides the underlying YOLO integration and raw output, but does not include all of the optional visualization helpers and examples available through Patreon.

1. Download the toolkit from [Patreon](https://www.patreon.com/posts/yolo-plugin-pose-139729511) or the open-source `.tox` from this repository.
2. Place the .tox file next you project .toe file so TouchDesigner can reference it locally. ⚠️NOTE if you don't do this saving your project will take an eternity.
3. Drop it into your TouchDesigner project. Then drop in any of the additional visualization helpers.

## Using This Plugin in Your Projects

This plugin uses Ultralytics YOLO, which is generally AGPL-3.0 unless you have an Enterprise License.

You can use the detection output as data in your TouchDesigner projects, but closed-source commercial use is not automatically cleared.

If you modify the YOLO runtime or use a custom Ultralytics ONNX model, you may need to open-source those changes and/or model files under AGPL-3.0, or obtain an Enterprise License from Ultralytics.

What this means for you:

✅ The plugin exposes its detection results as data for use in your TouchDesigner projects.

✅ The plugin is designed so TouchDesigner communicates with a separate local YOLO runtime over WebSockets. Your TouchDesigner network sends image data to the local server and receives detection results back.

✅ The core `yolo.tox` and its corresponding YOLO-related source are provided in this repository under AGPL-3.0. This includes the editable TouchDesigner component, web app, WebSocket server, model-running code, and included ONNX model files.

✅ Additional visualization and control helpers are available on Patreon. These are separate TouchDesigner components that consume the plugin's output data, are not required to run the open-source YOLO component, and some can also be used with other tracking systems such as RTMO.

✅ You can use, study, modify, and redistribute the open-source YOLO-related parts of this plugin, as long as you follow the AGPL-3.0 license.

⚠️ Closed-source commercial use is not automatically cleared. If you are using this plugin in a proprietary commercial project, paid deployment, client installation, or closed-source product, you should review the AGPL-3.0 requirements carefully or obtain an Enterprise License from Ultralytics.

⚠️ If you bring in your own custom Ultralytics YOLO ONNX model, you should assume that AGPL-3.0 still applies to that model. Exporting an Ultralytics model to ONNX does not remove the license requirements. If you use that model under AGPL-3.0, you may need to open-source the model, the training/export changes, and any model-related modifications under AGPL-3.0 as well.

⚠️ If you modify the YOLO runtime, web app, WebSocket server, model-running code, included ONNX models, or other AGPL-3.0 licensed parts of the plugin, you may need to open-source those changes under AGPL-3.0.

⚠️ If you redistribute this plugin, bundle it into a larger proprietary product, or use it as a core part of a closed commercial deployment, additional AGPL-3.0 obligations may apply.

This plugin is structured to separate the open-source YOLO runtime from your TouchDesigner network, and the YOLO-related source is provided for AGPL-3.0 compliance. However, license compliance depends on how you use, modify, distribute, bundle, train, export, or deploy the software and model files.

If your project needs to remain closed-source, you should review the AGPL-3.0 requirements carefully, consult a qualified attorney if needed, or obtain an Enterprise License from Ultralytics.

That said, I’d highly encourage you to open-source your work when possible to support the broader open-source community.

This is not legal advice.

In the plugin UI, links to this source code can be found under:

`About → Help / View Source Code`

### Dev and Build Instructions

`npm i`

`npm run dev`

`npm run build`
