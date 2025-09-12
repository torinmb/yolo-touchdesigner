# Yolo TouchDesigner Plugin

Welcome! This project brings [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) into [TouchDesigner](https://derivative.ca/) with **no extra install required**.

### Features

-   **Object Tracking** – unique IDs that persist across frames
-   **Pose Estimation** – real-time multi-person body keypoints
-   **VizDrone Support** – trained for aerial and security footage
-   **Drop-in Component** – packaged as a `.tox` file you can load directly in TouchDesigner
-   **Custom Models** - Bring your own custom Yolov11 Models in by exporting them with ONNX

### Why WebSockets?

Using WebSockets creates a **clean boundary** between the open-sourced YOLO code and your TouchDesigner work.  
That means you can use this plugin in commercial projects without needing to open-source your entire codebase (see LICENSE for details).

### Get Started

1. Download the `.tox` file from [Patreon](https://www.patreon.com/c/blankensmithing).
2. Place the .tox file next you project .toe file so TouchDesigner can reference it locally. (NOTE if you don't do this saving your project will take an eturnity)
3. Drop it into your TouchDesigner project.

### Using this Plugin in Your Projects

This plugin bundles an open-source YOLO model (licensed under AGPL-3.0) together with my own TouchDesigner work. To stay compliant, I’ve released the **web app + networking code** that powers the model under AGPL-3.0 — the source is freely available.

**What this means for you:**

-   ✅ You can use the plugin in your **commercial or personal TouchDesigner projects** without needing to open-source your whole project.
-   ✅ You can forward or reuse the detection data (IDs, positions, etc.) however you like — it’s just data, not covered by the license.
-   ⚠️ The only time you’d need to share code is if you **modify the open-sourced parts inside the plugin** (the web app or WebSocket server). In that case, AGPL requires you to publish those modifications.

The plugin communicates with TouchDesigner over **WebSockets**, which creates a clear boundary between the AGPL-licensed code (the web component) and your project code. This separation means your TouchDesigner networks remain your own and can stay closed-source.

In short: **as long as you don’t modify the plugin’s source, you’re free to use it in commercial projects without open-sourcing your codebase.**

That said — if you’d like to, I’d highly encourage you to open-source your work to support the broader open-source community.
