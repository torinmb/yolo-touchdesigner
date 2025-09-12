# YOLO to ONNX Export Script

This script exports a YOLO `.pt` model to ONNX format using the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) library.

---

## Setup

### 1. Create a virtual environment

#### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\activate
```

---

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

Run the script with:

```bash
python export_yolo.py --model <path_to_model.pt> [--imgsz 640] [--device cpu|cuda:0|mps]
```

### Arguments

-   `--model` _(required)_  
    Path to the YOLO `.pt` model file (e.g. `yolo11s.pt`, `facen.pt`).

-   `--device` _(optional, default depends on system)_  
    Device to use for export:
    -   `cpu` → Works everywhere.
    -   `cuda:0` → NVIDIA GPU (Windows/Linux, if CUDA is available).
    -   `mps` → Apple Silicon (Mac).

---

## Examples

**Mac (Apple Silicon / MPS):**

```bash
python export_yolo.py --model yolo11s.pt --device mps
```

**Windows with CUDA GPU:**

```powershell
python export_yolo.py --model yolo11s.pt --device cuda:0
```

**Windows without CUDA (CPU fallback):**

```powershell
python export_yolo.py --model yolo11s.pt --device cpu
```

---

## Output

After export, the script prints the path to the generated `.onnx` file:

```
Exported: yolo11s.onnx
```
