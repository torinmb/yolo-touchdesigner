import argparse
from ultralytics import YOLO

# ---------- Parse command-line arguments ----------
parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
parser.add_argument(
    "--model", 
    type=str, 
    required=True, 
    help="Path to the YOLO .pt model file (e.g. facen.pt, yolo11s.pt)"
)
parser.add_argument(
    "--imgsz", 
    type=int, 
    default=640, 
    help="Image size for export (default: 640)"
)
parser.add_argument(
    "--device", 
    type=str, 
    default="mps", 
    help="Device for export (e.g. 'cpu', 'cuda:0', 'mps')"
)
args = parser.parse_args()

# ---------- Load model ----------
model = YOLO(args.model)

# ---------- Export ----------
onnx_path = model.export(
    format="onnx",
    imgsz=args.imgsz,
    dynamic=True,
    half=False,     # Set to True if using CUDA for FP16 export
    simplify=True,
    opset=14,
    device=args.device
)

print("Exported:", onnx_path)
