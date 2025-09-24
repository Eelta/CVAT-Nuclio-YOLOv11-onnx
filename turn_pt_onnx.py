from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(
    format="onnx",
    opset=12,
    simplify=True,
    imgsz=640,
    dynamic=False
)