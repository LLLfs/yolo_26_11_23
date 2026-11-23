from ultralytics import YOLO

# 加载你的模型（替换成你的.pt/.pth模型路径）
model = YOLO(r"best.pt")

# 导出ONNX格式（修复符号错误）
model.export(format="onnx", device=0, simplify=True, imgsz=640, opset=12)