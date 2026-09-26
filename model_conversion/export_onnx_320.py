from ultralytics import YOLO

# best_compat.pt 是 best.pt 的字节兼容副本，旧版 NumPy
# 检查点命名空间已针对当前 Python 3.8 环境完成规范化。
model = YOLO('/home/hhh/rk3568_yolo/best_compat.pt')
out = model.export(format='onnx', imgsz=320, batch=1, dynamic=False,
                   simplify=False, opset=12, device='cpu')
print(out)
