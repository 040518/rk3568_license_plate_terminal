# 模型转换脚本

转换顺序：

1. `export_onnx_320.py`：将 YOLO 权重导出为静态 `320x320` ONNX。
2. `split_outputs.py`：把检测头改为框和置信度两个输出，便于 RKNN 后处理。
3. `convert_det320_int8.py`：使用 `dataset.txt` 做通道级 INT8 校准，生成 `best_320_int8_split.rknn`。
4. `convert_ocr_int8.py`：使用 `dataset_ocr.txt` 对 `48x320` OCR 模型做 INT8 校准。

脚本中的输入输出文件名是 Ubuntu 工程目录下的相对路径。运行前请准备 RKNN Toolkit、ONNX 文件和校准数据集；最终生成的 RKNN 文件已放在 `rk3568_inference` 目录中供直接部署。
