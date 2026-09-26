# RK3568 推理程序

主程序是 `license_plate_ocr_camera_320.py`。它使用 RK3568 NPU INT8 车牌检测模型和 INT8 OCR 模型，支持 MIPI/V4L2 摄像头、GStreamer 最新帧队列、Qt JPEG 输出和分段耗时统计。

`profile_pipeline_320.py` 用于单独采集 CPU 预处理、NPU 检测、CPU 后处理和 OCR NPU 的耗时。模型输入尺寸固定为检测 `320x320`、OCR `48x320`；摄像头采集尺寸可以是 `1280x720`，不会改变模型输入尺寸。
