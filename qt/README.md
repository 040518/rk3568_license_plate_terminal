# Qt 最终界面

`003.pro`、`mainwindow.*` 和 `main.cpp` 是当前板端使用的最终版本。界面显示摄像头画面、FPS、目标数和最近一次有效 OCR 结果；没有识别到新车牌时会保留上一次车牌。

在 Ubuntu 交叉编译机执行：

```bash
cd /home/hhh/rk3568_qt
./build_qt.sh 003
```

程序启动时默认执行 `/root/rknn_yolo/license_plate_ocr_camera_320.py`，并使用 `--qt-output` 将 JPEG 帧和状态信息传给 Qt。
