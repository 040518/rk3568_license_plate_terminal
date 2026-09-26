# RK3568 车牌检测与 OCR 最终交付包

这个目录只保留当前验证可用的版本，按用途分为三个子目录：

- `qt`：RK3568 屏幕上的 Qt 界面源码和 ARM64 编译脚本。
- `rk3568_inference`：摄像头采集、车牌检测、OCR 和性能分析脚本，以及最终 RKNN 模型。
- `model_conversion`：从 ONNX 导出、拆分输出和 INT8 量化的最终脚本。

## 部署到开发板

将 `rk3568_inference` 中的文件复制到 `/root/rknn_yolo/`。程序默认使用：

```text
/root/rknn_yolo/best_320_int8_split.rknn
/root/rknn_yolo/ppocrv5_mobile_rec_48x320_int8.rknn
/root/rknn_yolo/ppocrv5_dict.txt
```

启动摄像头推理（Qt 启动时会自动使用同一命令）：

```bash
python3 /root/rknn_yolo/license_plate_ocr_camera_320.py \
  --camera /dev/video0 --width 1280 --height 720 --fps 15 \
  --no-display --qt-output
```

程序内部把摄像头帧缩放/letterbox 到检测模型的 `320x320`，每 5 帧对检测到的 ROI 做一次 `48x320` OCR；GStreamer 队列只保留最新帧，避免画面累积延迟。

## Qt 编译和运行

在 Ubuntu 交叉编译机上：

```bash
cp -a qt /home/hhh/rk3568_qt/003
cp qt/build_qt.sh /home/hhh/rk3568_qt/build_qt.sh
cd /home/hhh/rk3568_qt
chmod +x build_qt.sh
./build_qt.sh 003
```

生成的 `003` 推送到板端后运行。Qt 会在本地启动 Python 推理进程；LED 按钮会自动加载 `/lib/modules/5.10.160/leddriver.ko`，并控制 `/dev/dtsplatled`。

## 量化转换

`model_conversion` 中的脚本对应最终模型链路：先用 `export_onnx_320.py` 导出静态 `320x320` ONNX，再用 `split_outputs.py` 拆分检测头，最后分别运行检测和 OCR 的 INT8 转换脚本。转换需要 RKNN Toolkit、相应 ONNX 文件和校准数据集（`dataset.txt`、`dataset_ocr.txt`），这些大文件没有复制进交付包。
