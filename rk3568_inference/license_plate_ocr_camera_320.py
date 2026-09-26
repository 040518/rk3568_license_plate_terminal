"""基于 ATK-DLRK3568 的实时车牌检测与识别系统 —— 板端推理主程序。

================================================================================
一、项目概述
================================================================================
本项目是一个运行在正点原子 ATK-DLRK3568 开发板上的「实时车牌检测 + 文字识别」
嵌入式 AI 系统，完整链路为：

    MIPI/V4L2 摄像头
        │  采集原始帧（默认 1280x720 @ 30fps）
        ▼
    GStreamer v4l2src 管道 ──► 转 BGR、最新帧队列（丢旧帧防延迟）
        │
        ▼
    YOLO 车牌检测（RKNN NPU, INT8, 输入 320x320）
        │  输出：车牌候选框 (cx,cy,bw,bh) + 置信度
        │  └─ 后处理：letterbox 坐标反算 → NMS 去重 → 外扩 ROI
        ▼
    PP-OCR 文字识别（RKNN NPU, INT8, 输入 48x320, 每 5 帧抽帧一次）
        │  输出：CTC 序列 → 贪心解码 → 车牌字符串
        ▼
    结果输出
        ├─ Qt 界面：length-prefix JPEG + STATUS 状态行（--qt-output）
        └─ 本地显示：旋转/缩放后写 /dev/fb0 帧缓冲（默认）

================================================================================
二、项目目录结构
================================================================================
    rk3568_final_project/
    ├── rk3568_inference/              # 板端推理（部署到 /root/rknn_yolo/）
    │   ├── license_plate_ocr_camera_320.py   # 本文件：摄像头实时推理主程序
    │   ├── profile_pipeline_320.py           # 分阶段耗时剖析脚本（逐帧输出 CSV）
    │   ├── best_320_int8_split.rknn          # 车牌检测模型（YOLO, 320, INT8）
    │   ├── ppocrv5_mobile_rec_48x320_int8.rknn  # OCR 识别模型（48x320, INT8）
    │   └── ppocrv5_dict.txt                  # OCR 字符字典（下标 0 为空白符）
    ├── model_conversion/              # 模型转换（ONNX 导出 → 拆分 → INT8 量化）
    │   ├── export_onnx_320.py                # YOLO 权重 → 静态 320x320 ONNX
    │   ├── split_outputs.py                  # 检测头拆为框/置信度两个输出
    │   ├── convert_det320_int8.py            # 检测模型 INT8 量化
    │   └── convert_ocr_int8.py               # OCR 模型 INT8 量化
    └── qt/                            # Qt 5.12.9 界面源码 + ARM64 交叉编译脚本
        ├── 003.pro / mainwindow.* / main.cpp # 板端界面（显示画面/FPS/车牌结果）
        └── build_qt.sh                       # 交叉编译脚本

================================================================================
三、模型信息
================================================================================
    ┌──────────┬──────────────────────────────┬──────────────┬───────────────┐
    │ 模型     │ 文件                         │ 输入尺寸     │ 量化方式      │
    ├──────────┼──────────────────────────────┼──────────────┼───────────────┤
    │ 车牌检测 │ best_320_int8_split.rknn     │ 320x320x3    │ INT8          │
    │ OCR 识别 │ ppocrv5_mobile_rec_48x320…   │ 48x320x3     │ INT8          │
    └──────────┴──────────────────────────────┴──────────────┴───────────────┘
    - 检测输出两个张量：框 (N,4) 中心点宽高、置信度 (N,)。
    - OCR 输出 CTC 概率矩阵 (时间步, 字典长度)，经贪心解码得到文字。

================================================================================
四、本文件说明
================================================================================
本文件是板端实时推理主程序，默认摄像头节点为 /dev/video0（rkisp_mainpath）。
由于 rkisp_mainpath 是 V4L2 多平面设备，当前 BSP 的 OpenCV V4L2 后端无法打开，
因此采集改用 GStreamer v4l2src 管道（也为后续接入 DMABUF/RGA 留出升级空间）。

显示有两条路径：
  1. 默认写 /dev/fb0 帧缓冲（DSI 面板为 720x1280 竖向布局，需旋转+缩放）；
  2. 加 --qt-output 时改为向 stdout 输出「长度前缀 + JPEG」帧、向 stderr 输出
     STATUS/PROFILE 状态行，供 Qt 界面进程解析显示。

运行示例（Qt 启动时默认使用同一命令）：
    python3 /root/rknn_yolo/license_plate_ocr_camera_320.py \
      --camera /dev/video0 --width 1280 --height 720 --fps 15 \
      --no-display --qt-output
"""
import os, cv2, time, argparse, subprocess, sys, struct  # 系统、图像、计时、参数、子进程、标准流、二进制打包
import numpy as np                        # 数值计算：张量后处理、NMS
from rknnlite.api import RKNNLite        # RKNN NPU 推理运行时（RK3568 板端）

# ---- 路径与模型配置 ---------------------------------------------------------
ROOT = "/root/rknn_yolo"                          # 板端工作根目录
DET_MODEL = os.path.join(ROOT, "best_320_int8_split.rknn")        # 车牌检测模型（YOLO, 320, int8）
OCR_MODEL = os.path.join(ROOT, "ppocrv5_mobile_rec_48x320_int8.rknn")  # PP-OCR 识别模型（48x320, int8）
DICT_PATH = os.path.join(ROOT, "ppocrv5_dict.txt")                # OCR 字符字典文件

# ---- 检测超参数 -------------------------------------------------------------
DET_SIZE, DET_CONF, DET_IOU = 320, 0.35, 0.30     # 检测输入尺寸 / 置信度阈值 / NMS IoU 阈值

# ---- OCR 超参数 -------------------------------------------------------------
OCR_H, OCR_W, OCR_CONF = 48, 320, 0.25            # OCR 输入高/宽、平均置信度阈值
OCR_INTERVAL = 5                                  # 每隔多少帧做一次 OCR（抽帧识别）

# ---- 坐标微调/外扩（经验值，补偿模型偏差、给文字留边）-----------------------
DET_SHIFT_X, DET_SHIFT_Y = 4.0, 5.0               # 检测框中心点偏移补偿
OCR_PAD_X, OCR_PAD_Y = 0.10, 0.20                 # OCR 裁剪时在框四周外扩的比例

# 最近一次检测/OCR 的耗时记录（供周期性 PROFILE 日志输出用）
_last_det_timing = (0.0, 0.0, 0.0)  # CPU 前处理, 检测 NPU, CPU 后处理（毫秒）
_last_ocr_timing = (0.0, 0.0)       # CPU 裁剪/预处理, OCR NPU（毫秒）


def letterbox(img, size=320):
    """等比缩放并加灰边（letterbox），把任意比例图像填入 size x size 正方形。

    返回 (缩放加边后的图, 缩放比例 r, 左上角 padding 偏移 (l, t))。
    这是 YOLO 推理前标准预处理，保证输入不拉伸变形。
    """
    h, w = img.shape[:2]                          # 原图高、宽
    r = min(size / h, size / w)                   # 取较小缩放比，保证整图不超出目标尺寸
    nw, nh = int(round(w * r)), int(round(h * r)) # 缩放后实际宽高
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2    # 上下/左右各需补的灰边量
    l, t = int(round(dw - .1)), int(round(dh - .1))  # ±0.1 四舍五入，使补边尽量对称
    # 以灰色 (114,114,114) 填充边框，返回含缩放比和左上角偏移（用于坐标反算）
    out = cv2.copyMakeBorder(im, t, size - nh - t, l, size - nw - l,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (l, t)


def nms(boxes, scores, threshold):
    """非极大值抑制（NMS）：去掉重叠度(IoU)过高的重复框，保留置信度最高者。

    返回保留下来的框索引列表。
    """
    if not boxes:
        return []
    b = np.asarray(boxes, np.float32)             # 框坐标转浮点数组
    s = np.asarray(scores, np.float32)            # 置信度转浮点数组
    a = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])  # 每个框面积
    order = s.argsort()[::-1]                     # 按置信度从高到低排序
    keep = []                                     # 最终保留的索引
    while order.size:
        i = int(order[0])                         # 取当前置信度最高的框
        keep.append(i)
        if order.size == 1:
            break
        # 计算当前框与其余框的交集区域，得到 IoU
        xx1 = np.maximum(b[i, 0], b[order[1:], 0]); yy1 = np.maximum(b[i, 1], b[order[1:], 1])
        xx2 = np.minimum(b[i, 2], b[order[1:], 2]); yy2 = np.minimum(b[i, 3], b[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)          # 交集面积
        iou = inter / (a[i] + a[order[1:]] - inter + 1e-6)                   # IoU，加 1e-6 防除零
        order = order[np.where(iou <= threshold)[0] + 1]                     # 保留 IoU 不超阈值的框
    return keep


def detect(model, frame):
    """执行一帧车牌检测，并把前处理/NPU/后处理耗时写入全局 _last_det_timing。

    返回检测结果列表 [(框 [x1,y1,x2,y2], 置信度), ...]。
    """
    global _last_det_timing
    h, w = frame.shape[:2]                        # 原图尺寸，用于坐标映射与裁剪
    t0 = time.perf_counter()
    # 前处理：BGR→RGB，letterbox 到 320x320，扩展 batch 维
    inp, r, pad = letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), DET_SIZE)
    inp = inp[None].astype(np.uint8)              # 形状 (1,320,320,3)
    t1 = time.perf_counter()
    # NPU 推理，NHWC 数据布局
    out = model.inference(inputs=[inp], data_format=["nhwc"])
    t2 = time.perf_counter()
    _last_det_timing = (1000 * (t1 - t0), 1000 * (t2 - t1), 0.0)  # 先记前处理与 NPU 耗时
    if out is None:
        raise KeyboardInterrupt                   # 推理异常时走中断路径退出
    if len(out) < 2:
        return []                                 # 输出数量不足视为无检测
    # 后处理：解析框与置信度
    boxes = np.squeeze(np.asarray(out[0])).astype(np.float32)        # 输出 0: 框 (cx,cy,bw,bh)
    scores = np.squeeze(np.asarray(out[1])).reshape(-1).astype(np.float32)  # 输出 1: 置信度
    if boxes.ndim == 2 and boxes.shape[0] == 4:   # 形状 (4, N) 则转置为 (N, 4)
        boxes = boxes.T
    cand = []
    for d, s in zip(boxes, scores):
        if float(s) < DET_CONF:                   # 过滤低置信度框
            continue
        cx, cy, bw, bh = map(float, d[:4])        # 中心点 + 宽高
        # 中心点转左上/右下角，并加经验偏移补偿
        x1, x2 = cx - bw / 2 + DET_SHIFT_X, cx + bw / 2 + DET_SHIFT_X
        y1, y2 = cy - bh / 2 + DET_SHIFT_Y, cy + bh / 2 + DET_SHIFT_Y
        # letterbox 坐标反算回原图坐标系（减 padding 再除以缩放比）
        x1, x2 = (x1 - pad[0]) / r, (x2 - pad[0]) / r
        y1, y2 = (y1 - pad[1]) / r, (y2 - pad[1]) / r
        # 裁剪到原图范围内
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
        if x2 > x1 and y2 > y1:                   # 有效框才保留
            cand.append(([x1, y1, x2, y2], float(s)))
    # NMS 去重，并补记后处理耗时
    k = nms([x[0] for x in cand], [x[1] for x in cand], DET_IOU)
    _last_det_timing = (_last_det_timing[0], _last_det_timing[1],
                        1000 * (time.perf_counter() - t2))
    return [cand[i] for i in k]


def crop_input(frame, box):
    """按检测框裁剪车牌区域并做 OCR 预处理（外扩→缩放→填灰底）。

    返回 (输入张量, (x1,y1,x2,y2))；裁剪区为空时张量为 None。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1                      # 框宽、框高
    # 按比例向外扩展裁剪区域（给文字留边）
    x1 = max(0, int(x1 - bw * OCR_PAD_X)); x2 = min(w, int(x2 + bw * OCR_PAD_X))
    y1 = max(0, int(y1 - bh * OCR_PAD_Y)); y2 = min(h, int(y2 + bh * OCR_PAD_Y))
    c = frame[y1:y2, x1:x2]
    if c.size == 0:                                # 裁剪区为空
        return None, (x1, y1, x2, y2)
    # 转 RGB，等比缩放到高度 48、宽度不超过 320
    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    scale = OCR_H / max(1, c.shape[0])             # 按高度 48 计算缩放比
    nw = min(OCR_W, max(1, int(round(c.shape[1] * scale))))  # 缩放后宽度，上限 320
    c = cv2.resize(c, (nw, OCR_H))
    # 放到 127 灰底画布，右侧不足 320 部分保持灰底（PP-OCR 固定输入尺寸）
    canvas = np.full((OCR_H, OCR_W, 3), 127, np.uint8)
    canvas[:, :nw] = c
    return canvas[None], (x1, y1, x2, y2)


def recognize(model, inp, chars):
    """对裁剪好的车牌做 OCR 识别（NPU 推理 + CTC 贪心解码）。

    返回 (识别文字, 平均置信度)；无输入或置信度不足时文字为空串。
    """
    if inp is None:
        return "", 0.0
    global _last_ocr_timing
    t0 = time.perf_counter()
    out = model.inference(inputs=[inp], data_format=["nhwc"])  # OCR NPU 推理
    t1 = time.perf_counter()
    _last_ocr_timing = (_last_ocr_timing[0], 1000 * (t1 - t0))  # 记 OCR NPU 耗时
    if out is None:
        raise KeyboardInterrupt
    # CTC 解码：去掉 batch 维得到 (时间步, 字符数)
    p = np.asarray(out[0])
    p = p[0] if p.ndim == 3 else p
    if p.ndim != 2:
        return "", 0.0
    ids = np.argmax(p, 1)                          # 每个时间步概率最大的字符 id
    conf = np.max(p, 1)                            # 对应最大概率值
    text = []; vals = []; last = -1                # last 记录上一有效字符 id（用于去重）
    for i, c in zip(ids.tolist(), conf.tolist()):
        if i == 0 or i == last:                    # 跳过空白符和连续重复字符
            last = i
            continue
        last = i
        if i < len(chars):                         # id 在字典范围内才映射成字符
            text.append(chars[i]); vals.append(float(c))
    mean = float(np.mean(vals)) if vals else 0.0   # 有效字符平均置信度
    return ("".join(text).strip() if mean >= OCR_CONF else ""), mean


def display_process(w, h, fps):
    """启动 GStreamer 显示进程，从 stdin 读 BGR 帧经 kmssink 上屏。

    直接使用 caps；当前 BSP 未提供 rawvideoparse。
    """
    cmd = ["gst-launch-1.0", "-q", "fdsrc", "fd=0", "!",
           f"video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1", "!",
           "videoconvert", "!", "kmssink", "sync=false"]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    # ---- 命令行参数 ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="/dev/video0")          # 摄像头设备节点
    ap.add_argument("--width", type=int, default=1280)          # 采集分辨率宽
    ap.add_argument("--height", type=int, default=720)          # 采集分辨率高
    ap.add_argument("--fps", type=int, default=30)              # 采集帧率
    ap.add_argument("--no-display", action="store_true")        # 关闭显示输出
    ap.add_argument("--qt-output", action="store_true",
                    help="send length-prefixed JPEG to stdout and STATUS to stderr")
    args = ap.parse_args()

    # 校验模型与字典文件是否存在
    for p in (DET_MODEL, OCR_MODEL, DICT_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # 读取 OCR 字符字典；下标 0 为空白符(CTC blank)，前面补一个空串
    chars = [""] + [x.rstrip("\r\n") for x in open(DICT_PATH, encoding="utf-8")]

    # 加载并初始化检测 / OCR 两个 RKNN 模型
    det, ocr = RKNNLite(), RKNNLite()
    assert det.load_rknn(DET_MODEL) == 0 and ocr.load_rknn(OCR_MODEL) == 0  # 加载模型
    assert det.init_runtime() == 0 and ocr.init_runtime() == 0              # 初始化 NPU 运行时

    # rkisp_mainpath 是 V4L2 多平面设备，当前 BSP 的 OpenCV V4L2 后端无法打开。
    # 使用 GStreamer v4l2src，通过管道传输 BGR 帧，也为后续接入 DMABUF/RGA 留出升级空间。
    w, h = args.width, args.height
    # 采集管道：v4l2src 取 NV12 帧 -> 队列(容量1，允许丢旧帧) -> 转 BGR -> 输出到 stdout
    gst = ["gst-launch-1.0", "-q", "v4l2src", f"device={args.camera}", "io-mode=mmap", "!",
           f"video/x-raw,format=NV12,width={w},height={h},framerate={args.fps}/1", "!",
           "queue", "max-size-buffers=1", "max-size-bytes=0", "max-size-time=0", "leaky=downstream", "!",
           "videoconvert", "!",
           f"video/x-raw,format=BGR,width={w},height={h}", "!", "fdsink", "fd=1"]
    cap = subprocess.Popen(gst, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    # Rockchip 的 v4l2 插件会在第一个缓冲区之前向 fd 1 输出简短提示信息
    #（即使使用 gst-launch -q 也会输出），需要从原始数据流中剥离该提示。
    cap.stdout.read(60)
    frame_bytes = w * h * 3                        # 每帧 BGR 原始字节数
    print(f"camera={args.camera} actual={w}x{h} backend=gstreamer")

    # 开发板将 DSI 面板作为 32 位帧缓冲区提供。直接写入可以避免当前 BSP
    # 中不稳定的 fdsrc/kmssink 原始管道路径。
    fb = None if args.no_display else open("/dev/fb0", "wb", buffering=0)

    frame_id = 0; cached = []; last = []; t0 = time.perf_counter()
    try:
        while True:
            # ---- 从 GStreamer 管道读满一帧原始 BGR 数据 ----
            read_t0 = time.perf_counter()
            chunks = []; remaining = frame_bytes
            while remaining:
                part = cap.stdout.read(remaining)  # 分块读取直到凑满一帧
                if not part:
                    break
                chunks.append(part); remaining -= len(part)
            raw = b"".join(chunks)
            read_ms = 1000 * (time.perf_counter() - read_t0)
            if len(raw) != frame_bytes:            # 数据不完整视为采集失败
                print(f"camera read failed bytes={len(raw)} returncode={cap.poll()}")
                break
            # 原始字节 -> (h, w, 3) BGR 图像（copy 一份避免共享缓冲区）
            frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3)).copy()

            # ---- 车牌检测 ----
            frame_id += 1
            t_loop = time.perf_counter()
            last = detect(det, frame)              # 检测并记录各阶段耗时
            det_ms = 1000 * (time.perf_counter() - t_loop)   # 检测总耗时

            # ---- 逐框做 OCR 与标注 ----
            new = []
            for box, dc in last:
                # 裁剪 + OCR 预处理
                ct = time.perf_counter()
                inp, draw = crop_input(frame, box)
                crop_ms = 1000 * (time.perf_counter() - ct)
                # 按 OCR_INTERVAL 抽帧识别，否则跳过（省算力）
                text, oc = recognize(ocr, inp, chars) if frame_id % OCR_INTERVAL == 0 else ("", 0.0)
                # 画检测框与文字（未识别到则显示 OCR?）
                x1, y1, x2, y2 = map(int, draw)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, (text or "OCR?") + f" {dc:.2f}",
                            (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
                new.append((box, text, oc))
            cached = new

            # ---- 可选：向 Qt 端输出 JPEG 帧与状态信息 ----
            if args.qt_output:
                # JPEG 编码（质量 70）
                jt = time.perf_counter()
                ok_jpg, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg_ms = 1000 * (time.perf_counter() - jt)
                if ok_jpg:
                    # 以「4 字节小端长度前缀 + JPEG 数据」的形式写到 stdout
                    data = enc.tobytes()
                    sys.stdout.buffer.write(struct.pack('<I', len(data)))
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                # 向 stderr 输出状态行：FPS / 检测数 / 识别出的车牌串
                plates = [t for _, t, _ in new if t]
                cur_fps = frame_id / max(time.perf_counter() - t0, 1e-6)
                print('STATUS FPS=%.2f COUNT=%d PLATE=%s' % (cur_fps, len(last), '|'.join(plates)),
                      file=sys.stderr, flush=True)

            # ---- 周期性打印各阶段耗时 ----
            if frame_id % 30 == 0:
                dp, dn, do = _last_det_timing
                op, on = _last_ocr_timing
                print('PROFILE read=%.2f det_cpu_pre=%.2f det_npu=%.2f det_cpu_post=%.2f '
                      'ocr_cpu=%.2f ocr_npu=%.2f jpeg=%.2f det_total=%.2f loop_fps=%.2f' % (
                          read_ms, dp, dn, do, crop_ms if last else 0.0, on,
                          locals().get('jpeg_ms', 0.0), det_ms,
                          frame_id / max(time.perf_counter() - t0, 1e-6)),
                      file=sys.stderr, flush=True)

            # ---- 上屏显示：横屏转竖屏并写入 framebuffer ----
            if fb:
    # 摄像头画面为横向，面板为 720x1280 的竖向布局。
                panel = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)          # 顺时针旋转 90°
                panel = cv2.resize(panel, (720, 1280), interpolation=cv2.INTER_LINEAR)  # 缩放到面板尺寸
                bgra = cv2.cvtColor(panel, cv2.COLOR_BGR2BGRA)              # 补 Alpha 通道为 32 位
                try:
                    fb.seek(0)                                               # 从帧缓冲起始写入
                    fb.write(bgra.tobytes())
                except OSError:
                    fb = None                                                # 写入失败则关闭显示

            # 每 30 帧打印一次简要状态
            if frame_id % 30 == 0:
                print(f"frame={frame_id} fps={frame_id / max(time.perf_counter() - t0, 1e-6):.2f} det={len(last)}")
    except KeyboardInterrupt:
        pass                                     # 收到中断信号则正常退出
    finally:
        # 清理：结束采集进程、释放模型与显示资源
        cap.terminate(); cap.wait(timeout=3)
        det.release(); ocr.release()
        if fb:
            fb.close()


if __name__ == "__main__":
    main()
