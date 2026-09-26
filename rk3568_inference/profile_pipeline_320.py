#!/usr/bin/env python3
"""Profile the RK3568 320x320 plate + OCR pipeline by stage.

逐阶段剖析 RK3568 上「320x320 车牌检测 + PP-OCR 文字识别」流水线的耗时。
默认不写输出视频，这样测量的各个阶段耗时不会被视频编码器掩盖。
使用 --write-video 可额外计入视频写入成本。
"""
import argparse, csv, os, time      # 命令行参数、CSV 输出、系统、计时
import cv2                           # OpenCV：视频读取、图像预处理、画框
import numpy as np                   # 数值计算：张量后处理、NMS、统计
from rknnlite.api import RKNNLite   # RKNN NPU 推理运行时（RK3568 板端）

# ---- 路径与模型配置 ---------------------------------------------------------
ROOT = "/root/rknn_yolo"                          # 板端工作根目录
DET_MODEL = ROOT + "/best_320_int8_split.rknn"    # 车牌检测模型（YOLO, 320 输入, int8 量化）
OCR_MODEL = ROOT + "/ppocrv5_mobile_rec_48x320_int8.rknn"  # PP-OCR 识别模型（48x320, int8）
DICT_PATH = ROOT + "/ppocrv5_dict.txt"            # OCR 字符字典文件
VIDEO_IN = ROOT + "/demo.mp4"                     # 待测输入视频

# ---- 检测超参数 -------------------------------------------------------------
DET_SIZE, DET_CONF, DET_IOU = 320, 0.35, 0.30     # 检测输入尺寸 / 置信度阈值 / NMS IoU 阈值

# ---- OCR 超参数 -------------------------------------------------------------
OCR_H, OCR_W, OCR_CONF = 48, 320, 0.25            # OCR 输入高/宽、平均置信度阈值
OCR_INTERVAL = 5                                  # 每隔多少帧做一次 OCR（抽帧识别）

# ---- 坐标微调/外扩（经验值，用于补偿模型偏差、给文字留边）--------------------
DET_SHIFT_X, DET_SHIFT_Y = 4.0, 5.0               # 检测框中心点偏移补偿
OCR_PAD_X, OCR_PAD_Y = 0.10, 0.20                 # OCR 裁剪时在框四周外扩的比例


def letterbox(img, size=320):
    """等比缩放并加灰边（letterbox），把任意比例图像填入 size x size 正方形。

    返回:
        (缩放加边后的图, 缩放比例 r, 左上角 padding 偏移 (left, top))
    这是 YOLO 推理前标准预处理，保证输入不拉伸变形。
    """
    h, w = img.shape[:2]                          # 原图高、宽
    r = min(size / h, size / w)                   # 取较小缩放比，保证整图不超出目标尺寸
    nw, nh = int(round(w * r)), int(round(h * r)) # 缩放后的实际宽高
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # 计算上下/左右各需补多少灰边（像素）
    dw, dh = (size - nw) / 2.0, (size - nh) / 2.0
    # 用 ±0.1 做四舍五入，让左右/上下补边尽量对称
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
    right, bottom = int(round(dw + 0.1)), int(round(dh + 0.1))
    # 以灰色 (114,114,114) 填充边框，返回值含缩放比和左上角偏移（用于坐标反算）
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114)), r, (left, top)


def nms(boxes, scores, threshold):
    """非极大值抑制（NMS）：去掉重叠度(IoU)过高的重复框，保留置信度最高者。

    参数:
        boxes: 框列表，每个框为 [x1, y1, x2, y2]
        scores: 与 boxes 一一对应的置信度
        threshold: IoU 超过该值即视为同一目标，抑制低分框
    返回:
        保留下来的框索引列表
    """
    if not boxes:
        return []
    b = np.asarray(boxes, np.float32)             # 转为浮点数组便于向量化计算
    s = np.asarray(scores, np.float32)
    # 计算每个框的面积（宽*高，负值截断为 0）
    areas = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    order = s.argsort()[::-1]                     # 按置信度从高到低排序
    keep = []                                     # 最终保留的索引
    while order.size:
        i = int(order[0])                         # 取当前置信度最高的框
        keep.append(i)
        if order.size == 1:
            break
        # 计算当前框与其余框的交集区域坐标
        xx1 = np.maximum(b[i, 0], b[order[1:], 0]); yy1 = np.maximum(b[i, 1], b[order[1:], 1])
        xx2 = np.minimum(b[i, 2], b[order[1:], 2]); yy2 = np.minimum(b[i, 3], b[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)  # 交集面积
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)   # IoU，加 1e-6 防除零
        order = order[np.where(iou <= threshold)[0] + 1]             # 保留 IoU 不超阈值的框继续迭代
    return keep


def detect_timed(model, frame):
    """执行一帧车牌检测，并分别统计前处理/NPU推理/后处理耗时。

    返回:
        (detections, t_pre, t_npu, t_post)
        detections: [(框 [x1,y1,x2,y2], 置信度), ...]
        t_pre/t_npu/t_post: 各阶段耗时（毫秒）
    """
    # ---- 前处理：BGR→RGB，letterbox，扩展 batch 维 ----
    t0 = time.perf_counter()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # RKNN 模型输入为 RGB 顺序
    inp, ratio, pad = letterbox(rgb, DET_SIZE)     # 等比缩放加边到 320x320
    inp = np.expand_dims(inp, 0).astype(np.uint8)  # 增加 batch 维度，形状 (1,320,320,3)
    t_pre = (time.perf_counter() - t0) * 1000

    # ---- NPU 推理 ----
    t0 = time.perf_counter()
    outputs = model.inference(inputs=[inp], data_format=["nhwc"])  # NHWC 数据布局
    t_npu = (time.perf_counter() - t0) * 1000

    # ---- 后处理：解析输出框/置信度，坐标反算回原图 ----
    t0 = time.perf_counter()
    boxes = np.squeeze(np.asarray(outputs[0])).astype(np.float32)   # 输出 0: 框 (cx,cy,bw,bh)
    scores = np.squeeze(np.asarray(outputs[1])).astype(np.float32).reshape(-1)  # 输出 1: 置信度
    if boxes.ndim == 2 and boxes.shape[0] == 4:   # 若形状为 (4, N)，转置为 (N, 4)
        boxes = boxes.T
    candidates = []
    h0, w0 = frame.shape[:2]                       # 原图尺寸，用于坐标映射与裁剪
    for d, score in zip(boxes, scores):
        if float(score) < DET_CONF:                # 过滤低置信度框
            continue
        cx, cy, bw, bh = map(float, d[:4])         # 中心点 + 宽高
        # 中心点转左上/右下角，并加经验偏移补偿
        x1 = cx - bw / 2 + DET_SHIFT_X; y1 = cy - bh / 2 + DET_SHIFT_Y
        x2 = cx + bw / 2 + DET_SHIFT_X; y2 = cy + bh / 2 + DET_SHIFT_Y
        # 把 letterbox 坐标系坐标反算回原图坐标系（减去 padding 再除以缩放比）
        x1 = (x1 - pad[0]) / ratio; x2 = (x2 - pad[0]) / ratio
        y1 = (y1 - pad[1]) / ratio; y2 = (y2 - pad[1]) / ratio
        # 裁剪到原图范围内，避免越界
        x1 = max(0, min(x1, w0 - 1)); x2 = max(0, min(x2, w0 - 1))
        y1 = max(0, min(y1, h0 - 1)); y2 = max(0, min(y2, h0 - 1))
        if x2 > x1 and y2 > y1:                    # 有效框才保留
            candidates.append(([x1, y1, x2, y2], float(score)))
    # 用 NMS 去掉重叠重复框
    if candidates:
        keep = nms([x[0] for x in candidates], [x[1] for x in candidates], DET_IOU)
        detections = [candidates[i] for i in keep]
    else:
        detections = []
    t_post = (time.perf_counter() - t0) * 1000
    return detections, t_pre, t_npu, t_post


def crop_timed(frame, box):
    """按检测框裁剪车牌区域并做 OCR 预处理（外扩→缩放→填灰底），统计耗时。

    返回:
        (canvas, (x1,y1,x2,y2), t_ms)
        canvas: 形状 (1,48,320,3) 的 OCR 输入张量；若裁剪区为空则返回 None
    """
    t0 = time.perf_counter()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1                      # 框宽、框高
    # 按比例向外扩展裁剪区域（给文字留边，补偿检测框偏紧）
    x1 = max(0, int(x1 - bw * OCR_PAD_X)); x2 = min(w, int(x2 + bw * OCR_PAD_X))
    y1 = max(0, int(y1 - bh * OCR_PAD_Y)); y2 = min(h, int(y2 + bh * OCR_PAD_Y))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:                             # 裁剪区为空，直接返回
        return None, (x1, y1, x2, y2), (time.perf_counter() - t0) * 1000
    # 转 RGB，等比缩放到高度 48，宽度按比例（不超过 320）
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    ch, cw = rgb.shape[:2]
    scale = OCR_H / float(max(1, ch))              # 按高度 48 计算缩放比
    nw = min(OCR_W, max(1, int(round(cw * scale))))  # 缩放后宽度，上限 320
    resized = cv2.resize(rgb, (nw, OCR_H), interpolation=cv2.INTER_LINEAR)
    # 放到 127 灰底画布上，右侧不足 320 的部分保持灰底（PP-OCR 固定输入尺寸）
    canvas = np.full((OCR_H, OCR_W, 3), 127, dtype=np.uint8)
    canvas[:, :nw] = resized
    return canvas[None, ...], (x1, y1, x2, y2), (time.perf_counter() - t0) * 1000


def decode_ctc(out, chars):
    """对 PP-OCR 的 CTC 输出做贪心解码，得到识别文字，统计耗时。

    CTC 解码规则：每帧取概率最大的字符 id，跳过空白符(0)与连续重复字符。
    返回:
        (text, conf, t_ms)
        text: 平均置信度 >= OCR_CONF 时返回识别结果，否则返回空串
    """
    t0 = time.perf_counter()
    p = np.asarray(out[0])
    p = p[0] if p.ndim == 3 else p                  # 去掉 batch 维，得到 (时间步, 字符数)
    ids = np.argmax(p, axis=1)                     # 每个时间步概率最大的字符 id
    confs = np.max(p, axis=1)                      # 对应最大概率值
    text = []; vals = []; last = -1                # last 记录上一有效字符 id（用于去重）
    for idx, conf in zip(ids.tolist(), confs.tolist()):
        if idx == 0 or idx == last:                # 跳过空白符和连续重复字符
            last = idx
            continue
        last = idx
        if idx < len(chars):                       # id 在字典范围内才映射成字符
            text.append(chars[idx]); vals.append(float(conf))
    result = "".join(text).strip()                 # 拼接并去掉首尾空白
    c = float(np.mean(vals)) if vals else 0.0      # 取有效字符的平均置信度
    return (result if c >= OCR_CONF else ""), c, (time.perf_counter() - t0) * 1000


def main():
    # ---- 命令行参数 ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=ROOT + "/results/profile_320.csv")         # 逐帧明细输出路径
    ap.add_argument("--summary", default=ROOT + "/results/profile_320_summary.txt")  # 汇总统计输出路径
    ap.add_argument("--write-video", action="store_true")                        # 是否写结果视频
    args = ap.parse_args()

    # 读取 OCR 字符字典；下标 0 为空白符(CTC blank)，因此前面补一个空串
    chars = [""] + [
        x.rstrip("\r\n")
        for x in open(DICT_PATH, encoding="utf-8")
    ]

    # ---- 加载并初始化两个 RKNN 模型 ----
    det = RKNNLite()                    # 检测模型运行实例
    ocr = RKNNLite()                    # OCR 识别模型运行实例
    assert det.load_rknn(DET_MODEL) == 0   # 加载检测模型，失败即中止
    assert ocr.load_rknn(OCR_MODEL) == 0   # 加载 OCR 模型
    assert det.init_runtime() == 0         # 初始化检测模型 NPU 运行时
    assert ocr.init_runtime() == 0         # 初始化 OCR 模型 NPU 运行时

    # ---- 打开输入视频 ----
    cap = cv2.VideoCapture(VIDEO_IN)
    assert cap.isOpened()

    # ---- 可选：创建结果视频写入器 ----
    writer = None
    if args.write_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 视频宽
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # 视频高
        fps = cap.get(cv2.CAP_PROP_FPS) or 25         # 帧率，读不到则默认 25
        writer = cv2.VideoWriter(
            ROOT + "/results/profile_320.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),           # mp4v 编码
            fps,
            (w, h),
        )

    # ---- CSV 字段定义：逐帧记录各阶段耗时（毫秒）----
    fields = [
        "frame", "read_ms", "det_cpu_pre_ms", "det_npu_ms",
        "det_cpu_post_ms", "ocr_count", "ocr_cpu_pre_ms", "ocr_npu_ms",
        "ocr_cpu_decode_ms", "draw_ms", "write_ms", "total_ms",
    ]
    rows = []                     # 内存中累积每帧结果，最后做汇总统计
    frame_id = 0                  # 帧序号
    all_t0 = time.perf_counter()  # 整体计时起点（算总体 FPS）
    with open(args.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        while True:
            tframe = time.perf_counter()          # 单帧总耗时起点
            tr = time.perf_counter()
            ok, frame = cap.read()                # 读取一帧
            read = (time.perf_counter() - tr) * 1000
            if not ok:                            # 读到视频末尾则结束
                break

            frame_id += 1
            # 检测：返回检测框及前处理/NPU/后处理三段耗时
            dets, pre, npu, post = detect_timed(det, frame)
            # 初始化本帧 OCR 与绘制相关计时累加器
            oc = 0                                 # 本帧实际做 OCR 的次数
            op = on = od = draw = 0.0              # OCR 预处理/NPU/解码、绘制耗时
            for box, _ in dets:
                # 按 OCR_INTERVAL 抽帧识别，降低 OCR 计算开销
                if frame_id % OCR_INTERVAL == 0:
                    inp, draw_box, x = crop_timed(frame, box)   # 裁剪 + 预处理
                    op += x
                    if inp is not None:
                        t = time.perf_counter()
                        out = ocr.inference(inputs=[inp], data_format=["nhwc"])  # OCR NPU 推理
                        on += (time.perf_counter() - t) * 1000
                        _, _, x = decode_ctc(out, chars)        # CTC 解码
                        od += x
                        oc += 1
                # 在原图上画检测框（无论是否抽帧都画）
                t = time.perf_counter()
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw += (time.perf_counter() - t) * 1000
            # 可选：写结果视频并计时
            if writer is not None:
                t = time.perf_counter()
                writer.write(frame)
                write = (time.perf_counter() - t) * 1000
            else:
                write = 0.0
            # 汇总本帧所有字段，写入 CSV
            row = {"frame": frame_id, "read_ms": read, "det_cpu_pre_ms": pre,
                   "det_npu_ms": npu, "det_cpu_post_ms": post, "ocr_count": oc,
                   "ocr_cpu_pre_ms": op, "ocr_npu_ms": on, "ocr_cpu_decode_ms": od,
                   "draw_ms": draw, "write_ms": write,
                   "total_ms": (time.perf_counter() - tframe) * 1000}
            rows.append(row)
            wr.writerow(row)
    cap.release()
    if writer is not None:
        writer.release()

    # ---- 汇总统计：计算各阶段 avg / p50 / p95 / max ----
    keys = fields[1:]   # 去掉 "frame" 列，其余字段都参与统计
    lines = ["frames=%d elapsed_s=%.3f overall_fps=%.3f" % (
        frame_id, time.perf_counter() - all_t0,
        frame_id / max(1e-9, time.perf_counter() - all_t0))]
    for k in keys:
        a = np.array([float(r[k]) for r in rows], np.float64)
        lines.append("%-20s avg=%8.3f ms p50=%8.3f ms p95=%8.3f ms max=%8.3f ms" % (
            k, a.mean(), np.percentile(a, 50), np.percentile(a, 95), a.max()))
    with open(args.summary, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("CSV", args.csv)
    print("SUMMARY", args.summary)
    det.release()
    ocr.release()


if __name__ == "__main__":
    main()
