#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>
#include <QImage>
#include <QPixmap>
#include <QPainter>
#include <QStringList>
#include <QTimer>
#include <QElapsedTimer>
#include <QFile>


// =============================================================
// 构造函数
// =============================================================

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , pythonProcess(new QProcess(this))
{
    ui->setupUi(this);

    setWindowFlags(Qt::FramelessWindowHint);
    showFullScreen();

    // RK3568 DSI 面板为 720x1280 竖向布局。上方预留摄像头画面区域，
    // 下方保留三个运行状态指示区域。
    ui->cameraLabel->setGeometry(20, 70, 680, 520);
    ui->cameraLabel->setScaledContents(false);
    ui->cameraLabel->setAlignment(Qt::AlignCenter);

    // 竖向布局依次包含视频、三个易读的状态卡片和控制按钮。
    ui->topImageLabel->setGeometry(20, 12, 680, 42);
    ui->topImageLabel->setText(QStringLiteral("RK3568 车牌识别"));
    ui->topImageLabel->setStyleSheet(
        "QLabel { color:#ffffff; background:#14213d; border-radius:10px; "
        "font-size:24px; font-weight:bold; padding:6px; }");
    ui->cameraLabel->setStyleSheet(
        "QLabel { color:#dbeafe; background:#07111f; border:2px solid #3b82f6; "
        "border-radius:10px; }");
    ui->yoloLabel->setGeometry(20, 610, 210, 72);
    ui->ocrLabel->setGeometry(240, 610, 310, 72);
    ui->fpsLabel->setGeometry(560, 610, 140, 72);
    ui->yoloLabel->setAlignment(Qt::AlignCenter);
    ui->ocrLabel->setAlignment(Qt::AlignCenter);
    ui->fpsLabel->setAlignment(Qt::AlignCenter);
    const QString cardStyle =
        "QLabel { color:#ffffff; background:rgba(15,23,42,235); "
        "border:2px solid #334155; border-radius:12px; "
        "font-size:22px; font-weight:bold; padding:5px; }";
    ui->yoloLabel->setStyleSheet(cardStyle);
    ui->ocrLabel->setStyleSheet(cardStyle +
        "QLabel { border-color:#22c55e; color:#bbf7d0; }");
    ui->fpsLabel->setStyleSheet(cardStyle +
        "QLabel { border-color:#f59e0b; color:#fef3c7; }");
    ui->plateLabel->hide();
    ui->startButton->setGeometry(20, 720, 330, 72);
    ui->stopButton->setGeometry(370, 720, 330, 72);
    ui->ledButton->setGeometry(20, 810, 330, 72);
    ui->settingButton->setGeometry(370, 810, 330, 72);
    ui->startButton->setStyleSheet(
        "QPushButton { background:#2563eb; color:white; border:0; border-radius:12px; "
        "font-size:22px; font-weight:bold; } QPushButton:pressed { background:#1d4ed8; }");
    ui->stopButton->setStyleSheet(
        "QPushButton { background:#dc2626; color:white; border:0; border-radius:12px; "
        "font-size:22px; font-weight:bold; } QPushButton:pressed { background:#b91c1c; }");
    ui->ledButton->setStyleSheet(
        "QPushButton { background:#475569; color:white; border:0; border-radius:12px; "
        "font-size:22px; font-weight:bold; }");
    ui->settingButton->setStyleSheet(
        "QPushButton { background:#475569; color:white; border:0; border-radius:12px; "
        "font-size:22px; font-weight:bold; }");

    // 在第一次点击按钮前加载 LED 内核模块。开发板上驱动文件位于内核模块目录，
    // 加载后会创建对应的设备节点。
    if (!QFile::exists("/dev/dtsplatled")) {
        const int rc = QProcess::execute(
            "/usr/sbin/insmod",
            QStringList() << "/lib/modules/5.10.160/leddriver.ko");
        qDebug() << "leddriver insmod rc=" << rc;
    }
    ui->ledButton->setCheckable(true);
    ui->ledButton->setChecked(false);
    ui->ledButton->setText(QStringLiteral("LED OFF"));
    connect(ui->ledButton, &QPushButton::clicked,
            this, &MainWindow::on_ledButton_clicked);

    // =========================================================
    // 调整状态信息 Label
    // =========================================================

    ui->yoloLabel->setGeometry(
        40,
        740,
        250,
        50
    );

    ui->ocrLabel->setGeometry(
        300,
        740,
        200,
        50
    );

    ui->fpsLabel->setGeometry(
        520,
        740,
        140,
        50
    );

    // 在旧版几何布局代码执行后，重新应用最终的竖向卡片位置。
    ui->yoloLabel->setGeometry(20, 610, 210, 72);
    ui->ocrLabel->setGeometry(240, 610, 310, 72);
    ui->fpsLabel->setGeometry(560, 610, 140, 72);


    // =========================================================
    // Python stdout
    //
    // RK3568 Python stdout：
    //
    // [4字节JPEG长度][JPEG数据]
    //
    // =========================================================

    connect(
        pythonProcess,
        &QProcess::readyReadStandardOutput,
        this,
        [this]()
        {
            imageBuffer.append(
                pythonProcess->readAllStandardOutput()
            );

            processImageData();
        }
    );


    // =========================================================
    // Python stderr
    //
    // RK3568 Python：
    //
    // PLATE:xxx
    // FPS:3.12
    // YOLO:0.812,TIME:230.1ms,LPR:27.3ms
    //
    // =========================================================

    connect(
        pythonProcess,
        &QProcess::readyReadStandardError,
        this,
        [this]()
        {
            statusBuffer.append(
                pythonProcess->readAllStandardError()
            );

            while (true)
            {
                int index =
                    statusBuffer.indexOf('\n');

                if (index < 0)
                {
                    break;
                }

                QByteArray lineData =
                    statusBuffer.left(index);

                statusBuffer.remove(
                    0,
                    index + 1
                );

                QString line =
                    QString::fromLocal8Bit(
                        lineData
                    ).trimmed();

                if (line.isEmpty())
                {
                    continue;
                }

                if (line.startsWith("STATUS "))
                {
                    const QStringList fields = line.mid(7).split(
                        ' ', QString::SkipEmptyParts);
                    for (const QString &field : fields)
                    {
                        if (field.startsWith("FPS="))
                        {
                            bool ok = false;
                            const double value = field.mid(4).toDouble(&ok);
                            if (ok)
                                ui->fpsLabel->setText(QString("FPS: %1").arg(value, 0, 'f', 1));
                        }
                        else if (field.startsWith("PLATE="))
                        {
                            const QString plate = field.mid(6).trimmed();
                            // 空的 OCR 结果通常是暂态数据；在识别出新车牌前保留
                            // 上一次有效的车牌号。
                            if (!plate.isEmpty())
                                ui->ocrLabel->setText(QString("OCR: %1").arg(plate));
                        }
                        else if (field.startsWith("COUNT="))
                        {
                            ui->yoloLabel->setText(QStringLiteral("目标数: ") + field.mid(6));
                        }
                    }
                    continue;
                }

                qDebug()
                    << "RK3568:"
                    << line;


                // =================================================
                // OCR / 车牌号
                //
                // Python：
                // PLATE:冀AA9523
                //
                // Qt：
                // OCR：冀AA9523
                // =================================================

                if (line.startsWith("PLATE:"))
                {
                    QString plateText =
                        line.mid(6).trimmed();

                    if (plateText.isEmpty())
                    {
                        ui->ocrLabel->setText(
                            "OCR：未识别"
                        );
                    }
                    else
                    {
                        ui->ocrLabel->setText(
                            QString("OCR：%1")
                                .arg(plateText)
                        );
                    }

                    continue;
                }


                // =================================================
                // FPS
                // =================================================

                if (line.startsWith("FPS:"))
                {
                    QString fpsText =
                        line.mid(4).trimmed();

                    bool ok = false;

                    double fps =
                        fpsText.toDouble(&ok);

                    if (ok)
                    {
                        ui->fpsLabel->setText(
                            QString("FPS：%1")
                                .arg(
                                    fps,
                                    0,
                                    'f',
                                    1
                                )
                        );
                    }

                    continue;
                }


                // =================================================
                // YOLO
                //
                // Python：
                //
                // YOLO:0.812,TIME:230.1ms,LPR:27.3ms
                //
                // Qt：
                //
                // YOLO：0.812 230.1ms
                //
                // 注意：
                // LPR 的耗时不再显示到 OCR。
                // OCR 显示的是 PLATE: 中的车牌号。
                // =================================================

                if (line.startsWith("YOLO:"))
                {
                    QString yoloData =
                        line.mid(5).trimmed();

                    QStringList parts =
                        yoloData.split(
                            ',',
                            QString::SkipEmptyParts
                        );

                    QString confText = "--";
                    QString timeText = "--";

                    for (const QString &part : parts)
                    {
                        if (part.startsWith("TIME:"))
                        {
                            timeText =
                                part.mid(5).trimmed();
                        }
                        else if (part.startsWith("LPR:"))
                        {
                            // LPR耗时不显示到OCR栏
                            continue;
                        }
                        else
                        {
                            confText =
                                part.trimmed();
                        }
                    }

                    ui->yoloLabel->setText(
                        QString("YOLO：%1  %2")
                            .arg(
                                confText,
                                timeText
                            )
                    );

                    continue;
                }


                // =================================================
                // 其他信息
                //
                // RKNN warning、启动信息等直接显示到调试窗口
                // 不参与 UI 数据解析。
                // =================================================
            }
        }
    );


    // =========================================================
    // Python / SSH 进程错误
    // =========================================================

    connect(
        pythonProcess,
        QOverload<QProcess::ProcessError>::of(
            &QProcess::errorOccurred
        ),
        this,
        [this](QProcess::ProcessError error)
        {
            qDebug()
                << "SSH/Python进程错误:"
                << error;

            ui->ocrLabel->setText(
                "OCR：SSH启动失败"
            );
        }
    );


    // =========================================================
    // 进程退出
    // =========================================================

    connect(
        pythonProcess,
        QOverload<int, QProcess::ExitStatus>::of(
            &QProcess::finished
        ),
        this,
        [this](
            int exitCode,
            QProcess::ExitStatus exitStatus
        )
        {
            qDebug()
                << "RK3568程序退出:"
                << "code="
                << exitCode
                << "status="
                << exitStatus;
        }
    );

    // 在嵌入式面板上自动启动摄像头处理管线。
    QTimer::singleShot(800, this, &MainWindow::on_startButton_clicked);
}


// =============================================================
// 绘制背景
// =============================================================

void MainWindow::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);

    QPainter painter(this);

    QPixmap background(
        ":/new/prefix1/back.png"
    );

    if (background.isNull())
    {
        qDebug()
            << "背景图片加载失败";

        return;
    }

    painter.drawPixmap(
        rect(),
        background
    );
}


// =============================================================
// 析构函数
// =============================================================

MainWindow::~MainWindow()
{
    if (
        pythonProcess->state()
        !=
        QProcess::NotRunning
    )
    {
        pythonProcess->kill();

        pythonProcess->waitForFinished();
    }

    delete ui;
}

void MainWindow::on_ledButton_clicked(bool checked)
{
    QFile led("/dev/dtsplatled");
    if (!led.open(QIODevice::ReadWrite)) {
        qWarning() << "cannot open /dev/dtsplatled:" << led.errorString();
        ui->ledButton->setChecked(!checked);
        ui->ledButton->setText(checked ? QStringLiteral("LED OFF")
                                       : QStringLiteral("LED ON"));
        return;
    }

    const unsigned char value = checked ? 1 : 0;
    const qint64 written = led.write(reinterpret_cast<const char *>(&value), 1);
    led.close();
    if (written != 1) {
        qWarning() << "LED write failed" << led.errorString();
        ui->ledButton->setChecked(!checked);
        return;
    }
    ui->ledButton->setText(checked ? QStringLiteral("LED ON")
                                  : QStringLiteral("LED OFF"));
    qDebug() << "LED" << (checked ? "ON" : "OFF");
}


// =============================================================
// 开始识别
// =============================================================

void MainWindow::on_startButton_clicked()
{
    // =========================================================
    // 防止重复启动
    // =========================================================

    if (
        pythonProcess->state()
        !=
        QProcess::NotRunning
    )
    {
        qDebug()
            << "RK3568程序已经在运行";

        return;
    }


    // =========================================================
    // 清空缓存
    // =========================================================

    imageBuffer.clear();

    statusBuffer.clear();


    // =========================================================
    // 恢复界面初始状态
    // =========================================================

    ui->cameraLabel->clear();

    ui->cameraLabel->setText(
        "正在连接 RK3568 摄像头..."
    );

    ui->yoloLabel->setText(
        "YOLO：--"
    );

    ui->ocrLabel->setText(
        "OCR：--"
    );

    ui->fpsLabel->setText(
        "FPS：--"
    );

    // 不再单独显示车牌
    ui->plateLabel->clear();


    // =========================================================
    // SSH 启动 RK3568 Python
    //
    // Windows：
    //
    // ssh root@192.168.137.2
    //
    // RK3568：
    //
    // python3
    // /root/rknn_yolo/yolo_rknn_camera_lpr.py
    //
    // =========================================================

    /* 在 RK3568 本机运行；旧版 plink 命令仅适用于运行在 Windows 主机上的
       Qt 程序，无法在开发板上工作。 */
    const QStringList arguments = QStringList()
        << "/root/rknn_yolo/license_plate_ocr_camera_320.py"
        << "--camera" << "/dev/video0"
        << "--width" << "1280"
        << "--height" << "720"
              // 与当前端到端处理能力匹配。带丢弃策略的 GStreamer 队列将延迟限制在
              // 最新帧，避免积累数秒的过时画面。
              << "--fps" << "15"
        << "--no-display" << "--qt-output";

    /*
    const QString remoteCommand =
        "/home/hhh/tools/platform-tools/adb shell env "
        "RKNN_ROI=0.05,0.08,0.95,0.98 "
        "RKNN_DETECT_INTERVAL=2 "
        "RKNN_LPR_INTERVAL=2 "
        "RKNN_JPEG_QUALITY=65 "
        "python3 /root/rknn_yolo/license_plate_ocr_camera_320.py "
        "--camera /dev/video0 --width 1280 --height 720 --fps 30 "
        "--no-display --qt-output";


    QStringList oldArguments;

    oldArguments
        << "-batch"
        << "-hostkey"
        << "SHA256:9LWDDdNs2Q8vfUK5ctYxeO86xFSfwOS2cwLTWj6VTG8"
        << "-pw"
        << "0"
        << host
        << remoteCommand; */


    qDebug()
        << "启动 RK3568:"
        << "local python3"
        << arguments;


    // =========================================================
    // 启动 Windows OpenSSH
    //
    // Qt 5.12.9 MinGW 32-bit
    // 使用 Sysnative 访问真正的 System32
    // =========================================================

    pythonProcess->start(
        "python3",
        arguments
    );


    // =========================================================
    // 等待 SSH 进程启动
    // =========================================================

    if (!pythonProcess->waitForStarted(3000))
    {
        qDebug()
            << "SSH启动失败";

        ui->ocrLabel->setText(
            "OCR：SSH启动失败"
        );

        return;
    }


    qDebug()
        << "本机 Python 识别进程已启动";
}


// =============================================================
// 停止识别
// =============================================================

void MainWindow::on_stopButton_clicked()
{
    if (
        pythonProcess->state()
        ==
        QProcess::NotRunning
    )
    {
        qDebug()
            << "RK3568程序没有运行";

        return;
    }


    qDebug()
        << "停止 RK3568 识别";


    // =========================================================
    // 关闭 SSH
    //
    // QProcess::kill() 会杀掉 Windows ssh 进程，
    // SSH 断开后，远端程序通常会收到管道关闭。
    // =========================================================

    pythonProcess->kill();

    pythonProcess->waitForFinished(3000);


    // =========================================================
    // 清空缓存
    // =========================================================

    imageBuffer.clear();

    statusBuffer.clear();


    // =========================================================
    // 恢复界面
    // =========================================================

    ui->cameraLabel->clear();

    ui->cameraLabel->setText(
        "摄像头画面"
    );

    ui->yoloLabel->setText(
        "YOLO：--"
    );

    ui->ocrLabel->setText(
        "OCR：--"
    );

    ui->fpsLabel->setText(
        "FPS：--"
    );

    ui->plateLabel->clear();
}


// =============================================================
// 处理 Python 发送过来的 JPEG 数据
//
// Python：
//
// [4字节JPEG长度][JPEG数据]
// [4字节JPEG长度][JPEG数据]
// [4字节JPEG长度][JPEG数据]
//
// =============================================================

void MainWindow::processImageData()
{
    static int qtFrameCount = 0;
    static qint64 qtDecodeUs = 0;
    static qint64 qtDisplayUs = 0;
    while (true)
    {
        // =====================================================
        // 至少需要 4 字节长度
        // =====================================================

        if (
            imageBuffer.size()
            <
            4
        )
        {
            return;
        }


        // =====================================================
        // 读取 JPEG 长度
        //
        // Python：
        //
        // struct.pack("<I", len(jpeg_bytes))
        //
        // 小端
        // =====================================================

        const unsigned char *p =
            reinterpret_cast<
                const unsigned char *
            >(
                imageBuffer.constData()
            );


        quint32 dataSize =
            static_cast<quint32>(p[0])
            |
            (
                static_cast<quint32>(p[1])
                << 8
            )
            |
            (
                static_cast<quint32>(p[2])
                << 16
            )
            |
            (
                static_cast<quint32>(p[3])
                << 24
            );


        // =====================================================
        // 防止异常长度
        // =====================================================

        if (
            dataSize == 0
            ||
            dataSize > 10 * 1024 * 1024
        )
        {
            qDebug()
                << "JPEG数据长度异常:"
                << dataSize;

            imageBuffer.clear();

            return;
        }


        // =====================================================
        // 完整数据长度
        // =====================================================

        int totalSize =
            4
            +
            static_cast<int>(
                dataSize
            );


        // =====================================================
        // JPEG还没有接收完整
        // =====================================================

        if (
            imageBuffer.size()
            <
            totalSize
        )
        {
            return;
        }


        // =====================================================
        // 提取 JPEG
        // =====================================================

        QByteArray jpegData =
            imageBuffer.mid(
                4,
                static_cast<int>(
                    dataSize
                )
            );

        // 若网络/推理产生了积压，只保留缓冲区中最后一帧完整 JPEG，
        // 避免界面显示几秒前的旧画面（低延迟比显示每一帧更重要）。
        while (imageBuffer.size() >= totalSize + 4)
        {
            const unsigned char *next =
                reinterpret_cast<const unsigned char *>(
                    imageBuffer.constData() + totalSize);
            const quint32 nextSize = static_cast<quint32>(next[0]) |
                (static_cast<quint32>(next[1]) << 8) |
                (static_cast<quint32>(next[2]) << 16) |
                (static_cast<quint32>(next[3]) << 24);
            const int nextTotal = 4 + static_cast<int>(nextSize);
            if (nextSize == 0 || nextSize > 10 * 1024 * 1024 ||
                imageBuffer.size() < totalSize + nextTotal)
                break;
            jpegData = imageBuffer.mid(
                totalSize + 4,
                static_cast<int>(nextSize));
            totalSize += nextTotal;
        }


        // =====================================================
        // JPEG -> QImage
        // =====================================================

        QElapsedTimer qtTimer;
        qtTimer.start();
        QImage image;

        if (
            image.loadFromData(
                jpegData,
                "JPG"
            )
        )
        {
            qtDecodeUs += qtTimer.nsecsElapsed() / 1000;
            // =================================================
            // QImage -> QPixmap
            // =================================================

            QPixmap pixmap =
                QPixmap::fromImage(
                    image
                );


            // =================================================
            // 缩放到 cameraLabel
            // =================================================

            pixmap =
                pixmap.scaled(
                    ui->cameraLabel->size(),
                    Qt::KeepAspectRatio,
                    Qt::FastTransformation
                );


            // =================================================
            // 显示
            // =================================================

            qtTimer.restart();
            ui->cameraLabel->setPixmap(
                pixmap
            );
            qtDisplayUs += qtTimer.nsecsElapsed() / 1000;
            ++qtFrameCount;
            if (qtFrameCount % 30 == 0)
                qDebug() << "PROFILE_QT frames=" << qtFrameCount
                         << "jpeg_decode_ms=" << (qtDecodeUs / 30.0 / 1000.0)
                         << "display_ms=" << (qtDisplayUs / 30.0 / 1000.0);
            if (qtFrameCount % 30 == 0) { qtDecodeUs = 0; qtDisplayUs = 0; }
        }
        else
        {
            qDebug()
                << "JPEG解码失败";
        }


        // =====================================================
        // 删除已经处理的数据
        // =====================================================

        imageBuffer.remove(
            0,
            totalSize
        );
    }
}
