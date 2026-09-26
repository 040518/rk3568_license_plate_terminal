#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QByteArray>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:

    // 开始识别
    void on_startButton_clicked();

    // 停止识别
    void on_stopButton_clicked();

    // 切换 /dev/dtsplatled（0=关闭，1=开启）。
    void on_ledButton_clicked(bool checked);

private:

    // 处理 Python 发送过来的 JPEG 数据
    void processImageData();

    // 绘制背景
    void paintEvent(QPaintEvent *event) override;

private:

    Ui::MainWindow *ui;

    // Python 进程
    QProcess *pythonProcess;

    // JPEG 数据缓存
    QByteArray imageBuffer;

    // STATUS 数据缓存
    QByteArray statusBuffer;
};

#endif // MAINWINDOW_H
