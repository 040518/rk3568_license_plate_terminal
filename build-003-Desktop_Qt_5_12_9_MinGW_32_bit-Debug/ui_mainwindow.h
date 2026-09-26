/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.9
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QLabel *topImageLabel;
    QLabel *cameraLabel;
    QLabel *plateLabel;
    QLabel *yoloLabel;
    QLabel *ocrLabel;
    QLabel *fpsLabel;
    QPushButton *startButton;
    QPushButton *stopButton;
    QPushButton *ledButton;
    QPushButton *settingButton;
    QMenuBar *menubar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(720, 1280);
        QFont font;
        font.setFamily(QString::fromUtf8("Bahnschrift"));
        font.setPointSize(20);
        MainWindow->setFont(font);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        topImageLabel = new QLabel(centralwidget);
        topImageLabel->setObjectName(QString::fromUtf8("topImageLabel"));
        topImageLabel->setGeometry(QRect(170, 0, 341, 51));
        QFont font1;
        font1.setFamily(QString::fromUtf8("Agency FB"));
        font1.setPointSize(20);
        font1.setBold(true);
        font1.setWeight(75);
        topImageLabel->setFont(font1);
        topImageLabel->setAlignment(Qt::AlignCenter);
        cameraLabel = new QLabel(centralwidget);
        cameraLabel->setObjectName(QString::fromUtf8("cameraLabel"));
        cameraLabel->setGeometry(QRect(40, 100, 160, 120));
        cameraLabel->setStyleSheet(QString::fromUtf8("QLabel {\n"
"    border: 2px solid black;\n"
"    background-color: #202020;\n"
"    color: white;\n"
"}"));
        cameraLabel->setAlignment(Qt::AlignCenter);
        plateLabel = new QLabel(centralwidget);
        plateLabel->setObjectName(QString::fromUtf8("plateLabel"));
        plateLabel->setGeometry(QRect(40, 620, 640, 60));
        plateLabel->setFont(font);
        plateLabel->setAlignment(Qt::AlignCenter);
        yoloLabel = new QLabel(centralwidget);
        yoloLabel->setObjectName(QString::fromUtf8("yoloLabel"));
        yoloLabel->setGeometry(QRect(30, 700, 200, 50));
        QFont font2;
        font2.setFamily(QString::fromUtf8("Arial Narrow"));
        font2.setPointSize(20);
        yoloLabel->setFont(font2);
        yoloLabel->setStyleSheet(QString::fromUtf8("color: white;"));
        yoloLabel->setAlignment(Qt::AlignCenter);
        ocrLabel = new QLabel(centralwidget);
        ocrLabel->setObjectName(QString::fromUtf8("ocrLabel"));
        ocrLabel->setGeometry(QRect(270, 700, 200, 50));
        ocrLabel->setFont(font2);
        ocrLabel->setStyleSheet(QString::fromUtf8("color: white;"));
        fpsLabel = new QLabel(centralwidget);
        fpsLabel->setObjectName(QString::fromUtf8("fpsLabel"));
        fpsLabel->setGeometry(QRect(500, 700, 200, 50));
        fpsLabel->setFont(font2);
        fpsLabel->setStyleSheet(QString::fromUtf8("color: white;"));
        fpsLabel->setOpenExternalLinks(true);
        startButton = new QPushButton(centralwidget);
        startButton->setObjectName(QString::fromUtf8("startButton"));
        startButton->setGeometry(QRect(40, 790, 300, 70));
        stopButton = new QPushButton(centralwidget);
        stopButton->setObjectName(QString::fromUtf8("stopButton"));
        stopButton->setGeometry(QRect(380, 790, 300, 70));
        ledButton = new QPushButton(centralwidget);
        ledButton->setObjectName(QString::fromUtf8("ledButton"));
        ledButton->setGeometry(QRect(40, 880, 300, 70));
        settingButton = new QPushButton(centralwidget);
        settingButton->setObjectName(QString::fromUtf8("settingButton"));
        settingButton->setGeometry(QRect(380, 880, 300, 70));
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 720, 26));
        MainWindow->setMenuBar(menubar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "RK3568 AI \350\275\246\347\211\214\350\257\206\345\210\253", nullptr));
        topImageLabel->setText(QApplication::translate("MainWindow", "RK3568 AI \350\275\246\347\211\214\350\257\206\345\210\253\347\263\273\347\273\237", nullptr));
        cameraLabel->setText(QApplication::translate("MainWindow", "\346\221\204\345\203\217\345\244\264\347\224\273\351\235\242", nullptr));
        plateLabel->setText(QApplication::translate("MainWindow", "\350\275\246\347\211\214\357\274\232\347\255\211\345\276\205\350\257\206\345\210\253", nullptr));
        yoloLabel->setText(QApplication::translate("MainWindow", "YOLO\357\274\232--", nullptr));
        ocrLabel->setText(QApplication::translate("MainWindow", "OCR\357\274\232--", nullptr));
        fpsLabel->setText(QApplication::translate("MainWindow", "FPS\357\274\232--", nullptr));
        startButton->setText(QApplication::translate("MainWindow", "\345\274\200\345\247\213\350\257\206\345\210\253", nullptr));
        stopButton->setText(QApplication::translate("MainWindow", "\345\201\234\346\255\242\350\257\206\345\210\253", nullptr));
        ledButton->setText(QApplication::translate("MainWindow", "LED\346\216\247\345\210\266", nullptr));
        settingButton->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
