QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# 以下定义会让编译器在使用已标记为弃用的 Qt 功能时输出警告
#（具体警告内容取决于编译器）。请参考弃用 API 的文档进行代码迁移。
DEFINES += QT_DEPRECATED_WARNINGS

# 也可以让代码在使用弃用 API 时直接编译失败。
# 如需启用该功能，请取消下面一行的注释。
# 也可以只禁用指定 Qt 版本之前的弃用 API。
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # 禁用指定版本之前的所有弃用 API

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    camera.py \
    plate_video.py \
    plate_video_ocr.py \
    yolo_rknn_camera_lpr_board.py \
    RK3568_PERFORMANCE.md

RESOURCES += \
    resources.qrc
