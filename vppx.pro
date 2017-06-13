TEMPLATE = app
CONFIG += c++14
CONFIG -= app_bundle
CONFIG -= qt


QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS +=  -fopenmp

INCLUDEPATH += /usr/local/include/opencv2
INCLUDEPATH += /usr/local/include/eigen3/Eigen
INCLUDEPATH += /usr/local/include/eigen3/unsupported/Eigen
INCLUDEPATH += /home/yannick/instvpp/include/
INCLUDEPATH += /home/yannick/nanoflann/include/


LIBS += -L/usr/local/lib
LIBS += -lopencv_core
LIBS += -lopencv_imgproc
LIBS += -lopencv_highgui
LIBS += -lopencv_videoio
LIBS += -lopencv_video
LIBS += -lopencv_imgcodecs
LIBS += -lboost_system
LIBS += -fopenmp

SOURCES += \
    main.cc

HEADERS += \
    vppx.hh \
    symbols.hh \
    video_extruder_hough.hh \
    video_extruder_hough.hpp \
    fast_hough.hh \
    fast_hough.hpp \
    draw_trajectories_hough.hh \
    feature_matching_hough.hh \
    feature_matching_hough.hpp \
    list_distances.hh \
    list_distances.hpp \
    drescriptor.hh \
    drescriptor.hpp \
    define.hh \
    operations.hh \
    kdtree.hh \
    kdtree.hpp \
    kalmantracker.hh \
    kalmantracker.hpp \
    hungarian.hh \
    hungarian.hpp \
    multipointtracker.hpp \
    multipointtracker.hh \
    tools.hh \
    tools.hpp \
    measurement.hh \
    basic_kalman_filter.hh \
    basic_kalman_filter.hpp \
    unscented_kalman_filter.hh \
    unscented_kalman_filter.hpp \
    kalman_filters/t.hh
