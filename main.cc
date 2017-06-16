#include <iostream>
#include "algorithms/fast_hough.hh"

#include <chrono>

using namespace  std;
using namespace Eigen;
using namespace cv;
using namespace vpp;
using namespace std::chrono;
using namespace vppx;

int main(int argc, char *argv[])
{

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    Capture_Image(mode_capture_kalman,Theta_max::SMALL,Type_video_hough::ALL_POINTS,Type_tracking::video_extruder_video);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    cout << "la duree " << duration << endl;
    return 0;
}

