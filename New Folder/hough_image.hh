/*#ifndef HOUGH_H
#define HOUGH_H*/
#pragma once


#include <ctime>
#include <cstdlib>


#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>
#include <vpp/algorithms/video_extruder.hh>
#include <vpp/utils/opencv_utils.hh>
#include <vpp/draw/draw_trajectories.hh>


#include <eigen3/Eigen/Core>
#include <unsupported/Eigen/BVH>

#include "symbols.hh"


using namespace vpp;
using namespace std;
using namespace Eigen;
using namespace iod;
using namespace cv;

namespace vppx{

    enum class Theta_max : int32_t { SMALL = 255, MEDIUM = 500, LARGE = 1000 , XLARGE = 1500};
    void Hough_Accumulator(image2d<vuchar1> img, int mode , int T_theta);
    cv::Mat Hough_Accumulator_Video_Map_and_Clusters(image2d<vuchar1> img, int mode , int T_theta, std::vector<float>& t_accumulator, std::list<vint2>& interestedPoints, std::vector<vshort4> &t_accumulator_point, float rhomax);
    cv::Mat Hough_Accumulator_Video_Clusters(image2d<vuchar1> img, int mode , int T_theta, std::vector<float>& t_accumulator,std::list<vint2>& interestedPoints,std::vector<vshort4>& t_accumulator_point, float rhomax);
    float Hough_Lines_Parallel(image2d<vuchar1> img,std::vector<float>& t_accumulator,std::list<vint2>& interestedPoints,std::vector<vshort4>& t_accumulator_point,int Theta_max);
    void Hough_Lines_Parallel_Map(image2d<vuchar1> img);
    int getThetaMax(Theta_max discr);
    cv::Mat accumulatorToFrame(std::vector<float> t_accumulator, float max, float rhomax, int T_theta);
    cv::Mat accumulatorToFrame(std::list<vint2> interestedPoints, float rhomax, int T_theta);
    void Capture_Image(int mode, Theta_max discr);

#define boule_theta_255 0.123685
#define boule_theta_1000 0.0314474
#define boule_theta_1440 0.0218318
#define hough_sequentiel 1
#define hough_parallel 2
#define hough_modified_kmeans 3
#define hough_parallel_map 7
#define hough_test 8
#define mode_capture_webcam 1
#define mode_capture_photo 2
#define mode_capture_video 3



    }

#include "hough_image.hpp"


//#endif // HOUGH_H
