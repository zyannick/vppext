#ifndef HOUGH_IMAGE_HH
#define HOUGH_IMAGE_HH



#include <ctime>
#include <cstdlib>

#include <iod/parse_command_line.hh>
#include <iod/timer.hh>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>
#include <vpp/algorithms/video_extruder.hh>
#include <vpp/utils/opencv_utils.hh>
#include <vpp/draw/draw_trajectories.hh>
#include <draw_trajectories_hough.hh>


#include <eigen3/Eigen/Core>
//#include <unsupported/Eigen/BVH>

#include "symbols.hh"
#include "kalman.hh"


using namespace vpp;
using namespace std;
using namespace Eigen;
using namespace iod;
using namespace cv;

namespace vppx{

enum class Theta_max : int32_t { SMALL = 255, MEDIUM = 500, LARGE = 1000 , XLARGE = 1500};
enum class Type_video_hough : int8_t { ONLY_CLUSTERS = 1 , ALL_POINTS = 2 };
enum class Type_capture : int16_t { webcam = 0 , photo = 1 , video = 3 } ;
enum class Type_tracking : int16_t { lucas_kanade = 0 , kalman_track = 1 , orb = 2 , sift = 3 , surf = 4 , adapt = 5 };
void Hough_Accumulator(image2d<vuchar1> img, int mode , int T_theta);
cv::Mat Hough_Accumulator_Video_Map_and_Clusters(image2d<vuchar1> img, int mode , int T_theta,
                                                 std::vector<float>& t_accumulator, std::list<vint2>& interestedPoints,
                                                 float rhomax);
cv::Mat Hough_Accumulator_Video_Clusters(image2d<vuchar1> img, int mode , int T_theta,
                                         std::vector<float>& t_accumulator,std::list<vint2>& interestedPoints,
                                         float rhomax);
std::list<vint2> Hough_Lines_Parallel(image2d<vuchar1> img, std::vector<float>& t_accumulator,  int Theta_max,  float &max_of_the_accu ,int kernel_size);
std::list<vint2> Hough_Lines_Parallel_new(image2d<vuchar1> img,
                                      std::vector<float>& t_accumulator,
                                      int Theta_max, float& max_of_the_accu, int kernel_size);
std::list<vint2> Hough_Lines_Parallel_Box(image2d<vuchar1> img,
                                          std::vector<float>& t_accumulator,
                                          int Theta_max, float& max_of_the_accu);
std::list<vint2> Hough_Lines_Parallel_V2(image2d<vuchar1> img,
                                      std::vector<float>& t_accumulator,
                                      int Theta_max, float& max_of_the_accu, int threshold,
                                         image2d<vuchar3> &cluster_colors , int nb_old);
std::priority_queue<vint2> Hough_Lines_Parallel_V3(image2d<vuchar1> img,
                                      std::vector<float>& t_accumulator,
                                      int Theta_max, float& max_of_the_accu, int threshold,
                                         image2d<vuchar3> &cluster_colors , int nb_old);
void adap_thresold(std::list<vfloat3> &list_temp , float &threshold_hough , int &calls ,
                   int &nb_calls_limits_reached , int rhomax, int T_theta, std::vector<float> t_accumulator);
void reduce_number_of_max_local(std::list<vfloat3> &list_temp , float threshold_hough , int rhomax, int T_theta , std::vector<float> t_accumulator);
void Hough_Lines_Parallel_Map(image2d<vuchar1> img);
int getThetaMax(Theta_max discr);
cv::Mat accumulatorToFrame(std::vector<float> t_accumulator, float max, int rhomax, int T_theta);
cv::Mat accumulatorToFrame(std::list<vint2> interestedPoints, int rhomax, int T_theta);
void Capture_Image(int mode, Theta_max discr,Type_video_hough type_video );

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
#define mode_capture_try 4
#define mode_capture_vid_try 5
#define mode_capture_kalman 6
Matrix<float,3,3> GySobel3x3;
Matrix<float,3,3> GxSobel3x3;
Matrix<float,5,5> GySobel5x5;
Matrix<float,5,5> GxSobel5x5;
Matrix<float,7,7> GySobel7x7;
Matrix<float,7,7> GxSobel7x7;
Matrix<float,9,9> GySobel9x9;
Matrix<float,9,9> GxSobel9x9;


typedef vpp::keypoint_container<keypoint<int>, int> cluster_container_int;
typedef vpp::keypoint_container<keypoint<float>, float> cluster_container_float;

}

#include "fast_hough.hpp"



#endif // HOUGH_IMAGE_HH
