#ifndef MULTIKALMANFILTER_HH
#define MULTIKALMANFILTER_HH


#include <vpp/vpp.hh>
#include "kalman_filters/basic_kalman_filter.hh"
#include "kalman_filters/unscented_kalman_filter.hh"
using namespace vpp;
using namespace std;

/**
 * @brief The multikalmanfilter struct
 * from https://github.com/hariharsubramanyam/ObjectTracker/blob/master/include/tracker/kalman_tracker.hpp
 */


struct TrackingOutput {
       int id;
       vint2 location;
       vuchar3 color;
       std::vector<vint2> trajectory;
   };

struct basic_kalman_tracker{
    basic_kalman_tracker(vint2 startPt,
                          float dt = 0.2,
                          float magnitudeOfAccelerationNoise = 0.5,
                          size_t maxTrajectorySize = 20);

    const int getNumFramesWithoutUpdate();

    // Indicate to this Kalman filter that it did not get an update this frame.
    void noUpdateThisFrame();

    // Indicate that the Kalman filter was updated this frame.
    void gotUpdate();

    // Return the number of frames that this Kalman tracker has been alive.
    long getLifetime();

    vint2 predict();
    vint2 latestPrediction();
    vint2 correct(vint2 pt);
    TrackingOutput latestTrackingOutput();
//private :
    std::unique_ptr<BasicKalmanFilter> bkf;
    int numberOfFramesWithoutUpdate;
    int max_trajectory_size;
    // Store the latest prediction.
    vint2 prediction;
    std::shared_ptr<std::vector<vint2>> trajectory;
    vint2 lastPrediction;
    long lifetime;
    int id;
    vuchar3 color;
    void addPointToTrajectory(vint2 pt);

};

struct unscented_kalman_tracker{

    unscented_kalman_tracker() {}

    unscented_kalman_tracker(vint2 startPt,
                          float dt = 0.2,
                          int maxTrajectorySize = 20);

    const int getNumFramesWithoutUpdate();

    void neFaitrien() {}
    // Indicate to this Kalman filter that it did not get an update this frame.
    void noUpdateThisFrame();

    // Indicate that the Kalman filter was updated this frame.
    void gotUpdate();

    // Return the number of frames that this Kalman tracker has been alive.
    long getLifetime();

    vint2 predict();
    vint2 latestPrediction();
    vint2 correct(vint2 pt);
    TrackingOutput latestTrackingOutput();
//private :
    std::unique_ptr<Unscented_Kalman_Filter> ukf;
    float dt =0.33;
    int numberOfFramesWithoutUpdate;
    int max_trajectory_size;
    // Store the latest prediction.
    vint2 prediction;
    std::shared_ptr<std::vector<vint2>> trajectory;
    vint2 lastPrediction;
    long lifetime;
    int id;
    vuchar3 color;
    void addPointToTrajectory(vint2 pt);

};






#include "basickalmantracker.hpp"
#include "unscented_kalman_tracker.hpp"

#endif // MULTIKALMANFILTER_HH
