#ifndef FEATURE_MATCHING_HOUGH_HH
#define FEATURE_MATCHING_HOUGH_HH

#include <vpp/core/keypoint_trajectory.hh>
#include <vpp/core/keypoint_container.hh>
#include <vpp/core/symbols.hh>
#include <vpp/algorithms/descriptor_matcher.hh>


namespace vppx {

using namespace vpp;

struct feature_matching_hough_ctx {
    feature_matching_hough_ctx(box2d domain) : keypoints(domain), frame_id(0) {}

    // Keypoint container.
    // ctx.keypoint[i] to access the ith keypoint.
    keypoint_container<keypoint<int>, int> keypoints;

    // Trajectory container.
    // ctx.trajectory[i].position_at_frame(j) to access the ith keypoint position at frame j.
    std::vector<keypoint_trajectory> trajectories;

    // Current frame id.
    int frame_id;
};

struct kalman_tracker_hough_ctx {
    kalman_tracker_hough_ctx(vint2 startPt,
                             float dt = 0.2,
                             float magnitudeOfAccelerationNoise = 0.5,
                             size_t maxTrajectorySize = 20)  {}

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

    // Current frame id.
    std::unique_ptr<BasicKalmanFilter> kf;
    int numberOfFramesWithoutUpdate;
    int max_trajectory_size;
    // Store the latest prediction.
    vint2 prediction;
    vint2 lastPrediction;
    long lifetime;
    int id;
    vuchar3 color;
};

feature_matching_hough_ctx feature_matching_hough_init(box2d domain);

template <typename... OPTS>
void feature_matching_hough_update(feature_matching_hough_ctx& ctx,
                           const image2d<unsigned char>& frame1,
                           const image2d<unsigned char>& frame2,
                                 image2d<vuchar1> img, bool first,
                           OPTS... options);



void brute_force_matching(std::vector<vint2> descriptor1, std::vector<vint2> descriptor2,std::vector<int>& matches,int type);

}

#include "feature_matching_hough.hpp"

#endif // FEATURE_MATCHING_HOUGH_HH
