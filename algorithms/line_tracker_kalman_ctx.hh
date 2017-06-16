#ifndef LINE_TRACKER_KALMAN_CTX_HH
#define LINE_TRACKER_KALMAN_CTX_HH

#include <iod/array_view.hh>
#include <vpp/core/keypoint_trajectory.hh>
#include <vpp/core/keypoint_container.hh>
#include <vpp/core/symbols.hh>
#include <vpp/algorithms/symbols.hh>
#include <vpp/algorithms/optical_flow.hh>
#include <vpp/algorithms/fast_detector/fast.hh>
#include "kalman_tracker/kalmantracker.hh"
#include <hungarian_method/hungarian.hh>


namespace vppx
{
using namespace vpp;


struct line_tracker_kalman_ctx
{
  //line_tracker_kalman_ctx(box2d domain) : keypoints(domain), frame_id(0) , dt(0.0333) {}

  line_tracker_kalman_ctx(vint2 frameSize,
                             long lifetimeThreshold = 20,
                             float distanceThreshold = 0.1,
                             long missedFramesThreshold = 10,
                             float dt = 0.2,
                             float magnitudeOfAccelerationNoise = 0.5,
                             int lifetimeSuppressionThreshold = 20,
                             float distanceSuppressionThreshold = 0.1,
                             float ageSuppressionThreshold = 2);

  line_tracker_kalman_ctx(
                             long lifetimeThreshold = 20,
                             float distanceThreshold = 0.1,
                             long missedFramesThreshold = 10,
                             float dt = 0.2,
                             float magnitudeOfAccelerationNoise = 0.5,
                             int lifetimeSuppressionThreshold = 20,
                             float distanceSuppressionThreshold = 0.1,
                             float ageSuppressionThreshold = 2);

  // Keypoint container.
  // ctx.keypoint[i] to access the ith keypoint.
  //keypoint_container<keypoint<int>, int> keypoints;

  // Trajectory container.
  // ctx.trajectory[i].position_at_frame(j) to access the ith keypoint position at frame j.
  std::vector<keypoint_trajectory> trajectories;

  // Current frame id.
  int frame_id;



  std::vector<basic_kalman_tracker> basicKalmanTrackers;

  std::vector<unscented_kalman_tracker> unscentedKalmanTrackers;

  // We only care about trackers who have been alive for the
  // given lifetimeThreshold number of frames.
  long lifetimeThreshold;

  // The size of a frame.
  vint2 frameSize;

  // We won't associate a tracker with a mass center if the distance
  // between the two is greater than this fraction of the frame dimension
  // (taken as average between width and height).
  float distanceThreshold;

  // Kill a tracker if it has gone missedFramesThreshold frames
  // without receiving a measurement.
  long missedFramesThreshold;

  // Delta time, used to set up matrices for Kalman trackers.
  float dt;

  // Magnitude of acceleration noise. Used to set up Kalman trackers.
  float magnitudeOfAccelerationNoise;



  // Any Kalman filter with a lifetime above this value cannot be suppressed.
  int lifetimeSuppressionThreshold;

  // A Kalman filter can only be suppressed by another filter which is threshold * framDiagonal
  // or closer.
  float distanceSuppressionThreshold;

  // A Kalman filter can only be suppressed by another filter which is threshold times its age.
  float ageSuppressionThreshold;

  //template <typename K, typename MC, typename... OPTS>
  inline void
  tracking_in_hough_space(/*const K& keypoints,
                          MC match_callback,*/const std::vector<vfloat2>& massCenters,
                          /*const std::vector<cv::Rect>& boundingRects,*/
                          /*std::vector<TrackingOutput>& trackingOutputs,*/ int max_trajectory_size,
                          std::vector<float> i1,
                          std::vector<float> i2
                          /*OPTS... options*/);

  // Check if the Kalman filter at index i has another Kalman filter that can suppress it.
  bool hasSuppressorBasic(size_t i);
  bool hasSuppressorUnscented(size_t i);

};




}

#include "line_tracker_kalman_ctx.hpp"

#endif // LINE_TRACKER_KALMAN_CTX_HH
