#ifndef VIDEO_EXTRUDER_HOUGH_HH
#define VIDEO_EXTRUDER_HOUGH_HH

#include <vpp/core/keypoint_trajectory.hh>
#include <vpp/core/keypoint_container.hh>
#include <vpp/core/symbols.hh>

namespace vppx
{

    using namespace vpp;
  // Video extruder context.
  struct video_extruder_hough_ctx
  {
    video_extruder_hough_ctx(box2d domain) : keypoints(domain), frame_id(0) {}

    // Keypoint container.
    // ctx.keypoint[i] to access the ith keypoint.
    keypoint_container<keypoint<int>, int> keypoints;

    // Trajectory container.
    // ctx.trajectory[i].position_at_frame(j) to access the ith keypoint position at frame j.
    std::vector<keypoint_trajectory> trajectories;

    // Current frame id.
    int frame_id;
  };

  // Initialize the video extruder context.
  // \param domain: The video domain.
  video_extruder_hough_ctx video_extruder_hough_init(box2d domain);

  // Update the video extruder with a new video frame.
  //
  // Options :
  //   _detector_th: FAST detector threshold.
  //   _keypoint_spacing: Keypoint spacing in pixels. 1 for maximum density.
  //   _max_trajectory_length: Maximum length of trajectories.
  //   _winsize: Window size used by the optical flow ssd distance
  //   _nscales: Number of scales
  //   _regularization: Number of iterations of the regularisation

  template <typename... OPTS>
  void video_extruder_hough_update(video_extruder_hough_ctx& ctx,
                             const image2d<unsigned char>& frame1,
                             const image2d<unsigned char>& frame2,
                                   image2d<vuchar1> img,int type_video,
                             OPTS... options);

  // Synchronize an array of tr
  // template <typename... OPTS>
  // void sync_trajectory_attributes(video_extruder_ctx& ctx,
  //                                 T& attrs,
  //                                 OPTS... options);

}

// Implementation
#include "video_extruder_hough.hpp"

#endif // VIDEO_EXTRUDER_HOUGH_HH
