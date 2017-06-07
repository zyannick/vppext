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

feature_matching_hough_ctx feature_matching_hough_init(box2d domain);

template <typename... OPTS>
void feature_matching_hough_update(feature_matching_hough_ctx& ctx,
                           const image2d<unsigned char>& frame1,
                           const image2d<unsigned char>& frame2,
                                 image2d<vuchar1> img,
                           OPTS... options);



void brute_force_matching(std::vector<vint2> descriptor1, std::vector<vint2> descriptor2,std::vector<int>& matches,int type);

}

#include "feature_matching_hough.hpp"

#endif // FEATURE_MATCHING_HOUGH_HH
