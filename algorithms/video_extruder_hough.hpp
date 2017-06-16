#ifndef VIDEO_EXTRUDER_HOUGH_HPP
#define VIDEO_EXTRUDER_HOUGH_HPP



#include <iod/array_view.hh>
#include <vpp/core/keypoint_trajectory.hh>
#include <vpp/core/keypoint_container.hh>
#include <vpp/core/symbols.hh>
#include <vpp/algorithms/symbols.hh>
#include <vpp/algorithms/optical_flow.hh>
#include <vpp/algorithms/fast_detector/fast.hh>
#include <algorithms/fast_hough.hh>

namespace vppx
{
using namespace vpp;
// Init the video extruder.
video_extruder_hough_ctx video_extruder_hough_init(box2d domain)
{
    video_extruder_hough_ctx res(domain);
    res.frame_id = -1;
    return res;
}

// Update the video extruder with a new frame.
template <typename... OPTS>
void video_extruder_hough_update(video_extruder_hough_ctx& ctx,
                                 const image2d<unsigned char>& frame1,
                                 const image2d<unsigned char>& frame2,
                                 image2d<vuchar1> img,int mode_video,
                                 //float precision_runtime_balance = 0,
                                 OPTS... options)
{
    ctx.frame_id++;

    // Options.
    auto opts = D(options...);
    const int detector_th = opts.get(_detector_th, 10);
    const int keypoint_spacing = opts.get(_keypoint_spacing, 10);
    const int detector_period = opts.get(_detector_period, 5);
    const int max_trajectory_length = opts.get(_max_trajectory_length, 15);
    const int nscales = opts.get(_nscales, 3);
    const int winsize = opts.get(_winsize, 9);
    const int regularisation_niters = opts.get(_propagation, 2);

    float max_of_accu = 0;
    int T_theta = frame1.ncols();
    int rhomax = frame1.nrows();
    std::vector<float> t_accumulator(rhomax*T_theta);
    std::fill(t_accumulator.begin(),t_accumulator.end(),0);
    image2d<vuchar3> clusters_colors(make_box2d(rhomax,T_theta));
    auto kpp = Hough_Lines_Parallel_V2(img,t_accumulator,T_theta,max_of_accu);
    auto kps = Hough_Lines_Parallel_V2(img,t_accumulator,T_theta,max_of_accu);
    if(mode_video==2)
    {
        auto frame3 = from_opencv<uchar>(accumulatorToFrame(t_accumulator,max_of_accu,rhomax,T_theta));
        vpp::copy(frame3,frame2);
    }
    else if(mode_video==1)
    {
        auto frame3 = from_opencv<uchar>(accumulatorToFrame(kps,rhomax,T_theta));
        vpp::copy(frame3,frame2);
    }

/*
    // Optical flow vectors.
    ctx.keypoints.prepare_matching();
    semi_dense_optical_flow
            (iod::array_view(ctx.keypoints.size(),
                             [&] (int i) { return ctx.keypoints[i].position; })
             ,
             [&] (int i, vint2 pos, int distance) {
        if (frame1.has(pos))
            ctx.keypoints.move(i, pos);
        else ctx.keypoints.remove(i); },
    frame1, frame2,
    _winsize = winsize,
    _patchsize = 5,
    _propagation = regularisation_niters,
    _nscales = nscales);

    // Filtering.
    // Merge particules that converged to the same pixel.
    {
        image2d<int> idx(frame2.domain().nrows() / keypoint_spacing,
                         frame2.domain().ncols() / keypoint_spacing, _border = 1);

        fill_with_border(idx, -1);
        for (int i = 0; i < ctx.keypoints.size(); i++)
        {
            //cout << "position " << ctx.keypoints[i].position << endl;
            vint2 pos = ctx.keypoints[i].position / keypoint_spacing;
            //assert(idx.has(pos));
            if (idx(pos) >= 0)
            {
                assert(idx(pos) < ctx.keypoints.size());
                auto other = ctx.keypoints[idx(pos)];
                if (other.age < ctx.keypoints[i].age)
                {
                    ctx.keypoints.remove(idx(pos));
                    idx(pos) = i;
                }
                if (other.age > ctx.keypoints[i].age)
                    ctx.keypoints.remove(i);
            }
            else
                idx(pos) = i;
        }
    }

    // Remove points with very low fast scores.
    // This is already done using the threshold to elimate cluster with few points

    // Detect new keypoints.
    {
        if (!(ctx.frame_id % detector_period))
        {
            image2d<unsigned char> mask(frame2.domain().nrows(),
                                        frame2.domain().ncols(),
                                        _border = keypoint_spacing);

            fill_with_border(mask, 1);
            for (int i = 0; i < ctx.keypoints.size(); i++)
            {
                int r = ctx.keypoints[i].position[0];
                int c = ctx.keypoints[i].position[1];
                for (int dr = -keypoint_spacing; dr < keypoint_spacing; dr++)
                    for (int dc = -keypoint_spacing; dc < keypoint_spacing; dc++)
                        mask[r + dr][c + dc] = 0;
            }
            for (auto kp : kps)
            {
                std::cout << " rho 4 " << kp[0] << " theta " << kp[1] << std::endl;
                ctx.keypoints.add(keypoint<int>(kp));
            }
            ctx.keypoints.compact();
            //cout << "the other " << endl;

            ctx.keypoints.sync_attributes(ctx.trajectories, keypoint_trajectory(ctx.frame_id));
        }
    }

    // Trajectories update.
    for (int i = 0; i < ctx.keypoints.size(); i++)
    {
        if (ctx.keypoints[i].alive())
        {
            ctx.trajectories[i].move_to(ctx.keypoints[i].position.template cast<float>());
            if (ctx.trajectories[i].size() > max_trajectory_length)
                ctx.trajectories[i].pop_oldest_position();
        }
        else
            ctx.trajectories[i].die();
    }*/

}


line_tracker_kalman_ctx line_tracker_kalman_init(vint2 frameSize,
                                                 long lifetimeThreshold ,
                                                 float distanceThreshold ,
                                                 long missedFramesThreshold ,
                                                 float dt ,
                                                 float magnitudeOfAccelerationNoise ,
                                                 int lifetimeSuppressionThreshold ,
                                                 float distanceSuppressionThreshold ,
                                                 float ageSuppressionThreshold )
{
    line_tracker_kalman_ctx res(frameSize,lifetimeThreshold,distanceThreshold,missedFramesThreshold,
                                dt,magnitudeOfAccelerationNoise,lifetimeSuppressionThreshold,distanceSuppressionThreshold,ageSuppressionThreshold);
    res.frame_id = -1;
    return res;
}

line_tracker_kalman_ctx line_tracker_kalman_init(
                                                 long lifetimeThreshold ,
                                                 float distanceThreshold ,
                                                 long missedFramesThreshold ,
                                                 float dt ,
                                                 float magnitudeOfAccelerationNoise ,
                                                 int lifetimeSuppressionThreshold ,
                                                 float distanceSuppressionThreshold ,
                                                 float ageSuppressionThreshold )
{
    line_tracker_kalman_ctx res(lifetimeThreshold,distanceThreshold,missedFramesThreshold,
                                dt,magnitudeOfAccelerationNoise,lifetimeSuppressionThreshold,distanceSuppressionThreshold,ageSuppressionThreshold);
    res.frame_id = -1;
    return res;
}

// Update the video extruder with a new frame.
template <typename... OPTS>
void line_tracker_kalman_update(line_tracker_kalman_ctx& ctx,
                                 const image2d<float>& frame1,
                                 const image2d<float>& frame2,
                                std::vector<float> &_acc_1,
                                std::vector<float> &_acc_2,
                                bool first,
                                 image2d<vuchar1> img,int mode_video,
                                 //float precision_runtime_balance = 0,
                                 OPTS... options)
{
    ctx.frame_id++;

    // Options.
    auto opts = D(options...);

    const int life_time_threshold = opts.get(_life_time_threshold, 10);
    const int distance_threshold = opts.get(_distance_threshold, 10);
    const int missed_frame_threshold = opts.get(_missed_frame_threshold, 5);
    const int magnitude_acceleration_noise = opts.get(_magnitude_acceleration_noise, 5);
    const int life_time_supression_threshold = opts.get(_life_time_supression_threshold, 5);
    const int age_supression_threshold = opts.get(_age_supression_threshold, 5);
    const int max_trajectory_length = opts.get(_max_trajectory_length, 15);


    float max_of_accu = 0;
    int T_theta = frame1.ncols();
    int rhomax = frame1.nrows();
    std::vector<float> t_accumulator(rhomax*T_theta);
    std::fill(t_accumulator.begin(),t_accumulator.end(),0);
    image2d<vuchar3> clusters_colors(make_box2d(rhomax,T_theta));
    auto kps = vppx::Hough_Lines_Parallel_V2(img,t_accumulator,T_theta,max_of_accu);
    _acc_2 = t_accumulator;
    std::vector<vfloat2> massCenters;
    for(auto &k : kps)
    {
        massCenters.push_back(vfloat2(k[0],k[1]));
    }

    if(mode_video==2)
    {
        auto frame3 = from_opencv<uchar>(accumulatorToFrame(t_accumulator,max_of_accu,rhomax,T_theta));
        vpp::copy(frame3,frame2);
    }
    else if(mode_video==1)
    {
        auto frame3 = from_opencv<uchar>(accumulatorToFrame(kps,rhomax,T_theta));
        vpp::copy(frame3,frame2);
    }

    if(!first)
    ctx.tracking_in_hough_space(massCenters,3,_acc_1,_acc_2);

    _acc_1 = _acc_2;

    // Optical flow vectors.
    //ctx.keypoints.prepare_matching();



}

}


#endif // VIDEO_EXTRUDER_HOUGH_HPP
