#ifndef FEATURE_MATCHING_HOUGH_HPP
#define FEATURE_MATCHING_HOUGH_HPP

#include "feature_matching_hough.hh"
#include <iod/array_view.hh>
#include <vpp/core/keypoint_trajectory.hh>
#include <vpp/core/keypoint_container.hh>
#include <vpp/core/symbols.hh>
#include <vpp/algorithms/symbols.hh>
#include <vpp/algorithms/optical_flow.hh>
#include <vpp/algorithms/fast_detector/fast.hh>
#include <fast_hough.hh>
#include <vpp/algorithms/descriptor_matcher.hh>

/*****************/

/*
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"*/

namespace vppx {

using namespace vpp;
// Init the video extruder.
feature_matching_hough_ctx feature_matching_hough_init(box2d domain)
{
    feature_matching_hough_ctx res(domain);
    res.frame_id = -1;
    return res;
}


template <typename... OPTS>
void feature_matching_hough_update(feature_matching_hough_ctx& ftx,
                                   const image2d<unsigned char>& frame1,
                                   const image2d<unsigned char>& frame2,
                                   image2d<vuchar1> img,
                                   //float precision_runtime_balance = 0,
                                   OPTS... options)
{
    ftx.frame_id++;

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
    auto kps = Hough_Lines_Parallel(img,t_accumulator,T_theta,max_of_accu,3);
    auto frame3 = from_opencv<uchar>(accumulatorToFrame(kps,rhomax,T_theta));
    vpp::copy(frame3,frame2);



    // Filtering.
    // Merge particules that converged to the same pixel.
    {
        image2d<int> idx(frame2.domain().nrows() / keypoint_spacing,
                         frame2.domain().ncols() / keypoint_spacing, _border = 1);

        fill_with_border(idx, -1);
        for (int i = 0; i < ftx.keypoints.size(); i++)
        {
            vint2 pos = ftx.keypoints[i].position / keypoint_spacing;
            //assert(idx.has(pos));
            if (idx(pos) >= 0)
            {
                assert(idx(pos) < ftx.keypoints.size());
                auto other = ftx.keypoints[idx(pos)];
                if (other.age < ftx.keypoints[i].age)
                {
                    ftx.keypoints.remove(idx(pos));
                    idx(pos) = i;
                }
                if (other.age > ftx.keypoints[i].age)
                    ftx.keypoints.remove(i);
            }
            else
                idx(pos) = i;
        }
    }

    // Remove points with very low fast scores.
    // This is already done using the threshold to elimate cluster with few points

    // Detect new keypoints.
    {
        if (!(ftx.frame_id % detector_period))
        {
            image2d<unsigned char> mask(frame2.domain().nrows(),
                                        frame2.domain().ncols(),
                                        _border = keypoint_spacing);

            fill_with_border(mask, 1);
            for (int i = 0; i < ftx.keypoints.size(); i++)
            {
                int r = ftx.keypoints[i].position[0];
                int c = ftx.keypoints[i].position[1];
                for (int dr = -keypoint_spacing; dr < keypoint_spacing; dr++)
                    for (int dc = -keypoint_spacing; dc < keypoint_spacing; dc++)
                        mask[r + dr][c + dc] = 0;
            }
            for (auto kp : kps)
            {
                ftx.keypoints.add(keypoint<int>(kp));
            }
            ftx.keypoints.compact();
            cout << "the other " << endl;
            /*for (int i = 0; i < ctx.keypoints.size(); i++)
            {
                auto kp = ctx.keypoints[i];
                std::cout << "Point  " << kp.position << std::endl;
            }*/
            ftx.keypoints.sync_attributes(ftx.trajectories, keypoint_trajectory(ftx.frame_id));
        }
    }

    // Trajectories update.
    for (int i = 0; i < ftx.keypoints.size(); i++)
    {
        if (ftx.keypoints[i].alive())
        {
            ftx.trajectories[i].move_to(ftx.keypoints[i].position.template cast<float>());
            if (ftx.trajectories[i].size() > max_trajectory_length)
                ftx.trajectories[i].pop_oldest_position();
        }
        else
            ftx.trajectories[i].die();
    }

}


void brute_force_matching_basic_parallel(std::vector<vint2> descriptor1, std::vector<vint2> descriptor2,std::vector<int>& matches, int type)
{
    int size1 = descriptor1.size();
    int size2 = descriptor2.size();
    auto domain = make_box2d(size1,size2);
    std::vector<int> min_dist(size1,-1);
    image2d<float> mat_distance(domain);
    image2d<float> A(domain);
    pixel_wise(mat_distance, mat_distance.domain()) | [&] (auto &m, vint2 coord) {
        m = 0;
        int c2 = coord[1];
        int c1 = coord[0];
        float min_ = (descriptor1[c1] - descriptor2[c2]).norm();
        if(min_dist[c1]==-1 || min_dist[c1]<min_)
        {
            #pragma omp critical
            {
                min_dist[c1] = min_;
                matches[c1] = c2;
            }
        }
    };

}

void brute_force_matching_basic(std::vector<vint2> descriptor1, std::vector<vint2> descriptor2,std::vector<int>& matches, int type)
{
    int size1 = descriptor1.size();
    int size2 = descriptor2.size();
    auto domain = make_box2d(size1,size2);
    std::vector<int> min_dist(size1,-1);
    image2d<float> mat_distance(domain);
    image2d<float> A(domain);
    pixel_wise(mat_distance, mat_distance.domain()) | [&] (auto &m, vint2 coord) {
        m = 0;
        int c2 = coord[1];
        int c1 = coord[0];
        float min_ = (descriptor1[c1] - descriptor2[c2]).norm();
        if(min_dist[c1]==-1 || min_dist[c1]<min_)
        {
            #pragma omp critical
            {
                min_dist[c1] = min_;
                matches[c1] = c2;
            }
        }
    };

}



}

#endif // FEATURE_MATCHING_HPP
