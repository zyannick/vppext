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
#include <algorithms/fast_hough.hh>
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
                                    std::vector<float>& frame1,
                                    std::vector<float>& frame2,int type_video,image2d<vuchar1> img,
                                   int T_theta,int rhomax,bool first, std::list<vint2>& old_values,
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
    std::vector<float> t_accumulator(rhomax*T_theta);
    std::fill(t_accumulator.begin(),t_accumulator.end(),0);
    auto kps = Hough_Lines_Parallel_V2(img,t_accumulator,T_theta,max_of_accu);
    for(int i =0 ; i < frame2.size();i++)
    {
        frame2[i] = t_accumulator[i];
    }

    std::vector<std::vector<vfloat3>> three_best_matches;

    float threshold_norm = 50;


    if(first)
    {
        old_values = kps;
    }
    else
    {
        std::vector<int> matches;
        cout << "taille " << old_values.size() << endl;
        int old_max_rank = 0;
        for(auto &o : old_values)
        {
            std::vector<vfloat3> mins(3,vfloat3(0,0,-1));
            int max_rank = 0;
            for(auto &k : kps)
            {
                if( abs(old_max_rank-max_rank) > 10)
                {
                    break;
                }
                float the_norm = (o - k).squaredNorm();
                if(the_norm<threshold_norm)
                {
                    if((mins[0])[2]==-1 || the_norm<(mins[0])[2])
                    {
                        mins[2] = mins[1];
                        mins[1] = mins[0];
                        mins[0] = vfloat3(max_rank,old_max_rank,the_norm);
                    }
                    else if((mins[1])[2]==-1 ||the_norm<(mins[1])[2])
                    {
                        mins[2] = mins[1];
                        mins[1] = vfloat3(max_rank,old_max_rank,the_norm);
                    }
                    else if((mins[2])[2]==-1 || the_norm<(mins[2])[2])
                    {
                        mins[2] = vfloat3(max_rank,old_max_rank,the_norm);
                    }
                }
                max_rank++;
            }
            three_best_matches.push_back(mins);
            old_max_rank++;
        }

        std::vector<vint2> new_max_local;

        for(auto &k : kps)
        {
            new_max_local.push_back(k);
        }

        std::vector<vint2> old_max_local;

        for(auto &o : old_values)
        {
            old_max_local.push_back(o);
            //cout << o[0] << "   " << o[1] << endl;
        }
        int taille = 11;
        int mid = floor(taille/2);
        cout << "mid " << mid << endl;
        for(auto &three_max : three_best_matches)
        {
            std::vector<vfloat3> val = three_max;
            vfloat3 diff_mat = vfloat3(-1,-1,-1);

            for(int i = 0; i < val.size() ; i++)
            {
                std::vector<float> local_area1(taille*taille,0);
                std::vector<float> local_area2(taille*taille,0);
                if( (val[i])[2] != -1  )
                {
                    //cout << "here" << endl;
                    vint2 p2 = new_max_local[(val[i])[0]];
                    vint2 p1 = old_max_local[(val[i])[1]];
                    int a = 0;
                    for(int r = p1[0]-5 ; r < p1[0]+5 ;r++ ,a++)
                    {
                        int b =0;
                        for(int t = p1[1]-5 ; t < p1[1]+5 ; t++ ,b++)
                        {
                            //cout << "va5ls " << p1[0]  << "  " << p1[1] << endl;
                            local_area1[a*taille + b] = frame1[p1[0]*T_theta +p1[1] ];
                        }
                    }
                    a = 0;
                    for(int r = p2[0]-5 ; r < p2[0]+5 ;r++ , a++)
                    {
                        int b = 0;
                        for(int t = p2[1]-5 ; t < p2[1]+5 ; t++, b++)
                        {
                            //cout << "vals " << p2[0]  << "  " << p2[1] << endl;
                            //cout << "taille " << frame2.size() << " index " << p2[0]*T_theta +p2[1] << endl;
                            local_area2[a*taille + b] = frame2[p2[0]*T_theta +p2[1] ];
                        }
                    }
                    //cout << "here 4" << endl;
                    //diff_mat[i] = pearsoncoeff(local_area1,local_area2);
                    diff_mat[i] = Distance_between_curve_L2(local_area1,local_area2,taille);
                    cout << "val " << i << endl;
                    //cout << "problem" << endl;
                    cout << "p1 = [" << p1[0] << " , " << p1[1]  << "]  p2 = [ " << p2[0] << " , " << p2[1] << " ]    distance  = "
                         << diff_mat[i] << " et l'autre " << Distance_between_curve_L2(local_area1,local_area2,taille) << endl;
                }
            }
            MatrixXf::Index minRow, minCol;
            float min = diff_mat.minCoeff(&minRow, &minCol);
            int ind_match = minCol;

            //cout << "p1 " << p1{
        }
        old_values = kps;
    }




/*
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
    }*/

}

float Distance_between_curve_L1(std::vector<float>  frame1,std::vector<float>  frame2, int taille)
{
    float diff = 0;
    for(int i = 0 ; i < taille ; i++)
    {
        for(int j =0 ; j < taille ; j++)
        {
            diff += fabs(frame1[i * taille + j] - frame2[i * taille + j]);
        }
    }
    return diff;
}

float Distance_between_curve_L2(std::vector<float>  frame1, std::vector<float>  frame2, int taille)
{
    float diff = 0;
    for(int i = 0 ; i < taille ; i++)
    {
        for(int j =0 ; j < taille ; j++)
        {
            diff += pow((frame1[i * taille + j] - frame2[i * taille + j]),2);
            //cout << "diff " << diff << endl;
        }
    }
    diff = sqrt(diff);


    return diff;
}

float Correlation_Matrix_Pearson(Eigen::MatrixXd M1,Eigen::MatrixXd M2,int taille)
{

}


float sum_vector(std::vector<float> a)
{
    float s = 0;
    for (int i = 0; i < a.size(); i++)
    {
        s += a[i];
    }
    return s;
}

float mean_vector(std::vector<float> a)
{
    return sum_vector(a) / a.size();
}

float sqsum(std::vector<float> a)
{
    float s = 0;
    for (int i = 0; i < a.size(); i++)
    {
        s += pow(a[i], 2);
    }
    return s;
}

float stdev(std::vector<float> nums)
{
    float N = nums.size();
    return pow(sqsum(nums) / N - pow(sum_vector(nums) / N, 2), 0.5);
}

std::vector<float> operator-(std::vector<float> a, float b)
{
    std::vector<float> retvect;
    for (int i = 0; i < a.size(); i++)
    {
        retvect.push_back(a[i] - b);
    }
    return retvect;
}

std::vector<float> operator*(std::vector<float> a, std::vector<float> b)
{
    std::vector<float> retvect;
    for (int i = 0; i < a.size() ; i++)
    {
        retvect.push_back(a[i] * b[i]);
    }
    return retvect;
}

float pearsoncoeff(std::vector<float> X, std::vector<float> Y)
{
    return sum_vector((X - mean_vector(X))*(Y - mean_vector(Y))) / (X.size()*stdev(X)* stdev(Y));
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
