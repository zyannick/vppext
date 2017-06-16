#ifndef LINE_TRACKER_KALMAN_HPP
#define LINE_TRACKER_KALMAN_HPP

#include "line_tracker_kalman_ctx.hh"
#include <algorithms/fast_hough.hh>


namespace vppx
{

using namespace vpp;
using namespace Eigen;
// Init the video extruder.
/*line_tracker_kalman_ctx line_tracker_kalman_init(box2d domain)
{
    line_tracker_kalman_ctx res(domain);
    res.frame_id = -1;
    return res;
}*/



line_tracker_kalman_ctx::line_tracker_kalman_ctx (vint2 frameSize,
                                      long lifetimeThreshold,
                                      float distanceThreshold,
                                      long missedFramesThreshold,
                                      float dt,
                                      float magnitudeOfAccelerationNoise,
                                      int lifetimeSuppressionThreshold,
                                      float distanceSuppressionThreshold,
                                      float ageSuppressionThreshold) {
    this->unscentedKalmanTrackers = std::vector<unscented_kalman_tracker>();
    this->frameSize = frameSize;
    this->lifetimeThreshold = lifetimeThreshold;
    this->distanceThreshold = distanceThreshold;
    this->missedFramesThreshold = missedFramesThreshold;
    this->magnitudeOfAccelerationNoise = magnitudeOfAccelerationNoise;
    this->lifetimeSuppressionThreshold = lifetimeSuppressionThreshold;
    this->distanceSuppressionThreshold = distanceSuppressionThreshold;
    this->ageSuppressionThreshold = ageSuppressionThreshold;
    this->dt = dt;
}

line_tracker_kalman_ctx::line_tracker_kalman_ctx (long lifetimeThreshold,
                                      float distanceThreshold,
                                      long missedFramesThreshold,
                                      float dt,
                                      float magnitudeOfAccelerationNoise,
                                      int lifetimeSuppressionThreshold,
                                      float distanceSuppressionThreshold,
                                      float ageSuppressionThreshold) {
    this->unscentedKalmanTrackers = std::vector<unscented_kalman_tracker>();
    this->lifetimeThreshold = lifetimeThreshold;
    this->distanceThreshold = distanceThreshold;
    this->missedFramesThreshold = missedFramesThreshold;
    this->magnitudeOfAccelerationNoise = magnitudeOfAccelerationNoise;
    this->lifetimeSuppressionThreshold = lifetimeSuppressionThreshold;
    this->distanceSuppressionThreshold = distanceSuppressionThreshold;
    this->ageSuppressionThreshold = ageSuppressionThreshold;
    this->dt = dt;
}

//template <typename K, typename MC, typename... OPTS>
inline void
line_tracker_kalman_ctx::tracking_in_hough_space(/*const K& keypoints,
                        MC match_callback,*/const std::vector<vfloat2>& massCenters,
                                                    /*const std::vector<cv::Rect>& boundingRects,*/
                                                    /*std::vector<TrackingOutput>& trackingOutputs,*/int max_trajectory_size,
                         std::vector<float> frame1,
                         std::vector<float> frame2)
{


    //trackingOutputs.clear();

    cout << "appel du tracking" << endl;

    float dt = 0.033;

    // If we haven't found any mass centers, just update all the Kalman filters and return their predictions.
    if (massCenters.empty()) {
        for (int i = 0; i < this->unscentedKalmanTrackers.size(); i++) {
            // Indicate that the tracker didn't get an update this frame.
            this->unscentedKalmanTrackers[i].noUpdateThisFrame();

            // Remove the tracker if it is dead.
            if (this->unscentedKalmanTrackers[i].getNumFramesWithoutUpdate() > this->missedFramesThreshold) {
                this->unscentedKalmanTrackers.erase(this->unscentedKalmanTrackers.begin() + i);
                i--;
            }
        }
        // Update the remaining trackers.
        for (size_t i = 0; i < this->unscentedKalmanTrackers.size(); i++) {
            if (this->unscentedKalmanTrackers[i].getLifetime() > lifetimeThreshold) {
                this->unscentedKalmanTrackers[i].predict();
                //trackingOutputs.push_back(this->unscentedKalmanTrackers[i].latestTrackingOutput());
            }
        }
        return;
    }

    // If there are no Kalman trackers, make one for each detection.
    if (this->basicKalmanTrackers.empty()) {
        for (auto massCenter : massCenters) {
            this->unscentedKalmanTrackers.push_back(unscented_kalman_tracker(vint2(massCenter[0],massCenter[1]),this->dt,max_trajectory_size));
        }
    }


    // Create our cost matrix.
    size_t numKalmans = this->unscentedKalmanTrackers.size();
    size_t numCenters = massCenters.size();

    std::vector<std::vector<double>> costMatrix(numKalmans, std::vector<double>(numCenters));

    std::vector<int> assignment;


    // Get the latest prediction for the Kalman filters.
    std::vector<vfloat2> predictions(this->unscentedKalmanTrackers.size());
    for (size_t i = 0; i < this->unscentedKalmanTrackers.size(); i++) {
        vint2 tv = this->unscentedKalmanTrackers[i].latestPrediction();
        predictions[i] = vfloat2(tv[0],tv[1]);
    }

    // We need to associate each of the mass centers to their corresponding Kalman filter. First,
    // let's find the pairwise distances. However, we first divide this distance by the diagonal size
    // of the frame to ensure that it is between 0 and 1.
    vint2 framePoint = vint2(this->frameSize[0], this->frameSize[1]);
    int rhomax = this->frameSize[0];
    int T_theta = this->frameSize[1];
    double frameDiagonal = std::sqrt(framePoint.dot(framePoint));
    for (size_t i = 0; i < predictions.size(); i++) {
        for (size_t j = 0; j < massCenters.size(); j++) {
            vint2 p1 = vint2((predictions[i])[0],(predictions[i])[1]);
            vint2 p2 = vint2((massCenters[j])[0],(massCenters[j])[1]);
            //float diff1 = (predictions[i] - massCenters[j]).norm();
            MatrixXf predic(11,11);
            MatrixXf newValues(11,11);
            int a = 0;
            for(int r = p1[0]-5 ; r < p1[0]+5 ;r++,a++)
            {
                int b =0;
                for(int t = p1[1]-5 ; t < p1[1]+5 ; t++ ,b++)
                {
                    //cout << "va5ls " << p1[0]  << "  " << p1[1] << endl;
                    if(r>=rhomax || r<0 || t>=T_theta || t<0)
                    {
                        predic.coeffRef(a,b) = 0;
                    }
                    else
                    //cout << "r " << r << " t " << t << endl;
                    predic.coeffRef(a,b) = frame1[p1[0]*T_theta +p1[1] ];
                }
            }
            a = 0;
            for(int r = p2[0]-5 ; r < p2[0]+5 ;r++,a++)
            {
                int b = 0;
                for(int t = p2[1]-5 ; t < p2[1]+5 ; t++, b++)
                {
                    //cout << "vals " << p2[0]  << "  " << p2[1] << endl;
                    //cout << "taille " << frame2.size() << " index " << p2[0]*T_theta +p2[1] << endl;
                    if(r>=rhomax || r<0 || t>=T_theta || t<0)
                    {
                        newValues.coeffRef(a,b) = 0;
                    }
                    else
                    //cout << "r " << r << " t " << t << endl;
                    newValues.coeffRef(a,b) = frame2[p2[0]*T_theta +p2[1] ];
                }
            }
            float nor = (predic - newValues).norm();
            costMatrix[i][j] = (predictions[i] - massCenters[j]).norm() / frameDiagonal;
            //cout << " la difference " << (predictions[i] - massCenters[j]).norm() << endl;
        }
    }

    // Assign Kalman trackers to mass centers with the Hungarian algorithm.
    AssignmentProblemSolver solver;
    solver.Solve(costMatrix, assignment, AssignmentProblemSolver::optimal);

    // Unassign any Kalman trackers whose distance to their assignment is too large.
    std::vector<int> kalmansWithoutCenters;
    for (size_t i = 0; i < assignment.size(); i++) {
        if (assignment[i] != -1) {
            if (costMatrix[i][assignment[i]] > this->distanceThreshold) {
                assignment[i] = -1;
                kalmansWithoutCenters.push_back(i);
            }
        } else {
            this->unscentedKalmanTrackers[i].noUpdateThisFrame();
        }
    }
/*
    // If a Kalman tracker is contained in a bounding box and shares its
    // bounding box with another tracker, remove its assignment and mark it
    // as updated.
    /*for (size_t i = 0; i < assignment.size(); i++) {
        for (size_t j = 0; j < boundingRects.size(); j++) {
            if (boundingRects[j].contains(this->kalmanTrackers[i].latestPrediction())
                    && this->sharesBoundingRect(i, boundingRects[j])) {
                this->kalmanTrackers[i].gotUpdate();
                break;
            }
        }
    }*/

    // Remove any trackers that haven't been updated in a while.
    for (int i = 0; i < this->unscentedKalmanTrackers.size(); i++) {
        if (this->unscentedKalmanTrackers[i].getNumFramesWithoutUpdate() > this->missedFramesThreshold) {
            this->unscentedKalmanTrackers.erase(this->unscentedKalmanTrackers.begin() + i);
            assignment.erase(assignment.begin() + i);
            i--;
        }
    }

    // Find unassigned mass centers.
    std::vector<int> centersWithoutKalman;
    std::vector<int>::iterator it;
    for (size_t i = 0; i < massCenters.size(); i++) {
        it = std::find(assignment.begin(), assignment.end(), i);
        if (it == assignment.end()) {
            centersWithoutKalman.push_back(i);
        }
    }


    // Create new trackers for the unassigned mass centers.
    for (size_t i = 0; i < centersWithoutKalman.size(); i++) {
        vfloat2 to = massCenters[centersWithoutKalman[i]];
        this->unscentedKalmanTrackers.push_back(unscented_kalman_tracker(vint2(to[0],to[1])));
    }

    // Update the Kalman filters.
    for (size_t i = 0; i < assignment.size(); i++) {
        this->unscentedKalmanTrackers[i].predict();
        if (assignment[i] != -1) {
            vfloat2 tg = massCenters[assignment[i]];
            vint2 val(2);
            val << tg[0] , tg[1];
            this->unscentedKalmanTrackers[i].correct(val);
            this->unscentedKalmanTrackers[i].gotUpdate();
        }
    }

    // Remove any suppressed filters.
    for (size_t i = 0; i < this->unscentedKalmanTrackers.size(); i++) {
        if (this->hasSuppressorUnscented(i)) {
            this->unscentedKalmanTrackers.erase(this->unscentedKalmanTrackers.begin() + i);
            i--;
        }
    }

    // Now update the predictions.
    /*for (size_t i = 0; i < this->unscentedKalmanTrackers.size(); i++) {
        if (this->unscentedKalmanTrackers[i].getLifetime() > this->lifetimeThreshold) {
            trackingOutputs.push_back(this->unscentedKalmanTrackers[i].latestTrackingOutput());
        }
    }*/

}

bool line_tracker_kalman_ctx::hasSuppressorUnscented(size_t i) {
    double dist;
    vfloat2 framePoint = vfloat2(this->frameSize[0], this->frameSize[1]);
    double frameDiagonal = std::sqrt(framePoint.dot(framePoint));

    for (size_t j = 0; j < this->unscentedKalmanTrackers.size(); j++) {
        if (i == j) {
            continue;
        }

        if (this->unscentedKalmanTrackers[i].getLifetime() >= this->lifetimeSuppressionThreshold) {
            continue;
        }

        if (this->unscentedKalmanTrackers[j].getLifetime() <
                this->ageSuppressionThreshold * this->unscentedKalmanTrackers[i].getLifetime()) {
            continue;
        }

        dist = (this->unscentedKalmanTrackers[i].latestPrediction()
                        - this->unscentedKalmanTrackers[j].latestPrediction()).norm();
        dist /= frameDiagonal;

        if (dist <= this->distanceSuppressionThreshold) {
            return true;
        }
    }
    return false;
}



}

#endif // LINE_TRACKER_KALMAN_HPP
