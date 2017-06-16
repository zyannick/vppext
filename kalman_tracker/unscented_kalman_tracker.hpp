#ifndef UKALMANFILTER_HPP
#define UKALMANFILTER_HPP
#include "kalmantracker.hh"
#include <vpp/vpp.hh>

using namespace vpp;
using namespace Eigen;

unscented_kalman_tracker::unscented_kalman_tracker(vint2 startPt,
                                 float dt,
                                 int maxTrajectorySize) {




        // Seed the random number generator and pick a random ID and random color.
        srand(time(NULL) + startPt[0] + startPt[1]);
        this->id = rand();
        this->color = vuchar3(rand() % 256, rand() % 256, rand() % 256);

        this->max_trajectory_size = maxTrajectorySize;
        this->trajectory = std::make_shared<std::vector<vpp::vint2>>();
        this->numberOfFramesWithoutUpdate = 0;
        this->prediction = startPt;
        this->lifetime = 0;
        this->ukf = std::make_unique<Unscented_Kalman_Filter>();
        this->dt = dt;
        this->ukf->ProcessMeasurement(startPt,dt);
    }

    vint2 unscented_kalman_tracker::correct(vint2 pt) {
        vint2 estimated;
        estimated[0] = this->ukf->state_vector(0);
        estimated[1] = this->ukf->state_vector(1);
        return estimated;
    }

    vint2 unscented_kalman_tracker::predict() {
        this->ukf->Prediction(dt);
        vint2 predictedPt(this->ukf->state_vector(0), this->ukf->state_vector(1));
        /*this->kf->statePre.copyTo(this->kf->statePost);
        this->kf->errorCovPre.copyTo(this->kf->errorCovPost);*/
        this->addPointToTrajectory(predictedPt);
        this->prediction = predictedPt;
        return predictedPt;
        //return vint2(0,0);
    }

    vint2 unscented_kalman_tracker::latestPrediction() {
        return this->prediction;
    }

    void unscented_kalman_tracker::addPointToTrajectory(vint2 pt) {
        if (this->trajectory->size() >= this->max_trajectory_size) {
            this->trajectory->erase(this->trajectory->begin(),
                                    this->trajectory->begin()+1);
        }
        this->trajectory->push_back(pt);
    }


    long unscented_kalman_tracker::getLifetime() {
        return this->lifetime;
    }

    void unscented_kalman_tracker::noUpdateThisFrame() {
        this->lifetime++;
        this->numberOfFramesWithoutUpdate++;
    }

    const int unscented_kalman_tracker::getNumFramesWithoutUpdate() {
        return this->numberOfFramesWithoutUpdate;
    }

    void unscented_kalman_tracker::gotUpdate() {
        this->lifetime++;
        this->numberOfFramesWithoutUpdate = 0;
    }

    TrackingOutput unscented_kalman_tracker::latestTrackingOutput() {
        auto output = TrackingOutput{
            this->id,
            this->latestPrediction(),
            this->color,
            std::vector<vint2>()
        };
        std::copy(this->trajectory->cbegin(),
                  this->trajectory->cend(),
                  std::back_inserter(output.trajectory));
        return output;
    }

#endif // UKALMANFILTER_HPP
