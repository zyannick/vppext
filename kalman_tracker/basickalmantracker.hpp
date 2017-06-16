#ifndef MULTIKALMANFILTER_HPP
#define MULTIKALMANFILTER_HPP
#include "kalmantracker.hh"
#include <vpp/vpp.hh>

using namespace vpp;
using namespace Eigen;

basic_kalman_tracker::basic_kalman_tracker(vint2 startPt,
                                 float dt,
                                 float magnitudeOfAccelerationNoise,
                                 size_t maxTrajectorySize) {

        // Seed the random number generator and pick a random ID and random color.
        srand(time(NULL) + startPt[0] + startPt[1]);
        this->id = rand();
        this->color = vuchar3(rand() % 256, rand() % 256, rand() % 256);

        this->max_trajectory_size = maxTrajectorySize;
        this->trajectory = std::make_shared<std::vector<vpp::vint2>>();
        this->numberOfFramesWithoutUpdate = 0;
        this->prediction = startPt;
        this->lifetime = 0;

        int n = 4; // Number of states
        int m = 2; // Number of measurements

        //double dt = dt; // Time step

        Eigen::MatrixXd A(n, n); // System dynamics matrix
        Eigen::MatrixXd C(m, n); // Output matrix
        Eigen::MatrixXd Q(n, n); // Process noise covariance
        Eigen::MatrixXd R(m, m); // Measurement noise covariance
        Eigen::MatrixXd P(n, n); // Estimate error covariance

        // Discrete LTI projectile motion, measuring position only
        A << 1,0,dt,0,   0,1,0,dt,  0,0,1,0,  0,0,0,1;
        C.setIdentity();

        // Reasonable covariance matrices
        Q << pow(dt,4.0)/4.0, 0, pow(dt,3.0)/2.0, 0,
             0, pow(dt,4.0)/4.0 , 0 ,pow(dt,3.0)/2.0,
             pow(dt,3.0)/2.0, 0, pow(dt,2.0), 0,
             0, pow(dt,3.0)/2.0, 0, pow(dt,2.0);
        Q *= magnitudeOfAccelerationNoise;
        R << 5 , 5 , 5 ,5;
        P << .1, .1, .1, .1,
                .1, .1, .1, .1,
                .1, .1, .1, .1,
                .1, .1, .1, .1;

        this->bkf = std::make_unique<BasicKalmanFilter>(dt,A,C,Q,R,P);
        // Initialize filter with 4 dynamic parameters (x, y, x velocity, y
        // velocity), 2 measurement parameters (x, y), and no control parameters.
        Eigen::VectorXd measure(2);
        measure << startPt[0],startPt[1];
        this->bkf->init(0.0,measure);


    }

    vint2 basic_kalman_tracker::correct(vint2 pt) {
        Eigen::VectorXd measurement(2);
        measurement[0] = pt[0];
        measurement[1] = pt[1];
        vint2 estimated = this->bkf->correct(measurement);
        this->prediction[0] = estimated[0];
        this->prediction[1] = estimated[1];
        return estimated;
    }

    vint2 basic_kalman_tracker::predict() {
        vint2 prediction = this->bkf->predict();
        vint2 predictedPt(prediction[0], prediction[1]);
        /*this->kf->statePre.copyTo(this->kf->statePost);
        this->kf->errorCovPre.copyTo(this->kf->errorCovPost);*/
        this->addPointToTrajectory(predictedPt);
        this->prediction = predictedPt;
        return predictedPt;
        //return vint2(0,0);
    }

    vint2 basic_kalman_tracker::latestPrediction() {
        return this->prediction;
    }

    void basic_kalman_tracker::addPointToTrajectory(vint2 pt) {
        if (this->trajectory->size() >= this->max_trajectory_size) {
            this->trajectory->erase(this->trajectory->begin(),
                                    this->trajectory->begin()+1);
        }
        this->trajectory->push_back(pt);
    }


    long basic_kalman_tracker::getLifetime() {
        return this->lifetime;
    }

    void basic_kalman_tracker::noUpdateThisFrame() {
        this->lifetime++;
        this->numberOfFramesWithoutUpdate++;
    }

    const int basic_kalman_tracker::getNumFramesWithoutUpdate() {
        return this->numberOfFramesWithoutUpdate;
    }

    void basic_kalman_tracker::gotUpdate() {
        this->lifetime++;
        this->numberOfFramesWithoutUpdate = 0;
    }

    TrackingOutput basic_kalman_tracker::latestTrackingOutput() {
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

#endif // MULTIKALMANFILTER_HPP
