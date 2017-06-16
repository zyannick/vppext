#include "multipointtracker.hh"
#include "hungarian_method/hungarian.hh"

MultiPointTracker::MultiPointTracker()
{

}

MultiPointTracker::MultiPointTracker (vint2 frameSize,
                                      long lifetimeThreshold,
                                      float distanceThreshold,
                                      long missedFramesThreshold,
                                      float dt,
                                      float magnitudeOfAccelerationNoise,
                                      int lifetimeSuppressionThreshold,
                                      float distanceSuppressionThreshold,
                                      float ageSuppressionThreshold) {
    this->basicKalmanTrackers = std::vector<basic_kalman_tracker>();
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

MultiPointTracker::MultiPointTracker(vint2 frameSize,
                  long lifetimeThreshold,
                  float distanceThreshold ,
                  long missedFramesThreshold ,
                  float magnitudeOfAccelerationNoise  ,
                  int lifetimeSuppressionThreshold ,
                  float distanceSuppressionThreshold ,
                  float ageSuppressionThreshold )
{
    this->unscentedKalmanTrackers = std::vector<unscented_kalman_tracker>();
    this->frameSize = frameSize;
    this->lifetimeThreshold = lifetimeThreshold;
    this->distanceThreshold = distanceThreshold;
    this->missedFramesThreshold = missedFramesThreshold;
    this->magnitudeOfAccelerationNoise = magnitudeOfAccelerationNoise;
    this->lifetimeSuppressionThreshold = lifetimeSuppressionThreshold;
    this->distanceSuppressionThreshold = distanceSuppressionThreshold;
    this->ageSuppressionThreshold = ageSuppressionThreshold;
    this->dt = 0.2;
}

void MultiPointTracker::updateBasicKalmanTrackers(const std::vector<vfloat2>& massCenters,
                                /*const std::vector<cv::Rect>& boundingRects,*/
                                std::vector<TrackingOutput>& trackingOutputs) {
    trackingOutputs.clear();

    // If we haven't found any mass centers, just update all the Kalman filters and return their predictions.
    if (massCenters.empty()) {
        for (int i = 0; i < this->basicKalmanTrackers.size(); i++) {
            // Indicate that the tracker didn't get an update this frame.
            this->basicKalmanTrackers[i].noUpdateThisFrame();

            // Remove the tracker if it is dead.
            if (this->basicKalmanTrackers[i].getNumFramesWithoutUpdate() > this->missedFramesThreshold) {
                this->basicKalmanTrackers.erase(this->basicKalmanTrackers.begin() + i);
                i--;
            }
        }
        // Update the remaining trackers.
        for (size_t i = 0; i < this->basicKalmanTrackers.size(); i++) {
            if (this->basicKalmanTrackers[i].getLifetime() > lifetimeThreshold) {
                this->basicKalmanTrackers[i].predict();
                trackingOutputs.push_back(this->basicKalmanTrackers[i].latestTrackingOutput());
            }
        }
        return;
    }

    // If there are no Kalman trackers, make one for each detection.
    if (this->basicKalmanTrackers.empty()) {
        for (auto massCenter : massCenters) {
            this->basicKalmanTrackers.push_back(basic_kalman_tracker(vint2(massCenter[0],massCenter[1]),
                                                             this->dt,
                                                             this->magnitudeOfAccelerationNoise));
        }
    }


    // Create our cost matrix.
    size_t numKalmans = this->basicKalmanTrackers.size();
    size_t numCenters = massCenters.size();

    std::vector<std::vector<double>> costMatrix(numKalmans, std::vector<double>(numCenters));

    std::vector<int> assignment;


    // Get the latest prediction for the Kalman filters.
    std::vector<vfloat2> predictions(this->basicKalmanTrackers.size());
    for (size_t i = 0; i < this->basicKalmanTrackers.size(); i++) {
        vint2 tv = this->basicKalmanTrackers[i].latestPrediction();
        predictions[i] = vfloat2(tv[0],tv[1]);
    }

    // We need to associate each of the mass centers to their corresponding Kalman filter. First,
    // let's find the pairwise distances. However, we first divide this distance by the diagonal size
    // of the frame to ensure that it is between 0 and 1.
    vint2 framePoint = vint2(this->frameSize[0], this->frameSize[1]);
    double frameDiagonal = std::sqrt(framePoint.dot(framePoint));
    for (size_t i = 0; i < predictions.size(); i++) {
        for (size_t j = 0; j < massCenters.size(); j++) {
            costMatrix[i][j] = (predictions[i] - massCenters[j]).norm() / frameDiagonal;
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
            this->basicKalmanTrackers[i].noUpdateThisFrame();
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
    for (int i = 0; i < this->basicKalmanTrackers.size(); i++) {
        if (this->basicKalmanTrackers[i].getNumFramesWithoutUpdate() > this->missedFramesThreshold) {
            this->basicKalmanTrackers.erase(this->basicKalmanTrackers.begin() + i);
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
        this->basicKalmanTrackers.push_back(basic_kalman_tracker(vint2(to[0],to[1])));
    }

    // Update the Kalman filters.
    for (size_t i = 0; i < assignment.size(); i++) {
        this->basicKalmanTrackers[i].predict();
        if (assignment[i] != -1) {
            vfloat2 tg = massCenters[assignment[i]];
            vint2 val(2);
            val << tg[0] , tg[1];
            this->basicKalmanTrackers[i].correct(val);
            this->basicKalmanTrackers[i].gotUpdate();
        }
    }

    // Remove any suppressed filters.
    for (size_t i = 0; i < this->basicKalmanTrackers.size(); i++) {
        if (this->hasSuppressorBasic(i)) {
            this->basicKalmanTrackers.erase(this->basicKalmanTrackers.begin() + i);
            i--;
        }
    }

    // Now update the predictions.
    for (size_t i = 0; i < this->basicKalmanTrackers.size(); i++) {
        if (this->basicKalmanTrackers[i].getLifetime() > this->lifetimeThreshold) {
            trackingOutputs.push_back(this->basicKalmanTrackers[i].latestTrackingOutput());
        }
    }
}


void MultiPointTracker::updateUnscentedKalmanTrackers(const std::vector<vfloat2>& massCenters,
                                /*const std::vector<cv::Rect>& boundingRects,*/
                                std::vector<TrackingOutput>& trackingOutputs,int max_trajectory_size) {
    trackingOutputs.clear();

    float dt = 0.33;

    // If we haven't found any mass centers, just update all the Kalman filters and return their predictions.
    if (massCenters.empty()) {
        for (int i = 0; i < this->unscentedKalmanTrackers.size(); i++) {
            // Indicate that the tracker didn't get an update this frame.
            //this->unscentedKalmanTrackers[i].neFaitrien();
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
                trackingOutputs.push_back(this->unscentedKalmanTrackers[i].latestTrackingOutput());
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
    double frameDiagonal = std::sqrt(framePoint.dot(framePoint));
    for (size_t i = 0; i < predictions.size(); i++) {
        for (size_t j = 0; j < massCenters.size(); j++) {
            costMatrix[i][j] = (predictions[i] - massCenters[j]).norm() / frameDiagonal;
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
    for (size_t i = 0; i < this->unscentedKalmanTrackers.size(); i++) {
        if (this->unscentedKalmanTrackers[i].getLifetime() > this->lifetimeThreshold) {
            trackingOutputs.push_back(this->unscentedKalmanTrackers[i].latestTrackingOutput());
        }
    }
}

/*
bool MultiPointTracker::sharesBoundingRect(size_t i, cv::Rect boundingRect) {
    for (size_t j = 0; j < this->kalmanTrackers.size(); j++) {
        if (i == j) {
            continue;
        }
        if (boundingRect.contains(this->kalmanTrackers[i].latestPrediction())) {
            return true;
        }
    }
    return false;
}*/



bool MultiPointTracker::hasSuppressorBasic(size_t i) {
    double dist;
    vfloat2 framePoint = vfloat2(this->frameSize[0], this->frameSize[1]);
    double frameDiagonal = std::sqrt(framePoint.dot(framePoint));

    for (size_t j = 0; j < this->basicKalmanTrackers.size(); j++) {
        if (i == j) {
            continue;
        }

        if (this->basicKalmanTrackers[i].getLifetime() >= this->lifetimeSuppressionThreshold) {
            continue;
        }

        if (this->basicKalmanTrackers[j].getLifetime() <
                this->ageSuppressionThreshold * this->basicKalmanTrackers[i].getLifetime()) {
            continue;
        }

        dist = (this->basicKalmanTrackers[i].latestPrediction()
                        - this->basicKalmanTrackers[j].latestPrediction()).norm();
        dist /= frameDiagonal;

        if (dist <= this->distanceSuppressionThreshold) {
            return true;
        }
    }
    return false;
}

bool MultiPointTracker::hasSuppressorUnscented(size_t i) {
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


