#include "multipointtracker.hh"
#include "hungarian.hh"

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
    this->kalmanTrackers = std::vector<basic_kalman_tracker>();
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

void MultiPointTracker::update(const std::vector<vfloat2>& massCenters,
                                /*const std::vector<cv::Rect>& boundingRects,*/
                                std::vector<TrackingOutput>& trackingOutputs) {
    trackingOutputs.clear();

    // If we haven't found any mass centers, just update all the Kalman filters and return their predictions.
    if (massCenters.empty()) {
        for (int i = 0; i < this->kalmanTrackers.size(); i++) {
            // Indicate that the tracker didn't get an update this frame.
            this->kalmanTrackers[i].noUpdateThisFrame();

            // Remove the tracker if it is dead.
            if (this->kalmanTrackers[i].getNumFramesWithoutUpdate() > this->missedFramesThreshold) {
                this->kalmanTrackers.erase(this->kalmanTrackers.begin() + i);
                i--;
            }
        }
        // Update the remaining trackers.
        for (size_t i = 0; i < this->kalmanTrackers.size(); i++) {
            if (this->kalmanTrackers[i].getLifetime() > lifetimeThreshold) {
                this->kalmanTrackers[i].predict();
                trackingOutputs.push_back(this->kalmanTrackers[i].latestTrackingOutput());
            }
        }
        return;
    }

    // If there are no Kalman trackers, make one for each detection.
    if (this->kalmanTrackers.empty()) {
        for (auto massCenter : massCenters) {
            this->kalmanTrackers.push_back(basic_kalman_tracker(vint2(massCenter[0],massCenter[1]),
                                                             this->dt,
                                                             this->magnitudeOfAccelerationNoise));
        }
    }


    // Create our cost matrix.
    size_t numKalmans = this->kalmanTrackers.size();
    size_t numCenters = massCenters.size();

    std::vector<std::vector<double>> costMatrix(numKalmans, std::vector<double>(numCenters));

    std::vector<int> assignment;


    // Get the latest prediction for the Kalman filters.
    std::vector<vfloat2> predictions(this->kalmanTrackers.size());
    for (size_t i = 0; i < this->kalmanTrackers.size(); i++) {
        vint2 tv = this->kalmanTrackers[i].latestPrediction();
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
            this->kalmanTrackers[i].noUpdateThisFrame();
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
    for (int i = 0; i < this->kalmanTrackers.size(); i++) {
        if (this->kalmanTrackers[i].getNumFramesWithoutUpdate() > this->missedFramesThreshold) {
            this->kalmanTrackers.erase(this->kalmanTrackers.begin() + i);
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
        this->kalmanTrackers.push_back(basic_kalman_tracker(vint2(to[0],to[1])));
    }

    // Update the Kalman filters.
    for (size_t i = 0; i < assignment.size(); i++) {
        this->kalmanTrackers[i].predict();
        if (assignment[i] != -1) {
            vfloat2 tg = massCenters[assignment[i]];
            vint2 val(2);
            val << tg[0] , tg[1];
            this->kalmanTrackers[i].correct(val);
            this->kalmanTrackers[i].gotUpdate();
        }
    }

    // Remove any suppressed filters.
    for (size_t i = 0; i < this->kalmanTrackers.size(); i++) {
        if (this->hasSuppressor(i)) {
            this->kalmanTrackers.erase(this->kalmanTrackers.begin() + i);
            i--;
        }
    }

    // Now update the predictions.
    for (size_t i = 0; i < this->kalmanTrackers.size(); i++) {
        if (this->kalmanTrackers[i].getLifetime() > this->lifetimeThreshold) {
            trackingOutputs.push_back(this->kalmanTrackers[i].latestTrackingOutput());
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



bool MultiPointTracker::hasSuppressor(size_t i) {
    double dist;
    vfloat2 framePoint = vfloat2(this->frameSize[0], this->frameSize[1]);
    double frameDiagonal = std::sqrt(framePoint.dot(framePoint));

    for (size_t j = 0; j < this->kalmanTrackers.size(); j++) {
        if (i == j) {
            continue;
        }

        if (this->kalmanTrackers[i].getLifetime() >= this->lifetimeSuppressionThreshold) {
            continue;
        }

        if (this->kalmanTrackers[j].getLifetime() <
                this->ageSuppressionThreshold * this->kalmanTrackers[i].getLifetime()) {
            continue;
        }

        dist = (this->kalmanTrackers[i].latestPrediction()
                        - this->kalmanTrackers[j].latestPrediction()).norm();
        dist /= frameDiagonal;

        if (dist <= this->distanceSuppressionThreshold) {
            return true;
        }
    }
    return false;
}


