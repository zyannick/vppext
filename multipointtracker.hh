#ifndef MULTIPOINTTRACKER_H
#define MULTIPOINTTRACKER_H

#include <vpp/vpp.hh>
#include "kalmantracker.hh"
using namespace vpp;
using namespace std;

class MultiPointTracker
{
private:
    // The actual object trackers.
    std::vector<basic_kalman_tracker> kalmanTrackers;

    // We only care about trackers who have been alive for the
    // given lifetimeThreshold number of frames.
    long lifetimeThreshold;

    // The size of a frame.
    vint2 frameSize;

    // We won't associate a tracker with a mass center if the distance
    // between the two is greater than this fraction of the frame dimension
    // (taken as average between width and height).
    float distanceThreshold;

    // Kill a tracker if it has gone missedFramesThreshold frames
    // without receiving a measurement.
    long missedFramesThreshold;

    // Delta time, used to set up matrices for Kalman trackers.
    float dt;

    // Magnitude of acceleration noise. Used to set up Kalman trackers.
    float magnitudeOfAccelerationNoise;

    // Check if the Kalman filter at index i has another Kalman filter that can suppress it.
    bool hasSuppressor(size_t i);

    // Any Kalman filter with a lifetime above this value cannot be suppressed.
    int lifetimeSuppressionThreshold;

    // A Kalman filter can only be suppressed by another filter which is threshold * framDiagonal
    // or closer.
    float distanceSuppressionThreshold;

    // A Kalman filter can only be suppressed by another filter which is threshold times its age.
    float ageSuppressionThreshold;

    // Check if the tracker prediction at index i shares the given
    // bounding rectangle with another point.
    //bool sharesBoundingRect(size_t i, cv::Rect boundingRect);
public:
    MultiPointTracker();
    MultiPointTracker(vint2 frameSize,
                      long lifetimeThreshold = 20,
                      float distanceThreshold = 0.1,
                      long missedFramesThreshold = 10,
                      float dt = 0.2,
                      float magnitudeOfAccelerationNoise = 0.5,
                      int lifetimeSuppressionThreshold = 20,
                      float distanceSuppressionThreshold = 0.1,
                      float ageSuppressionThreshold = 2);
    // Update the object tracker with the mass centers of the observed boundings rects.
    void update(const std::vector<vfloat2>& massCenters,
                std::vector<TrackingOutput>& trackingOutputs);
};

#include "multipointtracker.hpp"

#endif // MULTIPOINTTRACKER_H
