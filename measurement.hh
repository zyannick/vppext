#ifndef MEASUREMENT_HH
#define MEASUREMENT_HH

#include "eigen3/Eigen/Dense"

class MeasurementPackage {
public:
  long long timestamp_;

  enum SensorType{
    LASER,
    RADAR
  } sensor_type_;

  Eigen::VectorXd raw_measurements_;

};

#endif // MEASUREMENT_HH
