#ifndef UKF_HH
#define UKF_HH

#include "algorithms/measurement.hh"
#include "eigen3/Eigen/Dense"
#include <vpp/vpp.hh>
#include <vector>
#include <string>
#include <fstream>
#include "miscellanous/tools.hh"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace vpp;

class Unscented_Kalman_Filter {
public:
  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* State dimension
  int state_dimension;

  ///* Augmented state dimension
  int augmented_state_dimension;

 //* radar measurement dimension
  int n_zlas_;

 //* radar measurement dimension
  int n_zrad_;

  ///* Sigma point spreading parameter
  double sigma_point_spreading_parameter;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd state_vector;

  ///* state covariance matrix P_
  MatrixXd covariance_matrix;

  ///* predicted sigma points matrix
  MatrixXd predicted_sigmas_matrix;

  ///* time when the state is true, in us
  long long previous_timestamp_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

  /**
   * Constructor
   */
  Unscented_Kalman_Filter();

  /**
   * Destructor
   */
  virtual ~Unscented_Kalman_Filter();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(const MeasurementPackage meas_package);
  void ProcessMeasurement(const vint2 values, float dt);

  /**
   * SetInitialValues
   * @param meas_package Use the first measurement to initialize the filter
   */
  void SetInitialValues(const MeasurementPackage meas_package);
  void SetInitialValues(const vint2 values, float dt);

  /**
   * Computes the AugmentedSigmaPoints
   * @param Xsig_out reference to augmented sigma point matrix
   */
  void AugmentedSigmaPoints(MatrixXd &Xsig_out);

  /**
  * Predicts the augmented sigma points
  * @param Xsig_out reference to predicted augmented sigma point matrix
  */
  void SigmaPointPrediction(const MatrixXd  &Xsig_aug, const double delta_t, MatrixXd  &Xsig_out);

  /**
  * Predicts the mean and covariance
  * @param predicted state vector x_out covariance matrix P_out
  */
  void PredictMeanAndCovariance(const MatrixXd &Xsig_pred, VectorXd &x_out, MatrixXd &P_out);


  /**
  * Computes the predicted state in radar measurement space z_pred, the predicted covariance S and Tc
  */
  void PredictRadarMeasurement(VectorXd &z_out, MatrixXd &S_out, MatrixXd &Tc_out);

  /**
  * Computes the predicted state in lidar measurement space z_pred, the predicted covariance S and Tc
  */
  void PredictLidarMeasurement(VectorXd &z_out, MatrixXd &S_out, MatrixXd &Tc_out);
  void PredictMeasurement(VectorXd &z_out, MatrixXd &S_out, MatrixXd &Tc_out);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package, VectorXd &z_pred, MatrixXd &Tc, MatrixXd &S);
  void Update(vint2 values,float dt, VectorXd &z_pred, MatrixXd &Tc, MatrixXd &S);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package, VectorXd &z_pred, MatrixXd &Tc, MatrixXd &S);
};

#include "unscented_kalman_filter.hpp"

#endif // UKF_HH
