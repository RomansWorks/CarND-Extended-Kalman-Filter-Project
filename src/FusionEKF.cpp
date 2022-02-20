#include "FusionEKF.h"
#include "Eigen/Dense"
#include "tools.h"
#include <iostream>

#include <stdexcept>

#undef eigen_assert
#define eigen_assert(x)                                                        \
  if (!(x)) {                                                                  \
    throw(std::runtime_error("Put your message here"));                        \
  }

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices

  // Measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0, 0, 0.0225;

  // Measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0, 0, 0.0009, 0, 0, 0, 0.09;

  // H_laser_ - translates the from state space to (laser) measurement space
  // (which contains only relative position, not speed)
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0, 0, 1, 0, 0;

  // Hj is just the placeholder. The linear approximation of the measurement
  // function depends on current state, so we calculate the jacobian later,
  // ad-hoc.
  Hj_ = MatrixXd(3, 4);

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */

  // TODO: Need to initialize H_laser_ and Hj, P. F, Q is calculated up ad-hoc
  // but we provide initial values here, and R is given in advance. No B control
  // matrix in this case.

  // Initial state and process covariance
  VectorXd x = VectorXd(4);
  x << 0, 0, 0, 0;
  MatrixXd P = MatrixXd(4, 4);
  P << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1000;
  // F is the prediction transition function
  MatrixXd F = MatrixXd(4, 4);
  F << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;
  Q_nu = MatrixXd(2, 2);
  Q_nu << noise_ax * noise_ax, 0, 0, noise_ay * noise_ay;
  MatrixXd Q = MatrixXd(4, 4);
  Q << (double)1 / 4 * noise_ax, 0, (double)1 / 2 * noise_ax, 0, 0,
      (double)1 / 4 * noise_ay, 0, (double)1 / 2 * noise_ay,
      (double)1 / 2 * noise_ax, 0, 1 * noise_ax, 0, 0, (double)1 / 2 * noise_ay,
      0, 1 * noise_ay;
  ekf_.Init(x, P, F, H_laser_, R_laser_, Q);
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates
      //         and initialize state.
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      double rho_dot = measurement_pack.raw_measurements_[2];
      double pos_x = rho * cos(phi);
      double pos_y = rho * sin(phi);
      double v_x =
          rho_dot *
          cos(pos_x); // TODO: Need to adjust to time? Is this even correct???
                      // (notes mention that it might not be)
      double v_y = rho_dot * cos(pos_y);
      ekf_.x_ << pos_x, pos_y, v_x,
          v_y; // TODO: Mutable? Or need to init a new VectorXd(4)?

      /*
        Calculating y = z - H * x'
        For lidar measurements, the error equation is y = z - H * x'. For radar
        measurements, the functions that map the x vector [px, py, vx, vy] to polar
        coordinates are non-linear. Instead of using H to calculate y = z - H * x', for
        radar measurements you'll have to use the equations that map from cartesian to
        polar coordinates: y = z - h(x'). Normalizing Angles In C++, atan2() returns
        values between -pi and pi. When calculating phi in y = z - h(x) for radar
        measurements, the resulting angle phi in the y vector should be adjusted so that
        it is between -pi and pi. The Kalman filter is expecting small angle values
        between the range -pi and pi. HINT: when working in radians, you can add 2\pi2π
        or subtract 2\pi2π until the angle is within the desired range.
      */
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // TODO: Initialize state.
      double pos_x = measurement_pack.raw_measurements_[0];
      double pos_y = measurement_pack.raw_measurements_[1];
      ekf_.x_ << pos_x, pos_y, 0,
          0; // TODO: Mutable? Or need to init a new VectorXd(4)?
    }

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */


  const double dt = (double) (measurement_pack.timestamp_ - previous_timestamp_) / 1000000;
  previous_timestamp_ = measurement_pack.timestamp_;


  // Update the prediction matrix
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // set the process covariance matrix
  const double dt_2 = pow(dt, 2);
  const double dt_3 = pow(dt, 3);
  const double dt_4 = pow(dt, 4);
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0, 0,
      dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay, dt_3 / 2 * noise_ax, 0,
      dt_2 * noise_ax, 0, 0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

  ekf_.Predict();

  /**
   * Update step
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateNonlinear(measurement_pack.raw_measurements_, radar_h);
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.UpdateLinear(measurement_pack.raw_measurements_);
  }

}

VectorXd radar_h(VectorXd x) {
  // Input - predicted state vector with cartesian space (px, py, vx, vy)
  // Output - predicted radar measurement vector in polar coordinates (rho, phi, rho_dot)
  const double px = x[0];
  const double py = x[1];
  const double vx = x[2];
  const double vy = x[3];

  const double rho = sqrt(px * px + py * py);
  double phi = atan2(py, px);

  const double rho_dot = (px * vx + py * vy) / rho;

  VectorXd res = VectorXd(3);
  res << rho, phi, rho_dot;
  return res;
}