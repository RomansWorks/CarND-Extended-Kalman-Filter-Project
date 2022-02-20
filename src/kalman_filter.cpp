#include "kalman_filter.h"
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
  I_ = MatrixXd::Identity(x_.size(), x_.size());
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::UpdateLinear(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  while (y[1] < -M_PI) {
    y[1] += 2 * M_PI;
  }
  while (y[1] > M_PI) {
    y[1] -= 2 * M_PI;
  }
  const MatrixXd Ht = H_.transpose();
  const MatrixXd S = H_ * P_ * Ht + R_;
  const MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + K * y;
  P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::UpdateNonlinear(const VectorXd &z, VectorXd (*h)(VectorXd)) {
  VectorXd y = z - h(x_);
  while (y[1] < -M_PI) {
    y[1] += 2 * M_PI;
  }
  while (y[1] > M_PI) {
    y[1] -= 2 * M_PI;
  }
  const MatrixXd Ht = H_.transpose();
  const MatrixXd S = H_ * P_ * Ht + R_;
  const MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + K * y;
  P_ = (I_ - K * H_) * P_;
}
