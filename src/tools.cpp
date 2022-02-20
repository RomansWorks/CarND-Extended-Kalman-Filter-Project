#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * DONE: Calculate the RMSE here.
   */
  int n = (int)estimations.size();
  VectorXd sumOfSquaredResiduals(4);
  sumOfSquaredResiduals << 0, 0, 0, 0;

  for (int i = 0; i < n; i++) {
    const VectorXd est = estimations[i];
    const VectorXd tru = ground_truth[i];
    const VectorXd res = est - tru;
    assert(res.cols() == 1);
    sumOfSquaredResiduals[0] += res[0] * res[0];
    sumOfSquaredResiduals[1] += res[1] * res[1];
    sumOfSquaredResiduals[2] += res[2] * res[2];
    sumOfSquaredResiduals[3] += res[3] * res[3];
  }

  const VectorXd rmse = (sumOfSquaredResiduals / n).cwiseSqrt();
  //  if (rmse[0] > 0.11 || rmse[1] > 0.11) {
  //    std::cout << std::endl << "Warning - RMSE:" << std::endl <<
  //    (sumOfSquaredResiduals / n).cwiseSqrt() << std::endl;
  //  }

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  /**
   * DONE: Calculate a Jacobian here.
   */
  MatrixXd Hj(3, 4);
  // recover state parameters
  const double px = x_state(0);
  const double py = x_state(1);
  const double vx = x_state(2);
  const double vy = x_state(3);

  // pre-compute a set of terms to avoid repeated calculation
  const double p_dist_sq = px * px + py * py;
  const double p_dist = sqrt(p_dist_sq);
  const double p_dist_cub = (p_dist_sq * p_dist);
  const double px_div_p_dist = px / p_dist;
  const double py_div_p_dist = py / p_dist;

  // check division by zero
  if (fabs(p_dist_sq) < 0.0001) {
    std::cerr << "CalculateJacobian () - Warning - Avoided division by Zero"
              << std::endl;
    return Hj;
  }

  // compute the Jacobian matrix
  Hj << px_div_p_dist, py_div_p_dist, 0, 0, -(py / p_dist_sq), (px / p_dist_sq),
      0, 0, py * (vx * py - vy * px) / p_dist_cub,
      px * (px * vy - py * vx) / p_dist_cub, px_div_p_dist, py_div_p_dist;

  return Hj;
}
