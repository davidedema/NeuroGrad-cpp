#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <stdexcept>

#include "nn/Loss.hh"
#include "utils/finite_diff.hh"

using autodiff::Scalar;
using autodiff::nn::bce;
using autodiff::nn::mse;

TEST_CASE("mse: matches hand-computed value and derivative", "[Loss]") {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> predicted(2);
  predicted << Scalar::variable(2.0), Scalar::constant(4.0);  // d/d predicted(0)

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> target(2);
  target << Scalar::constant(0.0), Scalar::constant(0.0);

  // mse = (2^2 + 4^2) / 2 = 10; d(mse)/d(predicted(0)) = 2*predicted(0)/2 = 2.
  Scalar loss = mse(predicted, target);
  REQUIRE(loss.value() == 10.0);
  REQUIRE(loss.derivative() == 2.0);
}

TEST_CASE("mse: throws on mismatched sizes", "[Loss]") {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> predicted(2);
  predicted << Scalar(1.0), Scalar(2.0);

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> target(3);
  target << Scalar(1.0), Scalar(2.0), Scalar(3.0);

  REQUIRE_THROWS_AS(mse(predicted, target), std::invalid_argument);
}

// predicted(0) varies while the rest of predicted and all of target stay
// fixed, mirroring the sigmoid finite-diff test in test_layer.cc: mse's
// derivative isn't as trivially hand-checkable across many points as the
// single case above, so compare against a central finite difference of
// the equivalent plain-double function instead.
TEST_CASE("mse: derivative matches finite-difference derivative", "[Loss][finite-diff]") {
  auto f = [](double p0) {
    const double d0 = p0 - 1.0;
    const double d1 = 0.5 - (-1.0);
    const double d2 = 3.0 - 2.0;
    return (d0 * d0 + d1 * d1 + d2 * d2) / 3.0;
  };

  for (double p0 : {-1.0, 0.0, 1.0, 2.5, 4.0}) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> predicted(3);
    predicted << Scalar::variable(p0), Scalar::constant(0.5), Scalar::constant(3.0);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> target(3);
    target << Scalar::constant(1.0), Scalar::constant(-1.0), Scalar::constant(2.0);

    Scalar loss = mse(predicted, target);
    double numerical = testutil::central_difference(f, p0);
    INFO("p0 = " << p0);
    REQUIRE(testutil::is_close(loss.derivative(), numerical));
  }
}

TEST_CASE("bce: matches hand-computed value", "[Loss]") {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> predicted(2);
  predicted << Scalar(0.7), Scalar(0.3);

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> target(2);
  target << Scalar(1.0), Scalar(0.0);

  // predicted(0): t=1 -> contributes log(0.7).
  // predicted(1): t=0 -> contributes log(1 - 0.3) = log(0.7).
  // bce = -(log(0.7) + log(0.7)) / 2 = -log(0.7).
  const double expected = -std::log(0.7);

  Scalar loss = bce(predicted, target);
  REQUIRE(testutil::is_close(loss.value(), expected));
}

TEST_CASE("bce: throws on mismatched sizes", "[Loss]") {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> predicted(2);
  predicted << Scalar(0.5), Scalar(0.5);

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> target(3);
  target << Scalar(1.0), Scalar(0.0), Scalar(1.0);

  REQUIRE_THROWS_AS(bce(predicted, target), std::invalid_argument);
}

// predicted(0) varies (kept well inside (0, 1), away from the boundary
// where log diverges) while the rest of predicted and all of target stay
// fixed; same finite-diff-vs-autodiff pattern as the mse test above.
TEST_CASE("bce: derivative matches finite-difference derivative", "[Loss][finite-diff]") {
  auto f = [](double p0) {
    const double term0 = 1.0 * std::log(p0) + 0.0 * std::log(1.0 - p0);
    const double term1 = 0.0 * std::log(0.6) + 1.0 * std::log(1.0 - 0.6);
    return -(term0 + term1) / 2.0;
  };

  for (double p0 : {0.2, 0.35, 0.5, 0.65, 0.8}) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> predicted(2);
    predicted << Scalar::variable(p0), Scalar::constant(0.6);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> target(2);
    target << Scalar::constant(1.0), Scalar::constant(0.0);

    Scalar loss = bce(predicted, target);
    double numerical = testutil::central_difference(f, p0);
    INFO("p0 = " << p0);
    REQUIRE(testutil::is_close(loss.derivative(), numerical));
  }
}
