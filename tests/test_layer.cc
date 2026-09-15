#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "autodiff/Dual.hh"
#include "nn/Layer.hh"
#include "utils/finite_diff.hh"

using autodiff::Scalar;
using autodiff::nn::Layer;

namespace {
Scalar identity(const Scalar& x) { return x; }
Scalar relu_activation(const Scalar& x) { return autodiff::forward::relu(x); }
Scalar sigmoid_activation(const Scalar& x) {
  return autodiff::forward::sigmoid(x);
}
}  // namespace

// Shared layer for the tests below:
//
//   W = [ 2.0  -1.0 ]   x = [ x0 ]   b = [ 1.0 ]
//       [ 0.5   3.0 ]       [ x1 ]       [ -2.0 ]
//
//   pre-activation = W*x + b:
//     y0_pre = W00*x0 + W01*x1 + b0  =>  dy0_pre/dx0 = W00
//     y1_pre = W10*x0 + W11*x1 + b1  =>  dy1_pre/dx0 = W10

TEST_CASE("Layer: forward computes activation(W*x + b)", "[Layer]") {
  Layer<Scalar>::Matrix W(2, 2);
  W << Scalar(2.0), Scalar(-1.0), Scalar(0.5), Scalar(3.0);
  Layer<Scalar>::Vector b(2);
  b << Scalar(1.0), Scalar(-2.0);

  Layer<Scalar> layer(W, b, identity);

  Layer<Scalar>::Vector x(2);
  x << Scalar(1.5), Scalar(-2.0);

  Layer<Scalar>::Vector y = layer.forward(x);

  REQUIRE(y(0).value() == 6.0);     // 2*1.5 + (-1)*(-2) + 1
  REQUIRE(y(1).value() == -7.25);   // 0.5*1.5 + 3*(-2) + (-2)
}

TEST_CASE("Layer: forward propagates the derivative through W*x + b", "[Layer]") {
  Layer<Scalar>::Matrix W(2, 2);
  W << Scalar(2.0), Scalar(-1.0), Scalar(0.5), Scalar(3.0);
  Layer<Scalar>::Vector b(2);
  b << Scalar(1.0), Scalar(-2.0);

  Layer<Scalar> layer(W, b, identity);

  Layer<Scalar>::Vector x(2);
  x << Scalar::variable(1.5), Scalar::constant(-2.0);  // differentiate w.r.t. x0

  Layer<Scalar>::Vector y = layer.forward(x);

  REQUIRE(y(0).derivative() == 2.0);  // dy0/dx0 = W00
  REQUIRE(y(1).derivative() == 0.5);  // dy1/dx0 = W10
}

// Same W/b/x as above: pre-activation is [6.0, -7.25] (one positive, one
// negative row), which exercises both branches of relu's derivative rule
// in a single case — the point of this test is that behavior, not the
// affine part already covered above.
TEST_CASE("Layer: forward with relu activation zeroes negative rows and their derivative",
          "[Layer]") {
  Layer<Scalar>::Matrix W(2, 2);
  W << Scalar(2.0), Scalar(-1.0), Scalar(0.5), Scalar(3.0);
  Layer<Scalar>::Vector b(2);
  b << Scalar(1.0), Scalar(-2.0);

  Layer<Scalar> layer(W, b, relu_activation);

  Layer<Scalar>::Vector x(2);
  x << Scalar::variable(1.5), Scalar::constant(-2.0);  // differentiate w.r.t. x0

  Layer<Scalar>::Vector y = layer.forward(x);

  // Row 0: pre-activation 6.0 > 0, relu passes value and derivative through.
  REQUIRE(y(0).value() == 6.0);
  REQUIRE(y(0).derivative() == 2.0);

  // Row 1: pre-activation -7.25 < 0, relu zeroes both value and derivative,
  // regardless of the (nonzero) incoming derivative 0.5.
  REQUIRE(y(1).value() == 0.0);
  REQUIRE(y(1).derivative() == 0.0);
}

// Sigmoid's derivative isn't hand-checkable the way identity/relu are, so
// this follows the Dual.hh convention instead: compare the layer's
// Dual-computed derivative against a central finite difference of the same
// composite function (affine layer + sigmoid) evaluated on plain doubles.
TEST_CASE("Layer: forward with sigmoid activation matches finite-difference derivative",
          "[Layer][finite-diff]") {
  Layer<Scalar>::Matrix W(2, 2);
  W << Scalar(2.0), Scalar(-1.0), Scalar(0.5), Scalar(3.0);
  Layer<Scalar>::Vector b(2);
  b << Scalar(1.0), Scalar(-2.0);

  Layer<Scalar> layer(W, b, sigmoid_activation);

  // Reference: row 0 of activation(W*x + b) as a function of x0 alone,
  // with x1 fixed at -2.0 (matching the Dual-side x below).
  auto f0 = [](double x0) {
    const double pre = 2.0 * x0 + (-1.0) * (-2.0) + 1.0;
    return 1.0 / (1.0 + std::exp(-pre));
  };

  for (double x0 : {-3.0, -1.0, 0.0, 1.5, 3.0}) {
    Layer<Scalar>::Vector x(2);
    x << Scalar::variable(x0), Scalar::constant(-2.0);

    Layer<Scalar>::Vector y = layer.forward(x);

    double numerical = testutil::central_difference(f0, x0);
    INFO("x0 = " << x0);
    REQUIRE(testutil::is_close(y(0).derivative(), numerical));
  }
}
