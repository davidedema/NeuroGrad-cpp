#include <catch2/catch_test_macros.hpp>

#include "nn/Layer.hh"

using autodiff::Scalar;
using autodiff::nn::Layer;

namespace {
Scalar identity(const Scalar& x) { return x; }
}  // namespace

// Layer.hh currently only fixes the interface (forward() returns a
// placeholder zero vector) — these tests are expected to fail ("red")
// until Stage 5 is implemented, same pattern Dual.hh used for Stage 1.
//
//   W = [ 2.0  -1.0 ]   x = [ x0 ]   b = [ 1.0 ]
//       [ 0.5   3.0 ]       [ x1 ]       [ -2.0 ]
//
//   y = identity(W*x + b):
//     y0 = W00*x0 + W01*x1 + b0  =>  dy0/dx0 = W00
//     y1 = W10*x0 + W11*x1 + b1  =>  dy1/dx0 = W10

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
