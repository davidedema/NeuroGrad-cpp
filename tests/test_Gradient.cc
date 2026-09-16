#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "autodiff/Dual.hh"
#include "nn/FFNetwork.hh"
#include "nn/Gradient.hh"
#include "nn/Layer.hh"
#include "nn/Loss.hh"
#include "utils/finite_diff.hh"

using autodiff::Scalar;
using autodiff::nn::FFNetwork;
using autodiff::nn::gradient;
using autodiff::nn::Layer;
using autodiff::nn::mse;

namespace {
Scalar identity(const Scalar& x) { return x; }
Scalar sigmoid_activation(const Scalar& x) { return autodiff::forward::sigmoid(x); }
}  // namespace

// Layer 1 (1 -> 2, identity): W1 = [ 2 ]   b1 = [ 0 ]
//                                  [ 3 ]        [ 0 ]
//   h = W1*x + b1 = [ 2x, 3x ]
//
// Layer 2 (2 -> 1, identity): W2 = [ 1 1 ]   b2 = [ 0 ]
//   y = h0 + h1 = 5x
//
// At x = 5: y = 25; loss = mse(y, 0) = y^2 = 625; dL/dy = 2*y = 50.
//   dL/dW1(0,0) = dL/dy * dy/dh0 * dh0/dW1(0,0) = 50 * 1 * x = 250
//   dL/dW1(1,0) = 50 * 1 * x = 250
//   dL/db1(0) = 50 * 1 * 1 = 50 ; dL/db1(1) = 50
//   dL/dW2(0,0) = dL/dy * dy/dW2(0,0) = 50 * h0 = 50*10 = 500
//   dL/dW2(0,1) = 50 * h1 = 50*15 = 750
//   dL/db2(0) = 50
TEST_CASE("gradient: matches hand-computed values for a small 2-layer network",
          "[Gradient]") {
  Layer<Scalar>::Matrix W1(2, 1);
  W1 << Scalar(2.0), Scalar(3.0);
  Layer<Scalar>::Vector b1(2);
  b1 << Scalar(0.0), Scalar(0.0);
  Layer<Scalar> layer1(W1, b1, identity);

  Layer<Scalar>::Matrix W2(1, 2);
  W2 << Scalar(1.0), Scalar(1.0);
  Layer<Scalar>::Vector b2(1);
  b2 << Scalar(0.0);
  Layer<Scalar> layer2(W2, b2, identity);

  FFNetwork<Scalar> net({layer1, layer2});

  FFNetwork<Scalar>::Vector x(1);
  x << Scalar::constant(5.0);
  FFNetwork<Scalar>::Vector target(1);
  target << Scalar::constant(0.0);

  auto grad = gradient(net, x, target);

  REQUIRE(grad[0].weights(0, 0) == 250.0);
  REQUIRE(grad[0].weights(1, 0) == 250.0);
  REQUIRE(grad[0].bias(0) == 50.0);
  REQUIRE(grad[0].bias(1) == 50.0);

  REQUIRE(grad[1].weights(0, 0) == 500.0);
  REQUIRE(grad[1].weights(0, 1) == 750.0);
  REQUIRE(grad[1].bias(0) == 50.0);
}

// gradient() calls network.forward() internally for every parameter; a
// wrongly-sized input should surface the same std::invalid_argument that
// Layer::forward() throws, not silently misbehave.
TEST_CASE("gradient: throws when the input size doesn't match the network",
          "[Gradient]") {
  Layer<Scalar>::Matrix W1(2, 1);
  W1 << Scalar(2.0), Scalar(3.0);
  Layer<Scalar>::Vector b1(2);
  b1 << Scalar(0.0), Scalar(0.0);
  Layer<Scalar> layer1(W1, b1, identity);

  FFNetwork<Scalar> net({layer1});

  FFNetwork<Scalar>::Vector wrong_x(2);  // layer1 expects size 1
  wrong_x << Scalar::constant(1.0), Scalar::constant(2.0);
  FFNetwork<Scalar>::Vector target(2);
  target << Scalar::constant(0.0), Scalar::constant(0.0);

  REQUIRE_THROWS_AS(gradient(net, wrong_x, target), std::invalid_argument);
}

// Independent cross-check (the Stage 9 pattern, applied early): flatten the
// network's parameters into a plain-double vector, run
// testutil::central_difference_gradient over a from-scratch, plain-double
// rebuild of the same forward pass + mse loss, and compare against
// gradient()'s forward-mode result entry by entry. Sigmoid activations (not
// identity) so the check exercises a genuinely nonlinear composition, not
// just an affine one.
TEST_CASE("gradient: matches a finite-difference cross-check of the whole network",
          "[Gradient][finite-diff]") {
  // Flat parameter order: W1 row-major (3x2), b1 (3), W2 row-major (1x3), b2 (1).
  const std::vector<double> params = {0.1, -0.2, 0.3,  0.4, -0.5, 0.6,  // W1
                                       0.1, -0.1, 0.2,                  // b1
                                       0.7, -0.3, 0.5,                  // W2
                                       0.05};                           // b2
  const double x0 = 0.5, x1 = -0.3, t0 = 1.0;

  auto rebuild_and_loss = [&](const std::vector<double>& p) {
    Layer<Scalar>::Matrix W1(3, 2);
    W1 << Scalar::constant(p[0]), Scalar::constant(p[1]), Scalar::constant(p[2]),
        Scalar::constant(p[3]), Scalar::constant(p[4]), Scalar::constant(p[5]);
    Layer<Scalar>::Vector b1(3);
    b1 << Scalar::constant(p[6]), Scalar::constant(p[7]), Scalar::constant(p[8]);
    Layer<Scalar> layer1(W1, b1, sigmoid_activation);

    Layer<Scalar>::Matrix W2(1, 3);
    W2 << Scalar::constant(p[9]), Scalar::constant(p[10]), Scalar::constant(p[11]);
    Layer<Scalar>::Vector b2(1);
    b2 << Scalar::constant(p[12]);
    Layer<Scalar> layer2(W2, b2, sigmoid_activation);

    FFNetwork<Scalar> net({layer1, layer2});

    FFNetwork<Scalar>::Vector x(2);
    x << Scalar::constant(x0), Scalar::constant(x1);
    FFNetwork<Scalar>::Vector target(1);
    target << Scalar::constant(t0);

    return mse(net.forward(x), target).value();
  };

  std::vector<double> numerical =
      testutil::central_difference_gradient(rebuild_and_loss, params);

  Layer<Scalar>::Matrix W1(3, 2);
  W1 << Scalar(params[0]), Scalar(params[1]), Scalar(params[2]), Scalar(params[3]),
      Scalar(params[4]), Scalar(params[5]);
  Layer<Scalar>::Vector b1(3);
  b1 << Scalar(params[6]), Scalar(params[7]), Scalar(params[8]);
  Layer<Scalar> layer1(W1, b1, sigmoid_activation);

  Layer<Scalar>::Matrix W2(1, 3);
  W2 << Scalar(params[9]), Scalar(params[10]), Scalar(params[11]);
  Layer<Scalar>::Vector b2(1);
  b2 << Scalar(params[12]);
  Layer<Scalar> layer2(W2, b2, sigmoid_activation);

  FFNetwork<Scalar> net({layer1, layer2});

  FFNetwork<Scalar>::Vector x(2);
  x << Scalar::constant(x0), Scalar::constant(x1);
  FFNetwork<Scalar>::Vector target(1);
  target << Scalar::constant(t0);

  auto grad = gradient(net, x, target);

  const std::vector<double> analytical = {
      grad[0].weights(0, 0), grad[0].weights(0, 1), grad[0].weights(1, 0),
      grad[0].weights(1, 1), grad[0].weights(2, 0), grad[0].weights(2, 1),
      grad[0].bias(0),       grad[0].bias(1),       grad[0].bias(2),
      grad[1].weights(0, 0), grad[1].weights(0, 1), grad[1].weights(0, 2),
      grad[1].bias(0)};

  REQUIRE(analytical.size() == numerical.size());
  for (std::size_t i = 0; i < analytical.size(); ++i) {
    INFO("param index = " << i);
    REQUIRE(testutil::is_close(analytical[i], numerical[i]));
  }
}
