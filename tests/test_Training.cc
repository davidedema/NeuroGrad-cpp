#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "autodiff/Dual.hh"
#include "nn/FFNetwork.hh"
#include "nn/Layer.hh"
#include "nn/Loss.hh"
#include "nn/Training.hh"
#include "utils/finite_diff.hh"

using autodiff::Scalar;
using autodiff::nn::bce;
using autodiff::nn::Dataset;
using autodiff::nn::FFNetwork;
using autodiff::nn::Layer;
using autodiff::nn::mse;
using autodiff::nn::train;
using autodiff::nn::train_step;

namespace {
Scalar identity(const Scalar& x) { return x; }
}  // namespace

// Layer 1 (1 -> 2, identity): W1 = [ 2 ]   b1 = [ 0 ]
//                                  [ 3 ]        [ 0 ]
// Layer 2 (2 -> 1, identity): W2 = [ 1 1 ]   b2 = [ 0 ]
//
// Same network as test_Gradient.cc's hand-computed case: at x = 5,
// y = 25, loss = mse(y, 0) = 625, and the gradient is
// dW1 = [250, 250], db1 = [50, 50], dW2 = [500, 750], db2 = [50].
TEST_CASE("train_step: returns the pre-update loss and applies gradient descent",
          "[Training]") {
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

  const double learning_rate = 0.01;
  double loss = train_step(net, x, target, learning_rate);

  REQUIRE(loss == 625.0);

  auto& layers = net.layers();
  REQUIRE(layers[0].weights()(0, 0).value() == 2.0 - learning_rate * 250.0);
  REQUIRE(layers[0].weights()(1, 0).value() == 3.0 - learning_rate * 250.0);
  REQUIRE(layers[0].bias()(0).value() == 0.0 - learning_rate * 50.0);
  REQUIRE(layers[0].bias()(1).value() == 0.0 - learning_rate * 50.0);

  REQUIRE(layers[1].weights()(0, 0).value() == 1.0 - learning_rate * 500.0);
  REQUIRE(layers[1].weights()(0, 1).value() == 1.0 - learning_rate * 750.0);
  REQUIRE(layers[1].bias()(0).value() == 0.0 - learning_rate * 50.0);
}

// A single layer with W = [0], b = [0.7] always outputs 0.7 regardless of x
// (identity activation), so the pre-update loss is hand-computable for
// either loss function without worrying about the network's actual
// forward-pass arithmetic — this isolates "is the passed-in loss function
// actually used" from the gradient/update machinery already covered above.
TEST_CASE("train_step: uses whichever loss function is passed in", "[Training]") {
  auto make_net = []() {
    Layer<Scalar>::Matrix W(1, 1);
    W << Scalar(0.0);
    Layer<Scalar>::Vector b(1);
    b << Scalar(0.7);
    return FFNetwork<Scalar>({Layer<Scalar>(W, b, identity)});
  };

  FFNetwork<Scalar>::Vector x(1);
  x << Scalar::constant(1.0);
  FFNetwork<Scalar>::Vector target(1);
  target << Scalar::constant(1.0);

  FFNetwork<Scalar> net_mse = make_net();
  double mse_loss = train_step(net_mse, x, target, 0.0, mse<Scalar>);
  REQUIRE(testutil::is_close(mse_loss, 0.09));  // (0.7 - 1)^2

  FFNetwork<Scalar> net_bce = make_net();
  double bce_loss = train_step(net_bce, x, target, 0.0, bce<Scalar>);
  REQUIRE(testutil::is_close(bce_loss, -std::log(0.7)));
}

TEST_CASE("train: throws on an empty dataset", "[Training]") {
  Layer<Scalar>::Matrix W(1, 1);
  W << Scalar(1.0);
  Layer<Scalar>::Vector b(1);
  b << Scalar(0.0);
  FFNetwork<Scalar> net({Layer<Scalar>(W, b, identity)});

  REQUIRE_THROWS_AS(train(net, Dataset<Scalar>{}, 0.1, 5), std::invalid_argument);
}

// W = [0] means the weight's gradient is always 0 (dOutput/dW = x = 0 for
// every example below), so only the bias moves — this keeps the per-example
// sequential update (each example in an epoch trains on the network as left
// by the previous example, not the epoch-start network) hand-computable:
//
//   epoch start: b0 = 0
//   example 1 (target 1.0): loss1 = (b0 - 1)^2 = 1; b1 = b0 - lr*2*(b0-1) = 0.2
//   example 2 (target 0.0): loss2 = b1^2 = 0.04;    b2 = b1 - lr*2*b1     = 0.16
//   epoch loss = mean(loss1, loss2) = 0.52
TEST_CASE("train: averages the pre-update loss over the dataset, "
          "updating sequentially within an epoch",
          "[Training]") {
  Layer<Scalar>::Matrix W(1, 1);
  W << Scalar(0.0);
  Layer<Scalar>::Vector b(1);
  b << Scalar(0.0);
  FFNetwork<Scalar> net({Layer<Scalar>(W, b, identity)});

  FFNetwork<Scalar>::Vector x(1);
  x << Scalar::constant(0.0);
  FFNetwork<Scalar>::Vector target1(1);
  target1 << Scalar::constant(1.0);
  FFNetwork<Scalar>::Vector target2(1);
  target2 << Scalar::constant(0.0);

  Dataset<Scalar> data = {{x, target1}, {x, target2}};

  auto losses = train(net, data, 0.1, 1);

  REQUIRE(losses.size() == 1);
  REQUIRE(testutil::is_close(losses[0], 0.52));
}

// Sanity check that training actually reduces loss over several epochs on a
// small dataset (not a finite-diff derivative check — that's covered for
// gradient() itself in test_Gradient.cc — just "does the loop converge").
TEST_CASE("train: loss decreases over epochs on a small dataset", "[Training]") {
  Layer<Scalar>::Matrix W(1, 1);
  W << Scalar(0.5);
  Layer<Scalar>::Vector b(1);
  b << Scalar(0.0);
  FFNetwork<Scalar> net({Layer<Scalar>(W, b, identity)});

  FFNetwork<Scalar>::Vector x1(1);
  x1 << Scalar::constant(1.0);
  FFNetwork<Scalar>::Vector target1(1);
  target1 << Scalar::constant(2.0);

  FFNetwork<Scalar>::Vector x2(1);
  x2 << Scalar::constant(-1.0);
  FFNetwork<Scalar>::Vector target2(1);
  target2 << Scalar::constant(-2.0);

  Dataset<Scalar> data = {{x1, target1}, {x2, target2}};

  auto losses = train(net, data, 0.1, 50);

  REQUIRE(losses.size() == 50);
  REQUIRE(losses.back() < losses.front());
}
