#include <catch2/catch_test_macros.hpp>

#include "nn/FFNetwork.hh"
#include "nn/Layer.hh"

using autodiff::Scalar;
using autodiff::nn::FFNetwork;
using autodiff::nn::Layer;

namespace {
Scalar identity(const Scalar& x) { return x; }
Scalar relu_activation(const Scalar& x) { return autodiff::forward::relu(x); }
}  // namespace

// This is the whole point of FFNetwork over a single Layer: consecutive
// layers with DIFFERENT input/output dimensions (2 -> 3 -> 1 here), chained
// so layer i's output feeds layer i+1's input.
//
//   Layer 1 (2 -> 3, identity): W1 = [ 1 0 ]   b1 = [ 0 ]
//                                    [ 0 1 ]        [ 0 ]
//                                    [ 1 1 ]        [ 1 ]
//     h = W1*x + b1 = [ x0, x1, x0 + x1 + 1 ]
//
//   Layer 2 (3 -> 1, identity): W2 = [ 1 1 1 ]   b2 = [ 0 ]
//     y = W2*h + b2 = x0 + x1 + (x0 + x1 + 1) = 2*x0 + 2*x1 + 1
//
// At x = (2, 3): y = 2*2 + 2*3 + 1 = 11; dy/dx0 = 2.
TEST_CASE("FFNetwork: forward chains layers with different in/out dimensions",
          "[FFNetwork]") {
  Layer<Scalar>::Matrix W1(3, 2);
  W1 << Scalar(1.0), Scalar(0.0), Scalar(0.0), Scalar(1.0), Scalar(1.0),
      Scalar(1.0);
  Layer<Scalar>::Vector b1(3);
  b1 << Scalar(0.0), Scalar(0.0), Scalar(1.0);
  Layer<Scalar> layer1(W1, b1, identity);

  Layer<Scalar>::Matrix W2(1, 3);
  W2 << Scalar(1.0), Scalar(1.0), Scalar(1.0);
  Layer<Scalar>::Vector b2(1);
  b2 << Scalar(0.0);
  Layer<Scalar> layer2(W2, b2, identity);

  FFNetwork<Scalar> net({layer1, layer2});

  FFNetwork<Scalar>::Vector x(2);
  x << Scalar(2.0), Scalar(3.0);

  FFNetwork<Scalar>::Vector y = net.forward(x);

  REQUIRE(y.size() == 1);
  REQUIRE(y(0).value() == 11.0);
}

TEST_CASE("FFNetwork: forward propagates the derivative through chained layers",
          "[FFNetwork]") {
  Layer<Scalar>::Matrix W1(3, 2);
  W1 << Scalar(1.0), Scalar(0.0), Scalar(0.0), Scalar(1.0), Scalar(1.0),
      Scalar(1.0);
  Layer<Scalar>::Vector b1(3);
  b1 << Scalar(0.0), Scalar(0.0), Scalar(1.0);
  Layer<Scalar> layer1(W1, b1, identity);

  Layer<Scalar>::Matrix W2(1, 3);
  W2 << Scalar(1.0), Scalar(1.0), Scalar(1.0);
  Layer<Scalar>::Vector b2(1);
  b2 << Scalar(0.0);
  Layer<Scalar> layer2(W2, b2, identity);

  FFNetwork<Scalar> net({layer1, layer2});

  FFNetwork<Scalar>::Vector x(2);
  x << Scalar::variable(2.0), Scalar::constant(3.0);  // differentiate w.r.t. x0

  FFNetwork<Scalar>::Vector y = net.forward(x);

  REQUIRE(y(0).derivative() == 2.0);  // dy/dx0 = 2 (see derivation above)
}

// Chains a relu layer into an identity layer to check that (a) each layer's
// own activation convention (relu's derivative-zeroing at negative
// pre-activations) survives being composed inside a network, not just a
// standalone Layer, and (b) a zeroed branch doesn't corrupt the derivative
// of a branch that stayed live.
//
//   Layer 1 (2 -> 2, relu): W1 = [ 1  0 ]   b1 = [ 0 ]
//                                [ 0 -1 ]        [ 0 ]
//     pre = [ x0, -x1 ]; h = relu(pre)
//
//   Layer 2 (2 -> 1, identity): W2 = [ 1 1 ]   b2 = [ 0 ]
//     y = h0 + h1
//
// At x = (2, 1): pre = (2, -1) -> h = (2, 0) -> y = 2.
//   dy/dx0: pre0 = x0 > 0, so relu passes the derivative through -> 1.
//   dy/dx1: pre1 = -x1 < 0, so relu zeroes the derivative -> 0, even though
//           pre1's own derivative w.r.t. x1 (-1) is nonzero.
TEST_CASE("FFNetwork: relu's derivative convention survives composition across layers",
          "[FFNetwork]") {
  Layer<Scalar>::Matrix W1(2, 2);
  W1 << Scalar(1.0), Scalar(0.0), Scalar(0.0), Scalar(-1.0);
  Layer<Scalar>::Vector b1(2);
  b1 << Scalar(0.0), Scalar(0.0);
  Layer<Scalar> layer1(W1, b1, relu_activation);

  Layer<Scalar>::Matrix W2(1, 2);
  W2 << Scalar(1.0), Scalar(1.0);
  Layer<Scalar>::Vector b2(1);
  b2 << Scalar(0.0);
  Layer<Scalar> layer2(W2, b2, identity);

  FFNetwork<Scalar> net({layer1, layer2});

  SECTION("derivative w.r.t. x0 (live branch)") {
    FFNetwork<Scalar>::Vector x(2);
    x << Scalar::variable(2.0), Scalar::constant(1.0);
    FFNetwork<Scalar>::Vector y = net.forward(x);
    REQUIRE(y(0).value() == 2.0);
    REQUIRE(y(0).derivative() == 1.0);
  }

  SECTION("derivative w.r.t. x1 (zeroed by relu)") {
    FFNetwork<Scalar>::Vector x(2);
    x << Scalar::constant(2.0), Scalar::variable(1.0);
    FFNetwork<Scalar>::Vector y = net.forward(x);
    REQUIRE(y(0).value() == 2.0);
    REQUIRE(y(0).derivative() == 0.0);
  }
}
