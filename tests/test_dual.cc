#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "autodiff/Dual.hh"
#include "utils/finite_diff.hh"

using autodiff::forward::Dual;
namespace ad = autodiff::forward;

// ---------------------------------------------------------------------
// Arithmetic operators: exact, hand-checkable values. No finite differences
// needed here since we know the closed-form derivative by construction —
// finite differences are for validating elementary functions further down,
// not a substitute for reasoning about simple cases directly.
// ---------------------------------------------------------------------

TEST_CASE("Dual: variable() and constant() seed correctly", "[Dual]") {
  Dual<double> x = Dual<double>::variable(2.0);
  REQUIRE(x.value() == 2.0);
  REQUIRE(x.derivative() == 1.0);

  Dual<double> c = Dual<double>::constant(3.0);
  REQUIRE(c.value() == 3.0);
  REQUIRE(c.derivative() == 0.0);
}

TEST_CASE("Dual: addition and subtraction are linear", "[Dual]") {
  Dual<double> x = Dual<double>::variable(2.0);  // d(x)/dx = 1
  Dual<double> c = Dual<double>::constant(5.0);  // d(5)/dx = 0

  SECTION("x + c") {
    Dual<double> z = x + c;
    REQUIRE(z.value() == 7.0);
    REQUIRE(z.derivative() == 1.0);
  }
  SECTION("x - c") {
    Dual<double> z = x - c;
    REQUIRE(z.value() == -3.0);
    REQUIRE(z.derivative() == 1.0);
  }
  SECTION("c - x") {
    Dual<double> z = c - x;
    REQUIRE(z.value() == 3.0);
    REQUIRE(z.derivative() == -1.0);
  }
}

TEST_CASE("Dual: multiplication follows the product rule", "[Dual]") {
  Dual<double> x = Dual<double>::variable(3.0);  // x = 3, dx/dx = 1
  Dual<double> y = Dual<double>(4.0, 2.0);       // stand-in for some y(x) with y=4, dy/dx=2
  Dual<double> z = x * y;
  // d(x*y)/dx = x'*y + x*y' = 1*4 + 3*2 = 10
  REQUIRE(z.value() == 12.0);
  REQUIRE(z.derivative() == 10.0);
}

TEST_CASE("Dual: division follows the quotient rule", "[Dual]") {
  Dual<double> x = Dual<double>(6.0, 1.0);
  Dual<double> y = Dual<double>(3.0, 2.0);
  Dual<double> z = x / y;
  // d(x/y)/dx = (x'*y - x*y') / y^2 = (1*3 - 6*2) / 9 = -1
  REQUIRE(z.value() == 2.0);
  REQUIRE(testutil::is_close(z.derivative(), -1.0));
}

TEST_CASE("Dual: unary negation", "[Dual]") {
  Dual<double> x = Dual<double>::variable(2.5);
  Dual<double> z = -x;
  REQUIRE(z.value() == -2.5);
  REQUIRE(z.derivative() == -1.0);
}

TEST_CASE("Dual: comparisons compare value only", "[Dual]") {
  Dual<double> a = Dual<double>(1.0, 100.0);  // huge derivative, shouldn't affect comparison
  Dual<double> b = Dual<double>(2.0, -100.0);
  REQUIRE(a < b);
  REQUIRE(b > a);
  REQUIRE(a != b);
  REQUIRE(Dual<double>(1.0, 5.0) == Dual<double>(1.0, -5.0));  // same value, different deriv
  REQUIRE(a <= b);
  REQUIRE(b >= a);
  REQUIRE(Dual<double>(1.0, 5.0) <= Dual<double>(1.0, -5.0));  // equal values
  REQUIRE(Dual<double>(1.0, 5.0) >= Dual<double>(1.0, -5.0));
}

// ---------------------------------------------------------------------
// Elementary functions: checked against central finite differences.
//
// This is the pattern to copy for every new differentiable building block
// added later (Layer, loss functions, ...): pick a handful of
// representative points (avoid points where the function is undefined,
// e.g. log at 0), and compare the analytic derivative from Dual against
// testutil::central_difference on the plain-double version of the same
// function. See tests/utils/finite_diff.hh.
// ---------------------------------------------------------------------

TEST_CASE("Dual: exp matches finite-difference derivative", "[Dual][finite-diff]") {
  for (double x0 : {-1.5, -0.1, 0.0, 0.3, 2.0}) {
    Dual<double> x = Dual<double>::variable(x0);
    Dual<double> y = ad::exp(x);
    double numerical =
        testutil::central_difference([](double v) { return std::exp(v); }, x0);
    INFO("x0 = " << x0);
    REQUIRE(testutil::is_close(y.derivative(), numerical));
  }
}

TEST_CASE("Dual: log matches finite-difference derivative", "[Dual][finite-diff]") {
  for (double x0 : {0.1, 0.5, 1.0, 3.0, 10.0}) {  // log needs x > 0
    Dual<double> x = Dual<double>::variable(x0);
    Dual<double> y = ad::log(x);
    double numerical =
        testutil::central_difference([](double v) { return std::log(v); }, x0);
    INFO("x0 = " << x0);
    REQUIRE(testutil::is_close(y.derivative(), numerical));
  }
}

TEST_CASE("Dual: sqrt matches finite-difference derivative", "[Dual][finite-diff]") {
  for (double x0 : {0.25, 1.0, 2.0, 9.0}) {  // sqrt needs x > 0
    Dual<double> x = Dual<double>::variable(x0);
    Dual<double> y = ad::sqrt(x);
    double numerical =
        testutil::central_difference([](double v) { return std::sqrt(v); }, x0);
    INFO("x0 = " << x0);
    REQUIRE(testutil::is_close(y.derivative(), numerical));
  }
}

TEST_CASE("Dual: sin and cos match finite-difference derivatives", "[Dual][finite-diff]") {
  for (double x0 : {-2.0, -0.5, 0.0, 0.5, 2.0}) {
    Dual<double> x = Dual<double>::variable(x0);

    Dual<double> s = ad::sin(x);
    double numerical_sin =
        testutil::central_difference([](double v) { return std::sin(v); }, x0);
    INFO("sin, x0 = " << x0);
    REQUIRE(testutil::is_close(s.derivative(), numerical_sin));

    Dual<double> c = ad::cos(x);
    double numerical_cos =
        testutil::central_difference([](double v) { return std::cos(v); }, x0);
    INFO("cos, x0 = " << x0);
    REQUIRE(testutil::is_close(c.derivative(), numerical_cos));
  }
}

TEST_CASE("Dual: tanh matches finite-difference derivative", "[Dual][finite-diff]") {
  for (double x0 : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
    Dual<double> x = Dual<double>::variable(x0);
    Dual<double> y = ad::tanh(x);
    double numerical =
        testutil::central_difference([](double v) { return std::tanh(v); }, x0);
    INFO("x0 = " << x0);
    REQUIRE(testutil::is_close(y.derivative(), numerical));
  }
}

TEST_CASE("Dual: sigmoid matches finite-difference derivative", "[Dual][finite-diff]") {
  auto sigmoid_double = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
  for (double x0 : {-4.0, -1.0, 0.0, 1.0, 4.0}) {
    Dual<double> x = Dual<double>::variable(x0);
    Dual<double> y = ad::sigmoid(x);
    double numerical = testutil::central_difference(sigmoid_double, x0);
    INFO("x0 = " << x0);
    REQUIRE(testutil::is_close(y.derivative(), numerical));
  }
}

TEST_CASE("Dual: pow(x, p) matches finite-difference derivative", "[Dual][finite-diff]") {
  for (double p : {2.0, 3.0, 0.5}) {
    for (double x0 : {0.5, 1.0, 2.0, 4.0}) {  // stay > 0 (pow(x,0.5) needs it)
      Dual<double> x = Dual<double>::variable(x0);
      Dual<double> y = ad::pow(x, p);
      double numerical = testutil::central_difference(
          [p](double v) { return std::pow(v, p); }, x0);
      INFO("p = " << p << ", x0 = " << x0);
      REQUIRE(testutil::is_close(y.derivative(), numerical));
    }
  }
}

// Every test above finite-diff-checks one elementary function in isolation.
// That alone can't catch a bug in how operators/functions COMPOSE (e.g. an
// operator that mutates the wrong side, or a derivative that gets dropped
// when chained through several ops) — this test chains arithmetic and
// several elementary functions together and checks the resulting derivative
// against a finite difference of the same composite function on doubles.
TEST_CASE("Dual: composite expression matches finite-difference derivative",
          "[Dual][finite-diff]") {
  auto f = [](double v) {
    return std::sin(v) * std::exp(v) + std::sqrt(v * v + 1.0);
  };
  for (double x0 : {-2.0, -0.5, 0.3, 1.0, 2.5}) {
    Dual<double> x = Dual<double>::variable(x0);
    Dual<double> y =
        ad::sin(x) * ad::exp(x) + ad::sqrt(x * x + Dual<double>(1.0));
    double numerical = testutil::central_difference(f, x0);
    INFO("x0 = " << x0);
    REQUIRE(testutil::is_close(y.derivative(), numerical));
  }
}

// relu is intentionally NOT finite-difference-checked at x = 0 (the
// function isn't differentiable there — finite differences would just tell
// you the average of the left/right derivative, not your chosen
// convention). Assert the convention explicitly instead.
TEST_CASE("Dual: relu", "[Dual]") {
  SECTION("x > 0: passes through, derivative 1") {
    Dual<double> x = Dual<double>::variable(2.0);
    Dual<double> y = ad::relu(x);
    REQUIRE(y.value() == 2.0);
    REQUIRE(y.derivative() == 1.0);
  }
  SECTION("x < 0: zeroed out, derivative 0") {
    Dual<double> x = Dual<double>::variable(-2.0);
    Dual<double> y = ad::relu(x);
    REQUIRE(y.value() == 0.0);
    REQUIRE(y.derivative() == 0.0);
  }
  SECTION("x == 0: subgradient convention is 0") {
    Dual<double> x = Dual<double>::variable(0.0);
    Dual<double> y = ad::relu(x);
    REQUIRE(y.value() == 0.0);
    REQUIRE(y.derivative() == 0.0);
  }
}

// Every case above seeds x via Dual::variable() (derivative == 1), which
// can't distinguish "correctly propagated the incoming derivative" from
// "hardcoded the output derivative to 1". Compose relu with a non-identity
// derivative to catch that distinction.
TEST_CASE("Dual: relu propagates the chain rule for a non-identity derivative", "[Dual]") {
  SECTION("x > 0: derivative is the chain rule product, not 1") {
    Dual<double> x = Dual<double>::variable(3.0);
    Dual<double> y = x * Dual<double>(2.0);  // y.value() = 6, y.derivative() = 2
    Dual<double> z = ad::relu(y);
    REQUIRE(z.value() == 6.0);
    REQUIRE(z.derivative() == 2.0);
  }
  SECTION("x < 0: derivative is zeroed regardless of incoming derivative") {
    Dual<double> x = Dual<double>::variable(-3.0);
    Dual<double> y = x * Dual<double>(2.0);  // y.value() = -6, y.derivative() = 2
    Dual<double> z = ad::relu(y);
    REQUIRE(z.value() == 0.0);
    REQUIRE(z.derivative() == 0.0);
  }
}
