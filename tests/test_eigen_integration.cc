#include <catch2/catch_test_macros.hpp>
#include <Eigen/Core>

#include "autodiff/Dual.hh"
#include "autodiff/EigenSupport.hh"

using autodiff::forward::Dual;

// Roadmap Stage 4 exit criterion: a small Eigen::Matrix<Dual<double>,...>
// must compile and do a matmul, and the result's derivatives must match a
// hand-computed expectation (not just "it compiles").
//
//   W = [ 2.0  -1.0 ]   x = [ x0 ]   (x0 is the seeded variable, x1 constant)
//       [ 0.5   3.0 ]       [ x1 ]
//
//   y = W*x:
//     y0 = W00*x0 + W01*x1  =>  dy0/dx0 = W00 = 2.0
//     y1 = W10*x0 + W11*x1  =>  dy1/dx0 = W10 = 0.5
TEST_CASE("Eigen: 2x2 Dual matrix times vector matches hand-computed derivative",
          "[Eigen]") {
  using Mat2 = Eigen::Matrix<Dual<double>, 2, 2>;
  using Vec2 = Eigen::Matrix<Dual<double>, 2, 1>;

  Mat2 W;
  W(0, 0) = Dual<double>::constant(2.0);
  W(0, 1) = Dual<double>::constant(-1.0);
  W(1, 0) = Dual<double>::constant(0.5);
  W(1, 1) = Dual<double>::constant(3.0);

  Vec2 x;
  x(0) = Dual<double>::variable(1.5);   // differentiate w.r.t. x0
  x(1) = Dual<double>::constant(-2.0);

  Vec2 y = W * x;

  REQUIRE(y(0).value() == 5.0);       // 2*1.5 + (-1)*(-2)
  REQUIRE(y(0).derivative() == 2.0);  // dy0/dx0 = W00

  REQUIRE(y(1).value() == -5.25);     // 0.5*1.5 + 3*(-2)
  REQUIRE(y(1).derivative() == 0.5);  // dy1/dx0 = W10
}
