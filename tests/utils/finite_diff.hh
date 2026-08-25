#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

// Shared test infrastructure: every class's tests that need to validate a
// derivative against a numerical reference should use these helpers instead
// of reimplementing finite differences locally. Keeps the check consistent
// (same tolerance convention, same step size) across Dual, and later Layer /
// NeuralNetwork gradient checks (roadmap Stage 8).

namespace testutil {

// Central-difference derivative of a scalar function f: double -> double
// at point x. O(h^2) accurate; h = 1e-6 is a reasonable default for
// double precision (small enough for accuracy, large enough to avoid
// catastrophic cancellation).
inline double central_difference(const std::function<double(double)>& f,
                                  double x, double h = 1e-6) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

// Central-difference gradient of f: R^n -> R at point x. Perturbs one
// coordinate at a time. This is the same "loop over parameters, perturb,
// recompute" pattern the roadmap describes for Stage 8 (whole-network
// gradient check) — reuse it there rather than writing a second version.
inline std::vector<double> central_difference_gradient(
    const std::function<double(const std::vector<double>&)>& f,
    std::vector<double> x, double h = 1e-6) {
  std::vector<double> grad(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double original = x[i];

    x[i] = original + h;
    const double f_plus = f(x);

    x[i] = original - h;
    const double f_minus = f(x);

    x[i] = original;  // restore before moving to the next coordinate
    grad[i] = (f_plus - f_minus) / (2.0 * h);
  }
  return grad;
}

// Closeness check that behaves sensibly both near zero (absolute
// tolerance dominates) and away from zero (relative tolerance dominates).
// Prefer this over a bare `==` or a single fixed epsilon for anything
// derived from finite differences.
inline bool is_close(double a, double b, double rel_tol = 1e-5,
                      double abs_tol = 1e-8) {
  return std::fabs(a - b) <=
         std::max(abs_tol, rel_tol * std::max(std::fabs(a), std::fabs(b)));
}

}  // namespace testutil
