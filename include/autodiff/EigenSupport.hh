#pragma once

#include <Eigen/Core>

#include "autodiff/Dual.hh"

// Roadmap Stage 4 (Eigen integration): makes
// `Eigen::Matrix<autodiff::forward::Dual<double>, ...>` compile and do a
// matmul. Eigen discovers how to treat a scalar type via
// Eigen::NumTraits<T> (declared inside `namespace Eigen`, not
// autodiff::forward) plus free functions `abs`, `abs2`, `sqrt` for Dual<T>,
// findable via ADL in namespace autodiff::forward — all defined in
// Dual.hh, alongside the other elementary functions.
//
// Verified by tests/test_eigen_integration.cc: a small 2x2
// Eigen::Matrix<Dual<double>,...> times a 2-vector, with the result's
// derivatives checked against a hand-computed expectation. Don't build
// Layer on top of this without that test passing — Eigen + custom scalar
// types is one of the fiddlier integration points in this project.

namespace Eigen {
template <>
struct NumTraits<autodiff::forward::Dual<double>>
    : NumTraits<double> {};
}  // namespace Eigen