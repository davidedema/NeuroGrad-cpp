#pragma once

#include <Eigen/Core>

#include "autodiff/Dual.hh"

// Placeholder for roadmap Stage 4 (Eigen integration).
//
// Goal: get `Eigen::Matrix<autodiff::forward::Dual<double>, ...>` compiling
// and doing a matmul. This needs, at minimum:
//   1. A specialization of Eigen::NumTraits<autodiff::forward::Dual<double>>.
//   2. Free functions `abs`, `abs2`, `sqrt` for Dual<T>, findable via ADL in
//      namespace autodiff::forward (sqrt already exists on Dual<T> once
//      Stage 1 is done; abs2(x) can just be x*x for a real-valued scalar
//      like this one).
//
// Suggested first test once this is filled in: build a small 2x2
// Eigen::Matrix<Dual<double>,...>, multiply it by a 2-vector, and confirm
// the result's derivatives match a hand-computed expectation before moving
// on to a Layer class. Don't build Layer on top of this until that test
// passes — Eigen + custom scalar types is one of the fiddlier integration
// points in this project.

namespace autodiff {
namespace forward {

// TODO(you): Eigen::NumTraits<Dual<double>> specialization goes here (note:
// this must be declared inside `namespace Eigen { ... }`, not this one —
// see the roadmap notes for the pattern used by similar autodiff types).

}  // namespace forward
}  // namespace autodiff
