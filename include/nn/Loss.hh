#pragma once

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <string>

#include "autodiff/Scalar.hh"

/**
 * @file Loss.hh
 * @brief Loss functions used to train a network (MSE, binary cross-entropy).
 */

// Free function, not a class: a loss has no state of its own to carry
// between calls, so it follows the same "elementary function" pattern
// already used for sigmoid/relu etc. in Dual.hh, rather than Layer's
// class-based approach (see docs/notes/implementation-roadmap.md Stage 6).

namespace autodiff {
namespace nn {

/**
 * @brief Mean-squared-error between a predicted vector and a target vector.
 *
 * @tparam T scalar type (defaults to `autodiff::Scalar`).
 * @param predicted `[n]` predicted vector (e.g. a network's output).
 * @param target `[n]` target vector, must be the same size as @p predicted.
 * @return T scalar loss, `mean((predicted_i - target_i)^2)`.
 * @throws std::invalid_argument if @p predicted and @p target have
 *   different sizes.
 */
template <typename T = autodiff::Scalar>
[[nodiscard]] T mse(const Eigen::Matrix<T, Eigen::Dynamic, 1>& predicted,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& target) {
  if (predicted.size() != target.size()) {
    throw std::invalid_argument(
        "mse: predicted size " + std::to_string(predicted.size()) +
        " does not match target size " + std::to_string(target.size()));
  }

  T sum = T(0);
  for (Eigen::Index i = 0; i < predicted.size(); ++i) {
    const T diff = predicted(i) - target(i);
    sum += diff * diff;
  }
  return sum / T(predicted.size());
}

/**
 * @brief Binary cross-entropy between a predicted probability vector and a
 *   0/1 target vector.
 *
 * @tparam T scalar type (defaults to `autodiff::Scalar`).
 * @param predicted `[n]` predicted probabilities (e.g. sigmoid outputs),
 *   each expected strictly inside `(0, 1)` — `log` diverges at the
 *   boundary, so a `predicted_i` of exactly 0 or 1 is not supported.
 * @param target `[n]` 0/1 labels, must be the same size as @p predicted.
 * @return T scalar loss, `-mean(target_i*log(predicted_i) +
 *   (1-target_i)*log(1-predicted_i))`.
 * @throws std::invalid_argument if @p predicted and @p target have
 *   different sizes.
 */
template <typename T = autodiff::Scalar>
[[nodiscard]] T bce(const Eigen::Matrix<T, Eigen::Dynamic, 1>& predicted,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& target) {
  if (predicted.size() != target.size()) {
    throw std::invalid_argument(
        "bce: predicted size " + std::to_string(predicted.size()) +
        " does not match target size " + std::to_string(target.size()));
  }

  using std::log;

  T sum = T(0);
  for (Eigen::Index i = 0; i < predicted.size(); ++i) {
    const T& p = predicted(i);
    const T& t = target(i);
    sum += t * log(p) + (T(1) - t) * log(T(1) - p);
  }
  return -sum / T(predicted.size());
}

}  // namespace nn
}  // namespace autodiff
