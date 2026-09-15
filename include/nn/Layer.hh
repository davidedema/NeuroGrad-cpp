#pragma once

#include <Eigen/Core>
#include <functional>
#include <utility>

#include "autodiff/EigenSupport.hh"
#include "autodiff/Scalar.hh"

/**
 * @file Layer.hh
 * @brief A single fully-connected layer: `activation(W*x + b)`.
 */

// This header fixes the public interface and structure only (roadmap
// Stage 5) — forward()'s body is a placeholder, so the project compiles
// and tests/test_layer.cc runs (and fails "red") before you've implemented
// anything, same pattern Dual.hh used for Stage 1. Fill in forward() and
// watch tests/test_layer.cc turn green.
//
// See docs/notes/implementation-roadmap.md Stage 5: W and b are Eigen
// matrices/vectors of Scalar (the Stage 3 swap-point alias, so this class
// is ready for a reverse-mode engine later without changes), and the
// activation is passed in rather than hardcoded, so tanh/sigmoid/relu (all
// already implemented on Dual, see Dual.hh) can be swapped without
// rewriting Layer.

namespace autodiff {
namespace nn {

/**
 * @brief A dense layer computing `activation(W*x + b)`.
 *
 * @tparam T scalar type used for weights, bias, input, and output
 *   (defaults to `autodiff::Scalar`, the current forward/reverse-mode
 *   swap point).
 */
template <typename T = autodiff::Scalar>
class Layer {
 public:
  //! Underlying scalar type.
  using Scalar = T;
  //! Column vector of Scalar, dynamic size.
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  //! Matrix of Scalar, dynamic size.
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  //! Elementwise nonlinearity applied to `W*x + b`.
  using Activation = std::function<Scalar(const Scalar&)>;

  /**
   * @brief Builds a layer from an existing weight matrix, bias vector, and
   *   activation function.
   *
   * @param weights `[out_features x in_features]` weight matrix.
   * @param bias `[out_features]` bias vector.
   * @param activation elementwise nonlinearity applied to `W*x + b`.
   */
  Layer(Matrix weights, Vector bias, Activation activation);

  /**
   * @brief Computes `activation(W*x + b)`.
   *
   * @param x `[in_features]` input vector.
   * @return Vector `[out_features]` layer output.
   */
  [[nodiscard]] Vector forward(const Vector& x) const;

  //! Weight matrix, `[out_features x in_features]`.
  [[nodiscard]] const Matrix& weights() const noexcept { return weights_; }

  //! Bias vector, `[out_features]`.
  [[nodiscard]] const Vector& bias() const noexcept { return bias_; }

 private:
  Matrix weights_;
  Vector bias_;
  Activation activation_;
};

template <typename T>
Layer<T>::Layer(Matrix weights, Vector bias, Activation activation)
    : weights_(std::move(weights)),
      bias_(std::move(bias)),
      activation_(std::move(activation)) {}

template <typename T>
typename Layer<T>::Vector Layer<T>::forward(const Vector& x) const {
  // TODO(Stage 5): compute W*x + b, then apply activation_ elementwise
  // (Eigen's .unaryExpr(activation_) is the natural tool here).
  return Vector::Zero(weights_.rows());
}

}  // namespace nn
}  // namespace autodiff
