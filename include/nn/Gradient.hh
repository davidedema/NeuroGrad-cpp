#pragma once

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <string>

#include "autodiff/Scalar.hh"
#include "nn/FFNetwork.hh"
#include "nn/Loss.hh"


namespace autodiff {
namespace nn {

  //! Column vector of Scalar, dynamic size.
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  //! Matrix of Scalar, dynamic size.
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  // A gradient entry is a plain double, not a Scalar: it's already the
  // unwrapped derivative read out of a Dual, not something meant to be
  // differentiated further.
  template <typename T = autodiff::Scalar>
  struct LayerGradient {
    Eigen::MatrixXd weights;  // same shape as Layer<T>::weights()
    Eigen::VectorXd bias;     // same shape as Layer<T>::bias()
  };

  template <typename T = autodiff::Scalar>
  using Gradient = std::vector<LayerGradient<T>>;

  template <typename T = autodiff::Scalar>
  Gradient<T> gradient(FFNetwork<T> &network, const Vector x, const Vector target)
  {
    auto &layers = network.layers();

    // Result mirrors the network's shape: one weight/bias gradient per
    // layer, same dimensions as that layer's own weights()/bias().
    Gradient<T> grad;
    for (const auto &layer : layers) {
      LayerGradient<T> layer_grad;
      layer_grad.weights =
          Eigen::MatrixXd::Zero(layer.weights().rows(), layer.weights().cols());
      layer_grad.bias = Eigen::VectorXd::Zero(layer.bias().size());
      grad.push_back(layer_grad);
    }

    // Resets every parameter in the whole network to a constant
    // (derivative 0), keeping its current value — needed before seeding
    // the next parameter so only one is ever "live" at a time.
    auto zero_all_parameters = [&layers]() {
      for (auto &layer : layers) {
        auto &weights = layer.weights();
        auto &bias = layer.bias();
        for (Eigen::Index r = 0; r < weights.rows(); ++r) {
          for (Eigen::Index c = 0; c < weights.cols(); ++c) {
            weights(r, c) = T::constant(weights(r, c).value());
          }
        }
        for (Eigen::Index i = 0; i < bias.size(); ++i) {
          bias(i) = T::constant(bias(i).value());
        }
      }
    };

    // Full forward pass + loss with whichever single parameter is
    // currently seeded to variable(); its derivative is dL/dp_i.
    auto loss_derivative = [&]() {
      Vector output = network.forward(x);
      T loss = mse(output, target);
      return loss.derivative();
    };

    for (std::size_t l = 0; l < layers.size(); ++l) {
      auto &weights = layers[l].weights();
      auto &bias = layers[l].bias();

      for (Eigen::Index r = 0; r < weights.rows(); ++r) {
        for (Eigen::Index c = 0; c < weights.cols(); ++c) {
          zero_all_parameters();
          weights(r, c) = T::variable(weights(r, c).value());
          grad[l].weights(r, c) = loss_derivative();
        }
      }

      for (Eigen::Index i = 0; i < bias.size(); ++i) {
        zero_all_parameters();
        bias(i) = T::variable(bias(i).value());
        grad[l].bias(i) = loss_derivative();
      }
    }

    // Leave the network fully constant — no stray seeded derivative on
    // whichever parameter was seeded last.
    zero_all_parameters();

    return grad;
  }

}  // namespace nn
}  // namespace autodiff
