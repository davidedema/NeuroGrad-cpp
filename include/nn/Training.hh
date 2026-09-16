#pragma once

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "autodiff/Scalar.hh"
#include "nn/Gradient.hh"
#include "nn/Loss.hh"

namespace autodiff {
namespace nn {

  //! One (input, target) training example.
  template <typename T = autodiff::Scalar>
  using Example = std::pair<Eigen::Matrix<T, Eigen::Dynamic, 1>,
                             Eigen::Matrix<T, Eigen::Dynamic, 1>>;

  //! A full training set: an ordered collection of examples.
  template <typename T = autodiff::Scalar>
  using Dataset = std::vector<Example<T>>;

  //! One (predicted, target) pair, as returned by `inference()`.
  template <typename T = autodiff::Scalar>
  using Prediction = std::pair<Eigen::Matrix<T, Eigen::Dynamic, 1>,
                                Eigen::Matrix<T, Eigen::Dynamic, 1>>;

  /**
   * @brief One gradient-descent update on a single example: computes the
   *   gradient of @p loss w.r.t. every weight/bias in @p network and applies
   *   `p -= learning_rate * dL/dp` to each of them in place.
   *
   * @param network network to update, mutated in place.
   * @param x `[in_features]` input example.
   * @param target `[out_features]` expected output for @p x.
   * @param learning_rate step size applied to every parameter.
   * @param loss loss function to train against (defaults to `mse`).
   * @return double the loss *before* this step's update — the value the
   *   gradient just applied was actually computed against.
   */
  template <typename T = autodiff::Scalar>
  double train_step(FFNetwork<T>& network, const Vector& x, const Vector& target,
                     double learning_rate, LossFn<T> loss = mse<T>)
  {
    const double current_loss = loss(network.forward(x), target).value();

    Gradient<T> grad = gradient(network, x, target, loss);
    auto &layers = network.layers();
    for (std::size_t l = 0; l < layers.size(); ++l) {
      auto &weights = layers[l].weights();
      auto &bias = layers[l].bias();

      for (Eigen::Index r = 0; r < weights.rows(); ++r) {
        for (Eigen::Index c = 0; c < weights.cols(); ++c) {
          weights(r, c) -= learning_rate * grad[l].weights(r, c);
        }
      }

      for (Eigen::Index i = 0; i < bias.size(); ++i) {
        bias(i) -= learning_rate * grad[l].bias(i);
      }
    }

    return current_loss;
  }

  /**
   * @brief Trains @p network on @p dataset for @p epochs epochs: each epoch
   *   runs `train_step` once per example in @p dataset, in order.
   *
   * @param network network to train, mutated in place.
   * @param dataset training examples; must be non-empty.
   * @param learning_rate step size applied to every parameter.
   * @param epochs number of passes over the whole @p dataset.
   * @param loss loss function to train against (defaults to `mse`).
   * @return std::vector<double> one entry per epoch, the mean per-example
   *   loss (before that example's update) over @p dataset.
   * @throws std::invalid_argument if @p dataset is empty.
   */
  template <typename T = autodiff::Scalar>
  std::vector<double> train(FFNetwork<T>& network, const Dataset<T>& dataset,
                             double learning_rate, int epochs, LossFn<T> loss = mse<T>)
  {
    if (dataset.empty()) {
      throw std::invalid_argument("train: dataset must not be empty");
    }

    std::vector<double> losses;
    losses.reserve(epochs);

    for (int epoch = 0; epoch < epochs; ++epoch) {
      double epoch_loss = 0.0;
      for (const auto& example : dataset) {
        const auto& x = example.first;
        const auto& target = example.second;
        epoch_loss += train_step(network, x, target, learning_rate, loss);
      }

      std::cout << "Epoch " << epoch << "/" << epochs << " Train loss: " << epoch_loss << "\n";

      losses.push_back(epoch_loss / static_cast<double>(dataset.size()));
    }

    return losses;
  }

  /**
   * @brief Runs a single forward pass, no training involved.
   *
   * @param network network to evaluate (not mutated).
   * @param x `[in_features]` input example.
   * @return Vector `[out_features]` network output for @p x.
   */
  template <typename T = autodiff::Scalar>
  typename FFNetwork<T>::Vector inference_step(FFNetwork<T>& network, const Vector& x)
  {
    return network.forward(x);
  }

  /**
   * @brief Runs `inference_step` over every example in @p dataset, pairing
   *   each prediction with the target it should be compared against.
   *
   * @param network network to evaluate (not mutated).
   * @param dataset examples to run inference on; must be non-empty.
   * @return std::vector<Prediction<T>> one (predicted, target) pair per
   *   example, in @p dataset's order.
   * @throws std::invalid_argument if @p dataset is empty.
   */
  template <typename T = autodiff::Scalar>
  std::vector<Prediction<T>> inference(FFNetwork<T>& network, const Dataset<T>& dataset)
  {
    if (dataset.empty()) {
      throw std::invalid_argument("inference: dataset must not be empty");
    }

    std::vector<Prediction<T>> predictions;
    predictions.reserve(dataset.size());

    for (const auto& example : dataset) {
      const auto& x = example.first;
      const auto& target = example.second;
      predictions.push_back({inference_step(network, x), target});
    }

    return predictions;
  }


}  // namespace nn
}  // namespace autodiff
