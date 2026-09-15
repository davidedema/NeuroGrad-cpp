#pragma once

#include <Eigen/Core>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn/Layer.hh"

/**
 * @file FFNetwork.hh
 * @brief A feed-forward network: an ordered chain of `Layer`s.
 */

namespace autodiff {
namespace nn {

/**
 * @brief Chains an ordered sequence of `Layer`s into a feed-forward network.
 *
 * Each layer may have its own input/output dimensions and activation; the
 * output of one layer is fed as the input to the next, so `FFNetwork` is
 * what actually lets consecutive layers have different shapes (unlike a
 * single `Layer` on its own).
 *
 * @tparam T scalar type used by every layer in the chain (defaults to
 *   `autodiff::Scalar`, the current forward/reverse-mode swap point).
 */
template <typename T = autodiff::Scalar>
class FFNetwork{
    public:

        //! Underlying scalar type.
        using Scalar = T;
        //! Column vector of Scalar, dynamic size.
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        //! Ordered sequence of layers making up the network.
        using Layers = std::vector<Layer<Scalar>>;

        /**
         * @brief Builds a network from an ordered sequence of layers.
         *
         * @param layers layers in evaluation order; layer `i`'s output size
         *   must match layer `i+1`'s expected input size (checked by
         *   `Layer::forward()` at call time, not by this constructor).
         */
        FFNetwork(Layers layers);

        /**
         * @brief Computes the network's output by chaining each layer's
         *   forward pass in order.
         *
         * @param x `[in_features]` input vector, matching the first layer's
         *   expected input size.
         * @return Vector `[out_features]` output of the last layer.
         * @throws std::invalid_argument if any layer receives an input of
         *   the wrong size (see `Layer::forward()`).
         */
        [[nodiscard]] Vector forward(const Vector& x) const;

    private:
        Layers layers_;

};

template <typename T>
FFNetwork<T>::FFNetwork(Layers layers) : layers_(std::move(layers)) {}

template <typename T>
typename FFNetwork<T>::Vector FFNetwork<T>::forward(const Vector& x) const {

    Vector out = x;

    for (const auto &l : layers_)
    {
        out = l.forward(out);
    }

    return out;

}

// namespace nn
}
// namespace autodiff
}