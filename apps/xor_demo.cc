#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nn/FFNetwork.hh"
#include "nn/Layer.hh"
#include "nn/Training.hh"

// Stage 10 (roadmap): train a small network on XOR and confirm the loss
// actually drops. If it doesn't, the bug is almost certainly upstream —
// Stage 7's gradient or Stage 9's cross-check — not this training loop.
//
// This file fixes the structure only; fill in each TODO below one at a
// time, same pattern Dual.hh/Layer.hh used for their own stages.

using autodiff::Scalar;
using autodiff::nn::Dataset;
using autodiff::nn::FFNetwork;
using autodiff::nn::inference;
using autodiff::nn::Layer;
using autodiff::nn::train;

namespace {
Scalar sigmoid_activation(const Scalar& x) { return autodiff::forward::sigmoid(x); }

void print_weights(const FFNetwork<Scalar>& network) {
  const auto& layers = network.layers();
  for (std::size_t l = 0; l < layers.size(); ++l) {
    const auto& weights = layers[l].weights();
    const auto& bias = layers[l].bias();

    std::cout << "Layer " << l << " weights (" << weights.rows() << "x"
               << weights.cols() << "):\n";
    for (Eigen::Index r = 0; r < weights.rows(); ++r) {
      for (Eigen::Index c = 0; c < weights.cols(); ++c) {
        std::cout << weights(r, c).value() << " ";
      }
      std::cout << "\n";
    }

    std::cout << "Layer " << l << " bias: ";
    for (Eigen::Index i = 0; i < bias.size(); ++i) {
      std::cout << bias(i).value() << " ";
    }
    std::cout << "\n";
  }
}

// Plain-text format, one layer after another: "rows cols", then the
// weight matrix row-major, then the bias vector. load_weights() below
// reads this back and checks each layer's shape still matches.
void save_weights(const FFNetwork<Scalar>& network, const std::string& path) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("save_weights: could not open " + path + " for writing");
  }
  out << std::setprecision(17);

  const auto& layers = network.layers();
  out << layers.size() << "\n";
  for (const auto& layer : layers) {
    const auto& weights = layer.weights();
    const auto& bias = layer.bias();

    out << weights.rows() << " " << weights.cols() << "\n";
    for (Eigen::Index r = 0; r < weights.rows(); ++r) {
      for (Eigen::Index c = 0; c < weights.cols(); ++c) {
        out << weights(r, c).value() << " ";
      }
      out << "\n";
    }
    for (Eigen::Index i = 0; i < bias.size(); ++i) {
      out << bias(i).value() << " ";
    }
    out << "\n";
  }
}

void load_weights(FFNetwork<Scalar>& network, const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("load_weights: could not open " + path + " for reading");
  }

  auto& layers = network.layers();
  std::size_t num_layers = 0;
  in >> num_layers;
  if (num_layers != layers.size()) {
    throw std::invalid_argument(
        "load_weights: file has " + std::to_string(num_layers) +
        " layers, network has " + std::to_string(layers.size()));
  }

  for (auto& layer : layers) {
    auto& weights = layer.weights();
    auto& bias = layer.bias();

    Eigen::Index rows = 0, cols = 0;
    in >> rows >> cols;
    if (rows != weights.rows() || cols != weights.cols()) {
      throw std::invalid_argument("load_weights: layer shape mismatch");
    }

    for (Eigen::Index r = 0; r < rows; ++r) {
      for (Eigen::Index c = 0; c < cols; ++c) {
        double value = 0.0;
        in >> value;
        weights(r, c) = Scalar(value);
      }
    }
    for (Eigen::Index i = 0; i < bias.size(); ++i) {
      double value = 0.0;
      in >> value;
      bias(i) = Scalar(value);
    }
  }
}
}  // namespace

int main() {

  int hidden = 2;

  // Values are a function of the row/column index rather than hardcoded
  // literals, so they stay symmetry-breaking (no two hidden units start
  // identical) for whatever `hidden` is set to, not just 5.
  Layer<Scalar>::Matrix W1(hidden, 2);
  Layer<Scalar>::Vector b1(hidden);
  for (int r = 0; r < hidden; ++r) {
    W1(r, 0) = Scalar(0.5 - 0.2 * r);
    W1(r, 1) = Scalar(-0.3 + 0.25 * r);
    b1(r) = Scalar(0.1 - 0.075 * r);
  }
  Layer<Scalar> layer1(W1, b1, sigmoid_activation);

  Layer<Scalar>::Matrix W2(1, hidden);
  for (int c = 0; c < hidden; ++c) {
    W2(0, c) = Scalar(0.6 - 0.2 * c);
  }
  Layer<Scalar>::Vector b2(1);
  b2 << Scalar(0.1);
  Layer<Scalar> layer2(W2, b2, sigmoid_activation);

  FFNetwork<Scalar> net({layer1, layer2});

  FFNetwork<Scalar>::Vector x00(2);                                                                                                                                          
  x00 << Scalar::constant(0.0), Scalar::constant(0.0);                                                                                                                       
  FFNetwork<Scalar>::Vector x01(2);                                                                                                                                          
  x01 << Scalar::constant(0.0), Scalar::constant(1.0);                                                                                                                       
  FFNetwork<Scalar>::Vector x10(2);                                                                                                                                          
  x10 << Scalar::constant(1.0), Scalar::constant(0.0);                                                                                                                       
  FFNetwork<Scalar>::Vector x11(2);                                                                                                                                          
  x11 << Scalar::constant(1.0), Scalar::constant(1.0);                                                                                                                       
  FFNetwork<Scalar>::Vector t0(1);                                                                                                                                           
  t0 << Scalar::constant(0.0);                                                                                                                                               
  FFNetwork<Scalar>::Vector t1(1);                                                                                                                                           
  t1 << Scalar::constant(1.0);                                                                                                                                               

  Dataset<Scalar> data = {{x00, t0}, {x01, t1}, {x10, t1}, {x11, t0}}; 

  double learning_rate = 0.1;
  int epochs = 100000;
  // load_weights(net, "xor_weights.txt");
  auto losses = train(net, data, learning_rate, epochs);

  auto predictions = inference(net, data);
  for (const auto& [predicted, target] : predictions) {
    std::cout << "predicted: " << predicted(0).value()
               << "  target: " << target(0).value() << "\n";
  }

  // print_weights(net);
  // save_weights(net, "xor_weights.txt");
  // To reload later: load_weights(net, "xor_weights.txt");

  return 0;
}
