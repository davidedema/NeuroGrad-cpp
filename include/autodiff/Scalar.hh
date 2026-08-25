#pragma once

#include "autodiff/Dual.hh"
// #include "autodiff/Var.hh"  // reverse-mode engine, added in a later stage

// The single "swap point" for the whole project (roadmap Stage 3).
//
// Any NN code (Layer, NeuralNetwork, loss functions, ...) written from here
// on should use autodiff::Scalar rather than autodiff::forward::Dual
// directly. When a reverse-mode autodiff::reverse::Var type exists, trying
// it out for the elementwise math is a one-line change here.
//
// Caveat (documented so it doesn't feel like a broken promise later):
// reverse-mode needs an explicit tape and a `.backward()` call that
// forward-mode doesn't have, so this alias makes the forward-pass / layer
// code reusable across engines — it does NOT make the training loop itself
// 100% engine-agnostic without any changes.

namespace autodiff {
using Scalar = forward::Dual;
}
