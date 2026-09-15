# Implementation roadmap — NN gradient via autodiff (level 3)

Build the smallest working piece, verify it against a known-correct
reference (finite differences), then move up one layer — never stack a new
abstraction on something unverified.

## Stage 1 — `Dual.hh` (forward-mode scalar type)

- Two constructors matter: `Dual::variable(v)` seeds `derivative = 1` (seed
  whichever weight you're currently differentiating w.r.t.), and
  `Dual::constant(v)` seeds `derivative = 0` (everything else — inputs,
  other weights, targets). Also keep an implicit `Dual(double v)`
  constructor defaulting derivative to 0 so plain literals work in
  expressions.
- In `operator/=` and `operator*=`, compute the new derivative *before*
  overwriting the real part — easy off-by-order bug (e.g.
  `d_ = d_*o.v_ + v_*o.d_;` must happen before `v_ *= o.v_;` is applied).
- Elementary functions: `exp`, `log`, `sqrt`, `pow(x, constant)`, `sin`,
  `cos`, `tanh`, plus NN-specific `sigmoid` and `relu`. For `relu`, pick and
  document a subgradient convention at exactly 0 (0 is the conventional
  choice).
- Comparison operators (`<`, `>`, `==`, ...) compare only the real part —
  needed for `relu`'s branch and later for Eigen's internals.

Status: done. `include/autodiff/Dual.hh` implements the compound-assignment
operators, unary negation, value-only comparisons, and all listed
elementary/NN-specific functions; `tests/test_dual.cc` is green. `Dual` is
`template <typename T> class Dual` (`autodiff::forward::Dual<T>`,
structured per `docs/notes/template-style-reference.hh`'s style
conventions — Doxygen comments, compound-assignment ops first) rather than
the originally-scaffolded non-templated type; see the Stage 3 update below
for why. `tests/test_dual.cc` uses `Dual<double>`.

## Stage 2 — validate `Dual.hh` standalone

Before touching network code: for each elementary function, compare the
Dual-computed derivative against a central finite difference
`(f(x+h) - f(x-h)) / (2h)` on plain doubles, `h ~ 1e-6`, checking relative
error against ~`1e-6`. Cheapest place to catch a sign error.

Status: covered by `tests/test_dual.cc` + `tests/utils/finite_diff.hh` — each
elementary function is checked individually, plus one composite-expression
test that chains several of them together to catch composition bugs that a
per-function check alone would miss.

## Stage 3 — the "swap point" architecture (forward -> reverse mode later)

`Dual<T>` lives in `namespace autodiff::forward` (templated on the scalar
coefficient type, per the Stage 1 update above).
`include/autodiff/Scalar.hh` contains only
`namespace autodiff { using Scalar = forward::Dual<double>; }`. All NN
code from Stage 4 onward should be templated on a scalar type defaulting to
`autodiff::Scalar` (or just use the alias directly). When a reverse-mode
`Var` type is later built in `namespace autodiff::reverse`, switching
engines is a one-line change to that alias — **for the elementwise math
only**. Reverse-mode needs an explicit `.backward()` call and a tape, which
forward-mode doesn't have, so the swap point buys reuse of the
forward-pass/layer code, not a fully engine-agnostic training loop.

Status: done (`include/autodiff/Scalar.hh`).

## Stage 4 — Eigen integration for `Dual`

Before writing a Layer class, get `Eigen::Matrix<Dual, Dynamic, Dynamic>`
compiling and doing a matmul. Needs an `Eigen::NumTraits<Dual>`
specialization plus `abs`, `abs2`, `sqrt` free functions findable via ADL in
`autodiff::forward`. Isolate this step — get a 2x2 matrix-vector multiply of
`Dual`s working and printed before building anything on top of it.

Status: done. `include/autodiff/EigenSupport.hh` specializes
`Eigen::NumTraits<Dual<double>>`; the `abs`, `abs2`, `sqrt` ADL functions
live in `Dual.hh` alongside the other elementary functions (`abs`
intentionally returns derivative 0 — see the comment there — since Eigen
only calls it for magnitude comparisons/pivoting, never as part of a
differentiated computation). `tests/test_eigen_integration.cc` multiplies a
2x2 `Dual<double>` matrix by a vector and checks both the value and the
derivative against a hand-computed expectation.

## Stage 5 — Layer and forward pass

A `Layer` holding `Eigen::Matrix<Scalar,...> W` and `Eigen::Vector<Scalar,...>
b`, with `forward(x) = activation(W*x + b)`. Template on the activation
function (function pointer or small enum/switch) so tanh/sigmoid/relu can be
swapped without rewriting the layer.

Status: done. `include/nn/Layer.hh` (`template <typename T = autodiff::Scalar>
class Layer`) implements `forward(x)`: checks `x.size()` against the weight
matrix's column count and throws `std::invalid_argument` (with the actual
vs. expected size and the weight matrix's shape) on a mismatch, then
computes `(W*x + b).unaryExpr(activation)`, with `activation` passed in as
`std::function<Scalar(const Scalar&)>` rather than hardcoded so
identity/relu/sigmoid/tanh can be swapped freely. `tests/test_layer.cc`
covers: the affine part's value and derivative (identity activation), relu
zeroing both value and derivative on a negative row while passing a
positive row through unchanged, and sigmoid's derivative checked against
`testutil::central_difference` on the equivalent plain-double composite
function (the same finite-diff-vs-autodiff pattern `Dual.hh`'s tests use).

## Stage 6 — NeuralNetwork composition and a loss function

Before the gradient loop (below) can run a "full forward pass" to get a
scalar loss, two pieces are still missing: something that chains multiple
`Layer`s together, and something that reduces a network's output + target
into a single differentiable scalar.

- `NeuralNetwork` (built as `include/nn/FFNetwork.hh`): holds an ordered
  sequence of `Layer<Scalar>` (`std::vector<Layer<Scalar>>`); `forward(x)`
  pipes the input through each layer in turn (one layer's output is the
  next layer's input). Templated on the scalar type like `Layer`, following
  the same structure/Doxygen conventions (`docs/notes/template-style-reference.hh`).
- A loss function for XOR (mean-squared-error): a free
  `template <typename T> T mse(...)`-style function over the network's
  output and the target, reducing to a single `Scalar` — no class/state
  needed, matches the "elementary function" pattern already used for
  `sigmoid`/`relu` etc. in `Dual.hh`.
- Tests (`tests/test_FFNetwork.cc`): a small hand-computed forward pass
  through 2 chained layers with different in/out dimensions (2 -> 3 -> 1,
  the actual reason `FFNetwork` exists rather than a single `Layer`), a
  derivative-propagation check through that chain, and a check that relu's
  derivative-zeroing convention survives being composed across two layers
  (mirrors `test_layer.cc`'s identity/relu style). The loss function still
  needs its own tests once it's written, following the finite-diff-vs-autodiff
  pattern used for sigmoid in `test_layer.cc`.

Status: `FFNetwork` done and tested; the loss function is still outstanding
— needed before Stage 7's gradient loop, which reads `loss.derivative()`
from a full forward pass.

## Stage 7 — the gradient loop (forward-mode's O(P) mechanic)

For each parameter `p_i` (every weight and bias entry): zero every
parameter's derivative, seed only `p_i` to 1, run the full forward pass to
get the scalar loss as a `Dual`, read `loss.derivative()` as `dL/dp_i`. Loop
over all `P` parameters. Slow but mechanically transparent — the right
first version.

## Stage 8 — gradient descent + training loop

`w -= lr * grad` per parameter, looped over epochs, tracking loss.

## Stage 9 — end-to-end validation before trusting training

Before training on XOR, cross-check the Stage 7 gradient against a second,
independent finite-difference gradient (perturb each weight +-h on a
plain-`double` copy of the network, recompute loss, central-difference it —
this reuses `testutil::central_difference_gradient`). Agreement to ~1e-6
across all parameters means the training loop can be trusted; a mismatch
catches a bug before it looks like a convergence problem.

## Stage 10 — XOR

First real training run. If loss doesn't drop, the bug is almost certainly
upstream in Stage 7/9, not the optimizer.

## Deferred to later

- Swap in the reverse-mode engine behind the `Scalar` alias.
- Physics-informed network example (e.g. small PINN solving an ODE, needs
  derivatives of network output w.r.t. input as well as w.r.t. weights).
