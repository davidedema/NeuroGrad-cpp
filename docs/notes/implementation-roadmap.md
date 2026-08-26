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

Status: scaffolded (`include/autodiff/Dual.hh` — interface and structure
fixed, bodies `TODO`), tests written (`tests/test_dual.cc`, currently red).
`Dual` is now `template <typename T> class Dual`
(`autodiff::forward::Dual<T>`, structured per
`docs/notes/template-style-reference.hh`'s style conventions — Doxygen
comments, compound-assignment ops first) rather than the
originally-scaffolded non-templated type; see the Stage 3 update below for
why. `tests/test_dual.cc` uses `Dual<double>`.

## Stage 2 — validate `Dual.hh` standalone

Before touching network code: for each elementary function, compare the
Dual-computed derivative against a central finite difference
`(f(x+h) - f(x-h)) / (2h)` on plain doubles, `h ~ 1e-6`, checking relative
error against ~`1e-6`. Cheapest place to catch a sign error.

Status: covered by `tests/test_dual.cc` + `tests/utils/finite_diff.hh`.

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

Status: not started (`include/autodiff/EigenSupport.hh` is a placeholder).

## Stage 5 — Layer and forward pass

A `Layer` holding `Eigen::Matrix<Scalar,...> W` and `Eigen::Vector<Scalar,...>
b`, with `forward(x) = activation(W*x + b)`. Template on the activation
function (function pointer or small enum/switch) so tanh/sigmoid/relu can be
swapped without rewriting the layer.

## Stage 6 — the gradient loop (forward-mode's O(P) mechanic)

For each parameter `p_i` (every weight and bias entry): zero every
parameter's derivative, seed only `p_i` to 1, run the full forward pass to
get the scalar loss as a `Dual`, read `loss.derivative()` as `dL/dp_i`. Loop
over all `P` parameters. Slow but mechanically transparent — the right
first version.

## Stage 7 — gradient descent + training loop

`w -= lr * grad` per parameter, looped over epochs, tracking loss.

## Stage 8 — end-to-end validation before trusting training

Before training on XOR, cross-check the Stage 6 gradient against a second,
independent finite-difference gradient (perturb each weight +-h on a
plain-`double` copy of the network, recompute loss, central-difference it —
this reuses `testutil::central_difference_gradient`). Agreement to ~1e-6
across all parameters means the training loop can be trusted; a mismatch
catches a bug before it looks like a convergence problem.

## Stage 9 — XOR

First real training run. If loss doesn't drop, the bug is almost certainly
upstream in Stage 6/8, not the optimizer.

## Deferred to later

- Swap in the reverse-mode engine behind the `Scalar` alias.
- Physics-informed network example (e.g. small PINN solving an ODE, needs
  derivatives of network output w.r.t. input as well as w.r.t. weights).
