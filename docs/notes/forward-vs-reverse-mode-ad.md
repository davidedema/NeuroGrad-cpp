# Forward-mode vs. reverse-mode automatic differentiation

Why the project starts with forward-mode dual numbers, and what changes if a
reverse-mode engine is added later. Useful material for the final
report/relazione.

## What a Dual number computes

`Dual(value, derivative)` implements **forward-mode AD**. Seeding
`Dual::variable(w)` (derivative = 1) and propagating it through a
computation yields, in the derivative of the output, the directional
derivative of that output with respect to the seeded variable only. A
separate pass is needed per variable you want the derivative of.

## Why this matters for NN training

The loss L is a single scalar but depends on all P weights of the network.
Getting the full gradient (dL/dw_1, ..., dL/dw_P) via scalar forward-mode
dual numbers requires **P separate forward passes** (one per weight, each
time seeding a different weight's derivative to 1 and all others to 0). For
a small XOR net (~17 params) this is cheap; it would not scale well to
networks with many more parameters.

## What "backpropagation" actually is

Backprop = **reverse-mode AD**: one forward pass (storing intermediate
values) followed by a single backward sweep that accumulates
dL/d(everything) via the chain rule. Cost: O(1) passes (one forward + one
backward), independent of the number of parameters P.

Rule of thumb: forward-mode is efficient for few-inputs/many-outputs
(Jacobian columns); reverse-mode is efficient for many-inputs/few-outputs
(gradient of a scalar loss w.r.t. many weights) — exactly the NN training
case. This is why real frameworks (PyTorch, TensorFlow, etc.) use
reverse-mode.

## The "multi-dual"/"jet" middle ground

Making the derivative a vector of size P lets you compute the whole
gradient in one traversal of the graph, but each elementary op then costs
O(P), so total work is still O(P·ops) — same asymptotic cost as P separate
scalar passes, just reorganized. Not a shortcut to reverse-mode's O(ops)
total cost.

## Practical takeaway used for this project

Forward-mode dual numbers are mathematically valid, much simpler to
implement, and trivial to cross-check against finite differences — a
legitimate, simpler scope for a small XOR-sized network. Reverse-mode
(tape-based autodiff / "real" backprop) is more implementation work but is
the asymptotically correct/standard approach and scales better, which
matters if the exercise is extended to a larger or physics-informed
network.

## Decisions made

- Linear algebra: Eigen (fetched via CMake, not a system package — see the
  root `README.md`).
- Autodiff engine: forward-mode (`Dual`) now, with the code organized (see
  `include/autodiff/Scalar.hh`) so a reverse-mode engine can be substituted
  later without rewriting the forward-pass/layer code.
- Network examples (XOR, physics-informed/PINN, etc.): deferred until the
  autodiff core (Dual + Eigen integration) is solid and tested.
