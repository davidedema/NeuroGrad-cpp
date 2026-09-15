# NeuroGrad-cpp
C++ implementation of automatic differentiation for a neural network's
gradient, used to train weights via gradient descent (level 3 assignment).

Forward-mode AD (dual numbers) first; the code is organized so a
reverse-mode engine can be substituted later without rewriting the
forward-pass / layer code. See `docs/notes/` for the full design rationale
and the staged implementation plan — read that before writing code.

## Project layout

```
include/autodiff/
  Dual.hh          forward-mode AD scalar type (Stage 1)
  Scalar.hh        the "swap point" alias used by all NN code (Stage 3)
  EigenSupport.hh  Eigen integration for Dual (Stage 4)
  # Var.hh         reverse-mode engine, added later

include/nn/        Layer / NeuralNetwork / losses / optimizer (later stages,
                    not started yet)

tests/
  utils/finite_diff.hh   shared finite-difference helpers — reuse these for
                          every new class's derivative checks, don't
                          reimplement finite differences locally
  test_dual.cc            one file per class, named test_<ClassName>.cc
  test_eigen_integration.cc  Eigen + Dual matmul/derivative check (Stage 4)

apps/               runnable examples (XOR demo, etc.) — added later

docs/notes/         design decisions and a learning log, one entry per stage
```

## Building and testing

Requires CMake >= 3.16 and a C++17 compiler. Eigen and Catch2 are fetched
automatically by CMake (see `CMakeLists.txt`) — no manual dependency
installation needed on Linux, macOS, or Windows.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Right now `Dual.hh` fixes the interface and structure only (a templated
`Dual<T>` class, bodies marked `TODO`), so `ctest` is expected to report
failures — that's the starting point, not a bug. Fill in `Dual.hh` one
function at a time and watch the corresponding test in
`tests/test_dual.cc` turn green.

To build only the test binary (skips `apps/` once that target exists):

```sh
cmake --build build --target unit_tests --parallel
```

The test executable itself is Catch2-based, so it can also be run directly
for full output or to filter by tag (see "Testing convention" below):

```sh
./build/tests/unit_tests                 # all tests, verbose output
./build/tests/unit_tests "[Dual]"        # only the Dual test cases
./build/tests/unit_tests "[finite-diff]" # only the numerical derivative checks
```

## Testing convention

- One test file per class, named `test_<ClassName>.cc`, mirroring the
  header it tests (`include/autodiff/Dual.hh` -> `tests/test_dual.cc`).
- Tag test cases with the class name (e.g. `[Dual]`) and `[finite-diff]`
  for anything checked numerically, so subsets can be run with
  `ctest -L <label>` or Catch2's own `--tags` filter as the suite grows.
- Any derivative check against a numerical reference goes through
  `tests/utils/finite_diff.hh` (`central_difference`,
  `central_difference_gradient`, `is_close`) rather than reimplementing the
  finite-difference logic per test file. When the whole-network gradient
  check is added (roadmap Stage 8), it reuses
  `central_difference_gradient` — same helper, same tolerance convention.

## Continuous integration

`.github/workflows/ci.yml` builds and runs the test suite on Ubuntu,
macOS, and Windows on every push/PR to `main`. See `CONTRIBUTING.md` for
the branch/PR workflow this is meant to gate.
