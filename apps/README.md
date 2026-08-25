# apps

Runnable example programs go here later (e.g. `xor_demo.cc` training a
small network on XOR — roadmap Stage 9). Empty for now; deferred until the
autodiff core and Layer/NeuralNetwork classes are in place.

Once there's at least one `.cc` file here, add an `apps/CMakeLists.txt`
with an `add_executable(...)` for it — the root `CMakeLists.txt` already
picks it up automatically (`add_subdirectory(apps)` is conditional on that
file existing).
