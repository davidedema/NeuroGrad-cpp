# apps

Runnable example programs go here. `xor_demo.cc` (roadmap Stage 10) is
scaffolded — structure and includes fixed, `main()`'s body left as TODOs to
fill in one at a time (build the network, build the XOR dataset, train,
report results), same pattern `Dual.hh`/`Layer.hh` used for their own
stages.

`apps/CMakeLists.txt` wires it up with `add_executable(xor_demo
xor_demo.cc)`; the root `CMakeLists.txt` picks it up automatically
(`add_subdirectory(apps)` is conditional on that file existing).
