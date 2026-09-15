# One test executable per source file below, but they all link into a single
# binary (`unit_tests`) for now — simplest thing that works with Catch2's
# test discovery. Add a new `test_<ClassName>.cc` here whenever a new class
# is added under include/, mirroring its name 1:1 so it's easy to find the
# tests for a given class.
add_executable(unit_tests
  test_dual.cc
  test_eigen_integration.cc
  test_layer.cc
)

target_link_libraries(unit_tests PRIVATE autodiff Catch2::Catch2WithMain)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

include(Catch)
catch_discover_tests(unit_tests
  TEST_PREFIX ""
  PROPERTIES LABELS "unit"
)
