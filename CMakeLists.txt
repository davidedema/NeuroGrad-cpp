cmake_minimum_required(VERSION 3.16)
project(nn_autodiff CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# --- Eigen (header-only) -----------------------------------------------
# Fetched via CMake instead of relying on apt / brew / vcpkg, so every
# machine (yours, your teammate's, and CI) configures identically without
# needing an OS-specific package-manager step.
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://github.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# --- Catch2 (test framework) --------------------------------------------
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.4
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

# --- autodiff: our header-only library -----------------------------------
add_library(autodiff INTERFACE)
target_include_directories(autodiff INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(autodiff INTERFACE Eigen3::Eigen)
target_compile_features(autodiff INTERFACE cxx_std_17)

# Treat warnings seriously (helps catch AD bugs early) without being
# hostile across compilers.
if(MSVC)
  target_compile_options(autodiff INTERFACE /W4)
else()
  target_compile_options(autodiff INTERFACE -Wall -Wextra)
endif()

enable_testing()
include(CTest)
add_subdirectory(tests)

# apps/ will hold runnable examples (XOR demo, etc.) once we get there.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps/CMakeLists.txt)
  add_subdirectory(apps)
endif()
