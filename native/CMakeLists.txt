cmake_minimum_required(VERSION 3.24)

# RUDRA native: the layered C++ app described in docs/NATIVE_ARCHITECTURE.md.
# The dependency rule (a layer links only to layers below it) is enforced by
# PRIVATE linkage here and by tools/check_layers.py in CI.
project(RudraNative VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RUDRA_BUILD_TESTS       "Build the GoogleTest suite"                        ON)
option(RUDRA_BUILD_CLI         "Build the headless rudra-native CLI"              ON)
option(RUDRA_BUILD_APP         "Build the Qt application (needs Qt 6)"            OFF)
option(RUDRA_BUILD_HDR_PROBE   "Build rudra-hdr-probe, Gate B (needs Qt 6.6+)"    OFF)
option(RUDRA_WITH_LIBTORCH     "Build the LibTorch inference backend"             OFF)
option(RUDRA_WITH_ONNXRUNTIME  "Build the ONNX Runtime inference backend"         OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime package root (include/, lib/)")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(RudraDeps)
include(RudraWarnings)

add_subdirectory(platform)
add_subdirectory(core)
add_subdirectory(infer)
add_subdirectory(media)
add_subdirectory(render)
add_subdirectory(deliver)
add_subdirectory(engine)

if(RUDRA_BUILD_CLI)
  add_subdirectory(cli)
endif()
if(RUDRA_BUILD_APP)
  add_subdirectory(app)
endif()
if(RUDRA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "RUDRA native: tests=${RUDRA_BUILD_TESTS} cli=${RUDRA_BUILD_CLI} app=${RUDRA_BUILD_APP} hdr_probe=${RUDRA_BUILD_HDR_PROBE} "
               "libtorch=${RUDRA_WITH_LIBTORCH} onnxruntime=${RUDRA_WITH_ONNXRUNTIME}")
