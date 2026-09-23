# Third-party dependencies.
#
# vcpkg (native/vcpkg.json) is the supported way to provide them: configure with
# -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake, or use a
# preset from CMakePresets.json. Without vcpkg the two header/test libraries the
# core needs are fetched at a pinned tag, so a clean machine can still build
# and test the core and the CLI.
include(FetchContent)

find_package(nlohmann_json 3.11 CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

if(RUDRA_BUILD_TESTS)
  find_package(GTest 1.14 CONFIG QUIET)
  if(NOT GTest_FOUND)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    # Pinned by tag. CI installs GoogleTest from the platform package manager
    # (apt, Homebrew, vcpkg), so this fallback only runs on a bare machine.
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.15.2
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(googletest)
  endif()
endif()

if(RUDRA_WITH_LIBTORCH)
  # LibTorch: the official archive, or the pip torch package
  # (python -c "import torch; print(torch.utils.cmake_prefix_path)").
  find_package(Torch REQUIRED CONFIG)
endif()

if(RUDRA_WITH_ONNXRUNTIME)
  if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "RUDRA_WITH_ONNXRUNTIME needs -DONNXRUNTIME_ROOT=<path to the ORT package>")
  endif()
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
            PATHS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime"
            NO_DEFAULT_PATH REQUIRED)
  find_library(ORT_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")
endif()
