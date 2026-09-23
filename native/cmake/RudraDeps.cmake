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

set(RUDRA_TORCH_ROOT "" CACHE PATH
    "LibTorch or pip torch folder (include/, lib/), imported without TorchConfig")
if(RUDRA_WITH_LIBTORCH)
  if(RUDRA_TORCH_ROOT)
    # TorchConfig of a CUDA build calls enable_language(CUDA): it needs the
    # CUDA toolkit, at torch's exact version, integrated with this compiler,
    # to build a program that never compiles a line of CUDA. The runtime does
    # not need any of it, so the libraries are imported directly here and the
    # CUDA library is loaded on demand (infer/src/libtorch_backend.cpp).
    find_path(RUDRA_TORCH_INCLUDE torch/script.h
              PATHS "${RUDRA_TORCH_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
    set(_torch_libs "")
    foreach(_lib torch torch_cpu c10)
      find_library(RUDRA_TORCH_LIB_${_lib} NAMES ${_lib} PATHS "${RUDRA_TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
      list(APPEND _torch_libs "${RUDRA_TORCH_LIB_${_lib}}")
    endforeach()
    add_library(rudra_torch INTERFACE)
    target_include_directories(rudra_torch SYSTEM INTERFACE
      "${RUDRA_TORCH_INCLUDE}" "${RUDRA_TORCH_INCLUDE}/torch/csrc/api/include")
    target_link_libraries(rudra_torch INTERFACE ${_torch_libs})
    if(NOT MSVC)
      set(RUDRA_TORCH_CXX11_ABI 1 CACHE STRING "torch.compiled_with_cxx11_abi() of that torch")
      target_compile_definitions(rudra_torch INTERFACE _GLIBCXX_USE_CXX11_ABI=${RUDRA_TORCH_CXX11_ABI})
    endif()
    set(TORCH_LIBRARIES rudra_torch)
    message(STATUS "LibTorch imported from ${RUDRA_TORCH_ROOT} (no TorchConfig, no CUDA toolkit)")
  else()
    # LibTorch: the official archive, or the pip torch package
    # (python -c "import torch; print(torch.utils.cmake_prefix_path)").
    find_package(Torch REQUIRED CONFIG)
  endif()
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
