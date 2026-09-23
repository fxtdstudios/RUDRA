# infer: InferenceBackend and its runtimes. Runtime headers (torch, onnxruntime)
# appear only in src/*_backend.cpp; the public headers stay runtime-free.
set(_sources src/tiler.cpp)
if(RUDRA_WITH_LIBTORCH)
  list(APPEND _sources src/libtorch_backend.cpp)
endif()
if(RUDRA_WITH_ONNXRUNTIME)
  list(APPEND _sources src/onnxruntime_backend.cpp)
endif()
rudra_library(rudra_infer SOURCES ${_sources})
target_link_libraries(rudra_infer PUBLIC rudra::rudra_core)

if(RUDRA_WITH_LIBTORCH)
  target_compile_definitions(rudra_infer PRIVATE RUDRA_HAVE_LIBTORCH)
  target_link_libraries(rudra_infer PRIVATE ${TORCH_LIBRARIES})
  # LibTorch's own headers are noisy under -Wconversion/-Wshadow.
  set_source_files_properties(src/libtorch_backend.cpp PROPERTIES COMPILE_OPTIONS
    "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-conversion;-Wno-shadow;-Wno-pedantic;-Wno-unused-parameter>")
endif()
if(RUDRA_WITH_ONNXRUNTIME)
  target_compile_definitions(rudra_infer PRIVATE RUDRA_HAVE_ONNXRUNTIME)
  target_link_libraries(rudra_infer PRIVATE onnxruntime::onnxruntime)
endif()
