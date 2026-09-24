# infer: InferenceBackend and its runtimes. Runtime headers (torch, onnxruntime)
# appear only in src/*_backend.cpp; the public headers stay runtime-free.
set(_sources src/tiler.cpp src/self_test.cpp)
if(RUDRA_WITH_LIBTORCH)
  list(APPEND _sources src/libtorch_backend.cpp)
endif()
if(RUDRA_WITH_ONNXRUNTIME)
  list(APPEND _sources src/onnxruntime_backend.cpp)
endif()
rudra_library(rudra_infer SOURCES ${_sources})
target_link_libraries(rudra_infer PUBLIC rudra::rudra_core PRIVATE nlohmann_json::nlohmann_json)

if(RUDRA_WITH_LIBTORCH)
  target_compile_definitions(rudra_infer PRIVATE RUDRA_HAVE_LIBTORCH)
  target_link_libraries(rudra_infer PRIVATE ${TORCH_LIBRARIES})
  # LibTorch's own headers are noisy under -Wconversion/-Wshadow.
  # On MSVC its headers need the permissive front end and are kept quiet.
  set_source_files_properties(src/libtorch_backend.cpp PROPERTIES COMPILE_OPTIONS
    "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wno-conversion;-Wno-shadow;-Wno-pedantic;-Wno-unused-parameter>;$<$<CXX_COMPILER_ID:MSVC>:/permissive;/external:W0;/wd4244;/wd4267;/wd4251;/wd4275;/wd4100>")
  if(CMAKE_DL_LIBS)
    target_link_libraries(rudra_infer PRIVATE ${CMAKE_DL_LIBS})
  endif()
endif()
if(RUDRA_WITH_ONNXRUNTIME)
  target_compile_definitions(rudra_infer PRIVATE RUDRA_HAVE_ONNXRUNTIME)
  target_link_libraries(rudra_infer PRIVATE onnxruntime::onnxruntime)
endif()
