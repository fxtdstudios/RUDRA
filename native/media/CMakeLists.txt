# media: readers and writers. Still decode (rudra/decode.py) is here now, on
# OpenCV imgcodecs, the decoder the Python uses, so a frame decodes to the same
# floats in both. Video (libav) arrives in Phase 4.
if(RUDRA_WITH_OPENCV)
  find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
  rudra_library(rudra_media SOURCES src/still.cpp)
  target_link_libraries(rudra_media PUBLIC rudra::rudra_core PRIVATE ${OpenCV_LIBS})
  target_include_directories(rudra_media SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(rudra_media PUBLIC RUDRA_HAVE_STILL_DECODE)
else()
  rudra_library(rudra_media INTERFACE_ONLY)
  target_link_libraries(rudra_media INTERFACE rudra::rudra_core)
endif()
