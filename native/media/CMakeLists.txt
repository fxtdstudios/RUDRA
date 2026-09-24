# media: readers. Sequence open (ui/sequence.py) always; still decode
# (rudra/decode.py) on OpenCV imgcodecs, the decoder the Python uses, so a frame
# decodes to the same floats in both. Video (rudra/video.py) runs ffprobe and
# ffmpeg as programs, as the Python does.
set(_media_sources src/sequence.cpp src/video_probe.cpp src/video_decode.cpp)
if(RUDRA_WITH_OPENCV)
  list(APPEND _media_sources src/still.cpp src/png16.cpp)
endif()
rudra_library(rudra_media SOURCES ${_media_sources})
target_link_libraries(rudra_media PUBLIC rudra::rudra_core PRIVATE nlohmann_json::nlohmann_json)
if(RUDRA_WITH_OPENCV)
  find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
  target_link_libraries(rudra_media PRIVATE ${OpenCV_LIBS})
  target_include_directories(rudra_media SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(rudra_media PUBLIC RUDRA_HAVE_STILL_DECODE)
endif()
