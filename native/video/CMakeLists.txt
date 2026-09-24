# video: the video pipeline (Phase 4), rudra/video.py convert_video in C++.
# Above infer, media and deliver, which it joins; below engine, and free of Qt
# and the GPU, so rudra-native links it as well as the app.
rudra_library(rudra_video SOURCES src/predictor.cpp src/qc.cpp src/convert.cpp)
target_link_libraries(rudra_video PUBLIC rudra::rudra_infer rudra::rudra_media rudra::rudra_deliver
                      PRIVATE nlohmann_json::nlohmann_json)
