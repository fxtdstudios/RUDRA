# cli: rudra-native. Core, infer, media and deliver; must link without Qt.
add_executable(rudra-native main.cpp deliver.cpp ffmpeg_check.cpp)
target_link_libraries(rudra-native PRIVATE rudra::rudra_infer rudra::rudra_deliver rudra::rudra_video nlohmann_json::nlohmann_json)
if(RUDRA_WITH_OPENCV)
  # `master`, `master-check` and `video` need still decode (the spool is PNG).
  target_sources(rudra-native PRIVATE master.cpp video.cpp batch.cpp)
  target_link_libraries(rudra-native PRIVATE rudra::rudra_media)
endif()
rudra_warnings(rudra-native)
