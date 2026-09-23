# cli: rudra-native. Core, infer, media and deliver; must link without Qt.
add_executable(rudra-native main.cpp)
target_link_libraries(rudra-native PRIVATE rudra::rudra_infer rudra::rudra_deliver nlohmann_json::nlohmann_json)
if(RUDRA_WITH_OPENCV)
  # `master` and `master-check` need still decode.
  target_sources(rudra-native PRIVATE master.cpp)
  target_link_libraries(rudra-native PRIVATE rudra::rudra_media)
endif()
rudra_warnings(rudra-native)
