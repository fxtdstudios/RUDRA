# cli: rudra-native. Core + infer only; must link without Qt.
add_executable(rudra-native main.cpp)
target_link_libraries(rudra-native PRIVATE rudra::rudra_infer nlohmann_json::nlohmann_json)
rudra_warnings(rudra-native)
