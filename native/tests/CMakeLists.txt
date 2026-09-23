add_executable(rudra_tests test_platform.cpp test_core.cpp test_composite.cpp test_delivery.cpp test_engine.cpp)
target_link_libraries(rudra_tests PRIVATE rudra::rudra_core rudra::rudra_deliver rudra::rudra_engine
                      nlohmann_json::nlohmann_json GTest::gtest_main)
target_compile_definitions(rudra_tests PRIVATE RUDRA_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
rudra_warnings(rudra_tests)
if(RUDRA_WITH_OPENCV)
  target_sources(rudra_tests PRIVATE test_decode.cpp)
  target_link_libraries(rudra_tests PRIVATE rudra::rudra_media)
endif()

include(GoogleTest)
gtest_discover_tests(rudra_tests)

# Gate A in C++: point at an exported package to run it as a test
# (cmake -DRUDRA_TEST_PACKAGE=dist/models/sdr2hdr_shadow_v1).
set(RUDRA_TEST_PACKAGE "" CACHE PATH "Model package for the golden parity test")
if(RUDRA_TEST_PACKAGE AND RUDRA_BUILD_CLI)
  add_test(NAME model_package_parity COMMAND rudra-native diff "${RUDRA_TEST_PACKAGE}" --runtime all)
endif()
