add_executable(rudra_tests test_platform.cpp test_core.cpp test_composite.cpp test_delivery.cpp test_qc.cpp test_queue.cpp test_sequence.cpp test_viewer.cpp test_viewport.cpp test_engine.cpp test_theme.cpp test_actions.cpp test_session.cpp test_scope_draw.cpp test_readouts.cpp test_render_plan.cpp test_master_job.cpp test_model_catalog.cpp test_copy_texts.cpp test_video.cpp test_video_predict.cpp test_video_encode.cpp test_video_qc.cpp test_sequence_encode.cpp test_ffmpeg_check.cpp)
target_link_libraries(rudra_tests PRIVATE rudra::rudra_core rudra::rudra_deliver rudra::rudra_media rudra::rudra_video rudra::rudra_engine
                      nlohmann_json::nlohmann_json GTest::gtest_main)
target_compile_definitions(rudra_tests PRIVATE RUDRA_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
                           RUDRA_THEME_DIR="${RUDRA_THEME_DIR}" RUDRA_THEME_CSS="${RUDRA_THEME_CSS}"
                           RUDRA_APP_DIR="${PROJECT_SOURCE_DIR}/app" RUDRA_CMAKE_COMMAND="${CMAKE_COMMAND}")
add_dependencies(rudra_tests rudra_theme)
if(RUDRA_BUILD_CLI)   # test_sequence_encode runs `rudra-native deliver`
  add_dependencies(rudra_tests rudra-native)
  target_compile_definitions(rudra_tests PRIVATE RUDRA_CLI_PATH="$<TARGET_FILE:rudra-native>")
endif()   # test_theme reads the generated sheet
rudra_warnings(rudra_tests)
if(RUDRA_WITH_OPENCV)
  target_sources(rudra_tests PRIVATE test_decode.cpp test_video_master.cpp)
endif()

include(GoogleTest)
# Listed when ctest runs, not after the link: on Windows the runtimes' DLLs are
# copied beside the executable after the build, so running it at link time fails.
gtest_discover_tests(rudra_tests DISCOVERY_MODE PRE_TEST)

# Gate A in C++: point at an exported package to run it as a test
# (cmake -DRUDRA_TEST_PACKAGE=dist/models/sdr2hdr_shadow_v1).
set(RUDRA_TEST_PACKAGE "" CACHE PATH "Model package for the golden parity test")
if(RUDRA_TEST_PACKAGE AND RUDRA_BUILD_CLI)
  add_test(NAME model_package_parity COMMAND rudra-native diff "${RUDRA_TEST_PACKAGE}" --runtime all)
  if(RUDRA_WITH_OPENCV)
    # Phase 2 step 10: the frame path with the real model over real images.
    target_sources(rudra_tests PRIVATE test_frame_engine_real.cpp test_video_predict_real.cpp test_video_convert_real.cpp test_video_queue_real.cpp)
    target_link_libraries(rudra_tests PRIVATE rudra::rudra_infer)
    target_compile_definitions(rudra_tests PRIVATE RUDRA_TEST_PACKAGE_DIR="${RUDRA_TEST_PACKAGE}")
    # Phase 1 step 7: a Studio-identical master with no Python.
    add_test(NAME master_parity COMMAND rudra-native master-check "${RUDRA_TEST_PACKAGE}"
             "${CMAKE_CURRENT_SOURCE_DIR}/golden/master")
  endif()
endif()
