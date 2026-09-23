# app: the Qt application. The shell needs Qt 6.4+ Widgets; the QRhi viewer in
# render/ (Phase 2) needs Qt 6.6+ with Qt Shader Tools (RUDRA_BUILD_RENDER), and
# the release builds use Qt 6.8 LTS (ADR-002). Without it the shell builds
# alone, as the CI Qt shell job checks.
find_package(Qt6 6.4 REQUIRED COMPONENTS Widgets)
qt_standard_project_setup()
qt_add_executable(RUDRA main.cpp)
target_link_libraries(RUDRA PRIVATE Qt6::Widgets rudra::rudra_engine rudra::rudra_infer)
if(TARGET rudra_render_gpu)
  target_link_libraries(RUDRA PRIVATE rudra::rudra_render_gpu)
  target_compile_definitions(RUDRA PRIVATE RUDRA_APP_VIEWER)
endif()
rudra_warnings(RUDRA)
set_target_properties(RUDRA PROPERTIES WIN32_EXECUTABLE ON MACOSX_BUNDLE ON)
if(WIN32)
  target_compile_definitions(RUDRA PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
