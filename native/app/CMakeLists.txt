# app: the Qt application. The shell needs Qt 6.4+ Widgets; the QRhi viewer in
# render/ (Phase 2) needs Qt 6.6+ with Qt Shader Tools (RUDRA_BUILD_RENDER), and
# the release builds use Qt 6.8 LTS (ADR-002). Without it the shell builds
# alone, as the CI Qt shell job checks.
find_package(Qt6 6.4 REQUIRED COMPONENTS Widgets)
qt_standard_project_setup()
qt_add_executable(RUDRA main.cpp theme.cpp theme.hpp)
# The look (Phase 3 step 1): Plex (OFL) and the sheet generated from
# ui/theme.css by the rudra_theme target (cmake/RudraTheme.cmake).
add_dependencies(RUDRA rudra_theme)
set_source_files_properties("${RUDRA_THEME_DIR}/studio.qss" "${RUDRA_THEME_DIR}/tokens.txt"
                            PROPERTIES GENERATED TRUE)
set_source_files_properties("${RUDRA_THEME_DIR}/studio.qss" PROPERTIES QT_RESOURCE_ALIAS studio.qss)
set_source_files_properties("${RUDRA_THEME_DIR}/tokens.txt" PROPERTIES QT_RESOURCE_ALIAS tokens.txt)
qt_add_resources(RUDRA theme PREFIX /theme FILES "${RUDRA_THEME_DIR}/studio.qss" "${RUDRA_THEME_DIR}/tokens.txt")
file(GLOB rudra_fonts CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/fonts/*.ttf")
qt_add_resources(RUDRA fonts PREFIX / FILES ${rudra_fonts} BASE "${CMAKE_CURRENT_SOURCE_DIR}")
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
