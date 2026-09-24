# app: the Qt application. The shell needs Qt 6.4+ Widgets; the QRhi viewer in
# render/ (Phase 2) needs Qt 6.6+ with Qt Shader Tools (RUDRA_BUILD_RENDER), and
# the release builds use Qt 6.8 LTS (ADR-002). Without it the shell builds
# alone, as the CI Qt shell job checks.
find_package(Qt6 6.4 REQUIRED COMPONENTS Widgets)
qt_standard_project_setup()

# The window and the look as a library: RUDRA and the app's Qt tests link it.
add_library(rudra_app_ui STATIC main_window.cpp main_window.hpp layout.cpp widgets.cpp widgets.hpp scope_widgets.cpp scope_widgets.hpp region_editor.cpp region_editor.hpp model_dialogs.cpp model_dialogs.hpp export_sheet.cpp export_sheet.hpp video_queues.cpp video_queues.hpp queue_window.cpp queue_window.hpp workflow_check.cpp workflow_check.hpp theme.cpp theme.hpp)
target_include_directories(rudra_app_ui PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(rudra_app_ui PUBLIC Qt6::Widgets rudra::rudra_engine rudra::rudra_infer rudra::rudra_video
                      PRIVATE nlohmann_json::nlohmann_json)
if(TARGET rudra_render_gpu)
  target_link_libraries(rudra_app_ui PUBLIC rudra::rudra_render_gpu)
  target_compile_definitions(rudra_app_ui PUBLIC RUDRA_APP_VIEWER)
endif()
if(WIN32)
  target_compile_definitions(rudra_app_ui PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
rudra_warnings(rudra_app_ui)

# The look (Phase 3 step 1): Plex (OFL) and the sheet generated from
# ui/theme.css by the rudra_theme target (cmake/RudraTheme.cmake).
add_dependencies(rudra_app_ui rudra_theme)
set_source_files_properties("${RUDRA_THEME_DIR}/studio.qss" "${RUDRA_THEME_DIR}/tokens.txt"
                            PROPERTIES GENERATED TRUE)
set_source_files_properties("${RUDRA_THEME_DIR}/studio.qss" PROPERTIES QT_RESOURCE_ALIAS studio.qss)
set_source_files_properties("${RUDRA_THEME_DIR}/tokens.txt" PROPERTIES QT_RESOURCE_ALIAS tokens.txt)
qt_add_resources(rudra_app_ui theme PREFIX /theme FILES "${RUDRA_THEME_DIR}/studio.qss" "${RUDRA_THEME_DIR}/tokens.txt")
file(GLOB rudra_fonts CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/fonts/*.ttf")
qt_add_resources(rudra_app_ui fonts PREFIX / FILES ${rudra_fonts} BASE "${CMAKE_CURRENT_SOURCE_DIR}")
# The lockup's mark, the page's own asset.
set_source_files_properties("${PROJECT_SOURCE_DIR}/../ui/assets/rudra-mark-256.png" PROPERTIES QT_RESOURCE_ALIAS rudra-mark.png)
qt_add_resources(rudra_app_ui assets PREFIX /assets FILES "${PROJECT_SOURCE_DIR}/../ui/assets/rudra-mark-256.png")

# The executable and its identity: the icon, the version, and on macOS the
# bundle's Info.plist (HiDPI, the EDR display, the minimum system).
set(_rudra_icon_sources "")
if(APPLE)
  set(_rudra_icon_sources "${CMAKE_CURRENT_SOURCE_DIR}/icons/rudra.icns")
  set_source_files_properties(${_rudra_icon_sources} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
elseif(WIN32)
  configure_file(windows/rudra.rc.in "${CMAKE_CURRENT_BINARY_DIR}/rudra.rc" @ONLY)
  set(_rudra_icon_sources "${CMAKE_CURRENT_BINARY_DIR}/rudra.rc")
endif()
qt_add_executable(RUDRA main.cpp ${_rudra_icon_sources})
target_link_libraries(RUDRA PRIVATE rudra_app_ui)
target_compile_definitions(RUDRA PRIVATE RUDRA_VERSION_LABEL="${RUDRA_VERSION_LABEL}")
rudra_warnings(RUDRA)
set_target_properties(RUDRA PROPERTIES
  WIN32_EXECUTABLE ON
  MACOSX_BUNDLE ON
  MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/macos/Info.plist.in"
  MACOSX_BUNDLE_GUI_IDENTIFIER "com.fxtdstudios.rudra"
  MACOSX_BUNDLE_BUNDLE_NAME "RUDRA"
  MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
  MACOSX_BUNDLE_SHORT_VERSION_STRING "${RUDRA_VERSION_LABEL}"
  MACOSX_BUNDLE_ICON_FILE "rudra.icns"
  MACOSX_BUNDLE_COPYRIGHT "Copyright 2026 FXTD Studios. PolyForm Noncommercial 1.0.0.")

# The window against the Studio page, offscreen (Phase 3): its menubar, keys
# and actions (tests/golden/actions), and the look as it resolves.
if(RUDRA_BUILD_TESTS)
  find_package(Qt6 6.4 REQUIRED COMPONENTS Test)
  add_executable(rudra_app_tests tests/test_app.cpp)
  target_link_libraries(rudra_app_tests PRIVATE rudra_app_ui Qt6::Test nlohmann_json::nlohmann_json GTest::gtest)
  target_compile_definitions(rudra_app_tests PRIVATE RUDRA_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")
  rudra_warnings(rudra_app_tests)
  include(GoogleTest)
  gtest_discover_tests(rudra_app_tests TEST_PREFIX "app." PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
                       DISCOVERY_MODE PRE_TEST)
endif()
