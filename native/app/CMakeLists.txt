# app: the Qt application. The shell needs Qt 6.4+ Widgets; the QRhi viewer in
# render/ (Phase 2) needs Qt 6.6+ for the public <rhi/qrhi.h>, and the release
# builds use Qt 6.8 LTS (ADR-002).
find_package(Qt6 6.4 REQUIRED COMPONENTS Widgets)
qt_standard_project_setup()
qt_add_executable(RUDRA main.cpp)
target_link_libraries(RUDRA PRIVATE Qt6::Widgets rudra::rudra_engine rudra::rudra_infer)
rudra_warnings(RUDRA)
set_target_properties(RUDRA PROPERTIES WIN32_EXECUTABLE ON MACOSX_BUNDLE ON)
