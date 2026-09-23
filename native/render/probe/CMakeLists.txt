# Gate B's rudra-hdr-probe and day 8's rudra-gpu-parity. Qt 6.6+ for the public
# <rhi/qrhi.h>, Qt Shader Tools for qsb (ADR-002, ADR-003).
qt_add_executable(rudra-hdr-probe main.cpp hdr_probe_window.cpp hdr_probe_window.hpp)
qt_add_shaders(rudra-hdr-probe "rudra_probe_shaders"
  PREFIX "/shaders"
  FILES probe.vert probe.frag)
target_link_libraries(rudra-hdr-probe PRIVATE Qt6::Gui Qt6::GuiPrivate rudra::rudra_render)
set_target_properties(rudra-hdr-probe PROPERTIES AUTOMOC OFF)
if(WIN32)
  target_link_libraries(rudra-hdr-probe PRIVATE dxgi)   # the Windows HDR state, read independently of Qt
endif()
rudra_warnings(rudra-hdr-probe)

qt_add_executable(rudra-gpu-parity gpu_parity.cpp)
target_link_libraries(rudra-gpu-parity PRIVATE Qt6::Gui rudra::rudra_render_gpu rudra::rudra_core
                      nlohmann_json::nlohmann_json)
target_compile_definitions(rudra-gpu-parity PRIVATE
  RUDRA_COMPOSITE_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden/composite")
set_target_properties(rudra-gpu-parity PROPERTIES AUTOMOC OFF)
rudra_warnings(rudra-gpu-parity)
