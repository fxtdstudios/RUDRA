# rudra-hdr-probe: Gate B. Needs Qt 6.6+ for the public <rhi/qrhi.h> and Qt
# Shader Tools for qsb (ADR-002, ADR-003).
find_package(Qt6 6.6 REQUIRED COMPONENTS Gui ShaderTools)
qt_add_executable(rudra-hdr-probe main.cpp hdr_probe_window.cpp hdr_probe_window.hpp)
qt_add_shaders(rudra-hdr-probe "rudra_probe_shaders"
  PREFIX "/shaders"
  FILES probe.vert probe.frag)
target_link_libraries(rudra-hdr-probe PRIVATE Qt6::Gui Qt6::GuiPrivate rudra::rudra_render)
set_target_properties(rudra-hdr-probe PROPERTIES AUTOMOC OFF)
rudra_warnings(rudra-hdr-probe)
