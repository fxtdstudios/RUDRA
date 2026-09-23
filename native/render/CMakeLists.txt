# render: the viewer's GPU side. rudra_render is the Qt-free interface every
# layer above sees; rudra_render_gpu is its QRhi implementation (Qt 6.6+ with
# Qt Shader Tools), built with RUDRA_BUILD_RENDER.
rudra_library(rudra_render INTERFACE_ONLY)
target_link_libraries(rudra_render INTERFACE rudra::rudra_core)

if(RUDRA_BUILD_RENDER)
  find_package(Qt6 6.6 REQUIRED COMPONENTS Gui ShaderTools)
  rudra_library(rudra_render_gpu SOURCES src/gpu_composite.cpp src/viewer_window.cpp)
  target_link_libraries(rudra_render_gpu PUBLIC rudra::rudra_core PRIVATE Qt6::Gui Qt6::GuiPrivate)
  set_target_properties(rudra_render_gpu PROPERTIES AUTOMOC OFF)
  if(WIN32)
    target_compile_definitions(rudra_render_gpu PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
  qt_add_shaders(rudra_render_gpu "rudra_composite_shaders"
    PREFIX "/rudra/shaders"
    BASE shaders
    GLSL "300es,330"   # texelFetch and float render targets: no GLSL ES 1.0
    FILES shaders/fullscreen.vert shaders/composite.frag shaders/display.frag shaders/reduce.frag
          shaders/blit.vert shaders/blit.frag)
endif()

if(RUDRA_BUILD_HDR_PROBE)
  add_subdirectory(probe)
endif()
