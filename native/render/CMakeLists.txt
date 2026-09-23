# render: QRhi viewer (Phase 2); interface only until then.
rudra_library(rudra_render INTERFACE_ONLY)
target_link_libraries(rudra_render INTERFACE rudra::rudra_core)

if(RUDRA_BUILD_HDR_PROBE)
  add_subdirectory(probe)
endif()
