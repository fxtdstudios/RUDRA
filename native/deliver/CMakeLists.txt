# deliver: presets, ffmpeg pipe, QC, queue (Phase 4).
rudra_library(rudra_deliver INTERFACE_ONLY)
target_link_libraries(rudra_deliver INTERFACE rudra::rudra_core)
