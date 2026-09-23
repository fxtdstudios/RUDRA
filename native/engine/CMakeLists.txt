# engine: session, scheduler, caches, actors (Phase 3).
rudra_library(rudra_engine INTERFACE_ONLY)
target_link_libraries(rudra_engine INTERFACE rudra::rudra_core rudra::rudra_infer rudra::rudra_media rudra::rudra_render rudra::rudra_deliver)
