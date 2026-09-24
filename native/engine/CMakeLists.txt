# engine: the frame path (FrameEngine: the InferActor, generations, the frame
# cache), and later the session and the other actors (Phase 3).
rudra_library(rudra_engine SOURCES src/frame_engine.cpp src/actions.cpp src/session.cpp src/measure.cpp src/master_job.cpp src/model_catalog.cpp)
target_link_libraries(rudra_engine PUBLIC rudra::rudra_core rudra::rudra_infer rudra::rudra_media rudra::rudra_render
                      rudra::rudra_deliver rudra::rudra_video PRIVATE nlohmann_json::nlohmann_json)
