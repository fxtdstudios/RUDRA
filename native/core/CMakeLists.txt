# core (librudra): types, colour, baseline, tiling, manifests. No Qt, no GPU
# API, no inference runtime: builds and tests headless (principle P6).
rudra_library(rudra_core SOURCES src/baseline.cpp src/tiling.cpp src/model_manifest.cpp)
target_link_libraries(rudra_core PUBLIC rudra::rudra_platform PRIVATE nlohmann_json::nlohmann_json)
