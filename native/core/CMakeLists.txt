# core (librudra): types, colour, baseline, tiling, manifests. No Qt, no GPU
# API, no inference runtime: builds and tests headless (principle P6).
rudra_library(rudra_core SOURCES src/baseline.cpp src/tiling.cpp src/model_manifest.cpp
              src/composite.cpp src/gamut.cpp src/master.cpp src/measure.cpp
              src/grade.cpp src/hdr10.cpp src/metadata.cpp src/view.cpp src/scopes.cpp src/viewport.cpp src/guides.cpp
              src/js_format.cpp src/scope_draw.cpp src/readouts.cpp)
target_link_libraries(rudra_core PUBLIC rudra::rudra_platform PRIVATE nlohmann_json::nlohmann_json)
