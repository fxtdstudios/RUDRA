# One warning policy for every first-party target.
function(rudra_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
    # std::getenv is the portable call; MSVC's C4996 wants _dupenv_s.
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion
                                             -Wno-sign-conversion)
  endif()
endfunction()

# A first-party library: include/ is its public interface, src/ its private body.
function(rudra_library target)
  set(options INTERFACE_ONLY)
  cmake_parse_arguments(ARG "${options}" "" "SOURCES" ${ARGN})
  if(ARG_INTERFACE_ONLY)
    add_library(${target} INTERFACE)
    target_include_directories(${target} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(${target} INTERFACE cxx_std_20)
  else()
    add_library(${target} STATIC ${ARG_SOURCES})
    target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(${target} PUBLIC cxx_std_20)
    rudra_warnings(${target})
  endif()
  add_library(rudra::${target} ALIAS ${target})
endfunction()
