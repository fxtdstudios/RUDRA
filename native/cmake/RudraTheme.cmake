# The Studio's theme (Phase 3 step 1): ${RUDRA_THEME_DIR}/studio.qss and
# tokens.txt, generated from app/theme/pro.css (the Pro boards) at build time by rudra_theme.cmake.
# The app embeds them; test_theme checks them. No Qt and no Python needed.
set(RUDRA_THEME_CSS "${CMAKE_CURRENT_SOURCE_DIR}/app/theme/pro.css")
set(RUDRA_THEME_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/app/theme/studio.qss.in")
set(RUDRA_THEME_DIR "${CMAKE_BINARY_DIR}/theme")
add_custom_command(
  OUTPUT "${RUDRA_THEME_DIR}/studio.qss" "${RUDRA_THEME_DIR}/tokens.txt"
  COMMAND "${CMAKE_COMMAND}" "-DTHEME_CSS=${RUDRA_THEME_CSS}" "-DTEMPLATE=${RUDRA_THEME_TEMPLATE}"
          "-DOUT_DIR=${RUDRA_THEME_DIR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rudra_theme.cmake"
  DEPENDS "${RUDRA_THEME_CSS}" "${RUDRA_THEME_TEMPLATE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rudra_theme.cmake"
  COMMENT "Theme: app/theme/pro.css to studio.qss"
  VERBATIM)
add_custom_target(rudra_theme DEPENDS "${RUDRA_THEME_DIR}/studio.qss" "${RUDRA_THEME_DIR}/tokens.txt")
