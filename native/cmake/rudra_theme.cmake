# The Studio's theme as Qt style sheets (Phase 3 step 1), run as a script:
#   cmake -DTHEME_CSS=ui/theme.css -DTEMPLATE=app/theme/studio.qss.in -DOUT_DIR=... -P rudra_theme.cmake
#
# ui/theme.css is the one source of the look. Its :root custom properties are
# the tokens; the template refers to them as var(--name), the way the page's
# own CSS does, and to the first family of --ui and --mono as
# var(--ui-family) and var(--mono-family) (a QSS font-family is one family).
# Writes studio.qss and tokens.txt ("name value" per line, for the palette
# the app builds at run time). An unknown token or a var() left over is an
# error, so a renamed token breaks the build, not the look.

cmake_minimum_required(VERSION 3.24)

foreach(v THEME_CSS TEMPLATE OUT_DIR)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "rudra_theme.cmake: ${v} is not set")
  endif()
endforeach()

file(READ "${THEME_CSS}" css)
# A Windows checkout may have CRLF; no token value may carry a CR.
string(REPLACE "\r" "" css "${css}")

# Comments out first: they name colours the theme rejected (the old
# blue-black), and one of them could otherwise read as a token.
set(clean "")
while(TRUE)
  string(FIND "${css}" "/*" a)
  if(a EQUAL -1)
    string(APPEND clean "${css}")
    break()
  endif()
  string(SUBSTRING "${css}" 0 ${a} head)
  string(APPEND clean "${head}")
  string(SUBSTRING "${css}" ${a} -1 css)
  string(FIND "${css}" "*/" b)
  if(b EQUAL -1)
    message(FATAL_ERROR "${THEME_CSS}: unterminated comment")
  endif()
  math(EXPR b "${b} + 2")
  string(SUBSTRING "${css}" ${b} -1 css)
endwhile()

# CMake lists are ';'-separated: turn the declarations into lines first.
string(REPLACE ";" "\n" clean "${clean}")
string(FIND "${clean}" ":root{" r)
if(r EQUAL -1)
  message(FATAL_ERROR "${THEME_CSS}: no :root{ block")
endif()
string(SUBSTRING "${clean}" ${r} -1 root)
string(FIND "${root}" "}" e)
string(SUBSTRING "${root}" 0 ${e} root)

string(REGEX MATCHALL "--[a-z0-9-]+:[^\n]+" decls "${root}")
set(names "")
foreach(d IN LISTS decls)
  string(REGEX MATCH "^--([a-z0-9-]+):(.*)$" _ "${d}")
  set(n "${CMAKE_MATCH_1}")
  string(STRIP "${CMAKE_MATCH_2}" val)
  set(tok_${n} "${val}")
  list(APPEND names "${n}")
endforeach()
foreach(req bg panel ink accent ui mono)
  if(NOT DEFINED tok_${req})
    message(FATAL_ERROR "${THEME_CSS}: token --${req} is missing")
  endif()
endforeach()

# The first family of each font stack, unquoted.
foreach(f ui mono)
  string(REGEX MATCH "^\"?([^\",]+)" _ "${tok_${f}}")
  set(tok_${f}-family "${CMAKE_MATCH_1}")
  list(APPEND names "${f}-family")
endforeach()

file(READ "${TEMPLATE}" qss)
string(REPLACE "\r" "" qss "${qss}")
set(tokens_txt "")
foreach(n IN LISTS names)
  string(REPLACE "var(--${n})" "${tok_${n}}" qss "${qss}")
  string(APPEND tokens_txt "${n} ${tok_${n}}\n")
endforeach()
string(FIND "${qss}" "var(" left)
if(NOT left EQUAL -1)
  string(SUBSTRING "${qss}" ${left} 40 what)
  message(FATAL_ERROR "${TEMPLATE}: unknown token at \"${what}\"")
endif()

set(banner "/* Generated from ui/theme.css and native/app/theme/studio.qss.in. Do not edit. */\n")
file(MAKE_DIRECTORY "${OUT_DIR}")
# Written only when changed, so an unrelated build does not re-embed it.
function(write_if_changed path body)
  set(old "")
  if(EXISTS "${path}")
    file(READ "${path}" old)
  endif()
  if(NOT old STREQUAL body)
    # LF on every OS (file(CONFIGURE) with NEWLINE_STYLE, not file(WRITE)).
    set(RUDRA_THEME_BODY "${body}")
    file(CONFIGURE OUTPUT "${path}" CONTENT "@RUDRA_THEME_BODY@" @ONLY NEWLINE_STYLE UNIX)
  endif()
endfunction()
write_if_changed("${OUT_DIR}/studio.qss" "${banner}${qss}")
write_if_changed("${OUT_DIR}/tokens.txt" "${tokens_txt}")
