#!/usr/bin/env bash
# Runs one CI step and, when it fails, publishes the last lines of its output
# as an error annotation. Job logs need a GitHub login to read; annotations
# are public through the API (check-runs/<job>/annotations), so a failure can
# be read and fixed without one.
#
#   tools/ci_run.sh "Build" cmake --build build --parallel
#   tools/ci_run.sh "Goldens" bash -c 'python tools/emit_golden.py && git diff --exit-code ...'
set -uo pipefail
title=$1
shift
log=$(mktemp)
"$@" 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
if [ "$code" -ne 0 ]; then
  # Workflow commands take one line: %, CR and LF are escaped.
  body=$(tail -n "${CI_TAIL_LINES:-60}" "$log" | sed -e 's/%/%25/g' -e 's/\r/%0D/g' | awk 'BEGIN { ORS = "%0A" } { print }')
  echo "::error title=${title} failed (exit ${code})::${body}"
  # GoogleTest: one annotation per failing test (up to 8), its own output from
  # [ RUN ] to [ FAILED ], since the tail above only shows the summary.
  awk -v max="${CI_FAILED_BLOCKS:-8}" '
    /^\[ RUN      \]/ { n = 0; name = $4 }
    { buf[n++ % 40] = $0 }
    /^\[  FAILED  \] .* \([0-9]+ ms\)$/ && shown < max {
      shown++; body = ""; start = (n > 40) ? n - 40 : 0
      for (i = start; i < n; i++) { l = buf[i % 40]; gsub(/%/, "%25", l); gsub(/\r/, "", l); body = body l "%0A" }
      printf "::error title=%s: %s::%s\n", title, name, body
    }' title="$title" "$log"
fi
rm -f "$log"
exit "$code"
