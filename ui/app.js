/* RUDRA Studio — the whole client.

   The network runs once per frame, on the server, and hands back its raw
   fields. Everything after that — residual strength, recovery mode, preserve,
   region EV, display peak, and the A/B flip — is composed on this machine's
   GPU by ui/compositor.js, which is why the controls move at frame rate
   instead of at one HTTP round trip each.

   Nothing here is decoration. Every menu item runs something, every number is
   measured from the composite actually on screen, and the Region EV panel is
   a real grade that reaches the EXR — rudra/delivery/controls.py applies the
   identical maths when Master writes the file. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var PEAK_NITS = 10000, DIFFUSE_WHITE = 203;
  var SCOPE_LO = 0.05, SCOPE_HI = 4000;
  /* Playback used to be setInterval(160ms) -- a hard 6.25 fps ceiling that
     had nothing to do with the footage, plus a "skip the beat if busy" guard
     that dropped time instead of catching up. And nothing prefetched: each
     frame was fetched only once the playhead landed on it, so every beat paid
     a round trip plus a forward pass. Real time was not reachable by tuning
     that number; the loop had to become a clock with a read-ahead behind it. */
  var CACHE_FRAMES = 32;         // decoded frames held in memory at once
  var PREFETCH_AHEAD = 12;       // frames to keep decoded in front of the head
  var PREFETCH_PARALLEL = 3;     // concurrent read-ahead requests
  var DEFAULT_FPS = 24;

  function defaultRegions() {
    return [
      {label: "highlights", low_nits: 400, high_nits: 2000, ev: 0},
      {label: "speculars", low_nits: 2000, high_nits: 8000, ev: 0},
      {label: "shadows", low_nits: 0.05, high_nits: 12, ev: 0}
    ];
  }

  var state = {
    live: false, busy: false,
    frames: [], index: -1, playing: false, playTimer: null,
    fps: DEFAULT_FPS, playT0: 0, playBase: 0, playRaf: null,
    shownCount: 0, shownT0: 0, measuredFps: 0, dropped: 0,
    mode: "all", strength: 1, peakEv: 0, preserve: true, anchor: true, carryChroma: true,
    show: "model", flipHeld: false,
    view: 0,                 // 0 image, 1 false colour, 2 difference
    probeOn: false, probeLock: null,
    scale: null,             // null = fit to window; otherwise a multiplier
    panX: 0, panY: 0, panning: false,
    // Wipe: null when off, otherwise 0..1 across the plate. The baseline is on
    // the left, the reconstruction on the right.
    wipe: null, wipeDragging: false,
    regions: defaultRegions(), container: "aces",
    railLeft: true, railRight: true, scopesOpen: true, zoom: "fit",
    header: null, metrics: null, scopeData: null, master: null,
    maskPct: {highlight: 0, shadow: 0},
    undo: [], redo: []
  };

  var ctx = null;
  var hiMask = null, shMask = null;   // current frame's masks, full resolution
  var statsTimer = null, statsPending = false;
  /* Dragging the scrub bar walks every frame it passes over. Without this each
     one started its own forward pass on the server's GPU, concurrently and
     uncancelled -- one drag across twenty uncached frames queued twenty. */
  var inflight = null;

  /* Filenames are chosen by whoever made the file, and they land in innerHTML.
     A frame called <img src=x onerror=...> would otherwise run in this page's
     origin, which can drive /api/master. */
  function esc(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function log(line, kind) {
    var box = $("log");
    var row = document.createElement("div");
    if (kind) { row.className = kind; }
    row.textContent = line;
    box.appendChild(row);
    while (box.childNodes.length > 400) { box.removeChild(box.firstChild); }
    box.scrollTop = box.scrollHeight;
  }

  function displayNits() { return DIFFUSE_WHITE * Math.pow(2, state.peakEv); }
  function current() { return state.frames[state.index] || null; }

  /* A 64K lookup beats decoding a million halves by hand. */
  var HALF = (function () {
    var table = new Float32Array(65536);
    for (var h = 0; h < 65536; h++) {
      var s = (h & 0x8000) ? -1 : 1, e = (h & 0x7C00) >> 10, f = h & 0x03FF;
      table[h] = e === 0 ? s * Math.pow(2, -14) * (f / 1024)
               : e === 31 ? (f ? NaN : s * Infinity)
               : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return table;
  }());

  /* ---- formatting ------------------------------------------------------ */
  function ro(k, v, u) {
    return '<div class="ro"><span class="k">' + k + '</span>' +
           '<span class="v">' + v + '</span><span class="u">' + (u || "") + "</span></div>";
  }
  function fmt(n, d) {
    if (n === null || n === undefined || (typeof n === "number" && !isFinite(n))) { return "—"; }
    return Number(n).toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});
  }
  function signed(n, d) {
    if (n === null || n === undefined || !isFinite(n)) { return "—"; }
    return (n >= 0 ? "+" : "−") + fmt(Math.abs(n), d);
  }

  function showMetrics(m) {
    $("measA").innerHTML =
      ro("MaxCLL", fmt(m.maxcll, 0), "nits") +
      ro("MaxFALL", fmt(m.maxfall, 0), "nits") +
      ro("Peak", fmt(m.peak_nits, 1), "nits") +
      ro("P99", fmt(m.p99_nits, 1), "nits") +
      ro("Median", fmt(m.median_nits, 2), "nits");
    $("measB").innerHTML =
      ro("Above diffuse white", fmt(m.above_diffuse_white_pct, 2), "%") +
      ro("Above 1 000 nits", fmt(m.above_1000_nits_pct, 3), "%") +
      ro("Headroom, highlights", signed(m.headroom_highlight_stops, 2), "st") +
      ro("Headroom, shadows", signed(m.headroom_shadow_stops, 2), "st") +
      ro("Departure RMS", fmt(m.departure_rms_stops, 3), "st");
    $("statusMask").textContent =
      "masks " + fmt(m.highlight_mask_pct, 2) + "% highlight / " +
      fmt(m.shadow_mask_pct, 2) + "% shadow";
    var head = state.header || {};
    $("statusTime").textContent =
      (head.source_resolution || "—") + " · net " + fmt(head.elapsed_s, 2) +
      " s · grade " + fmt(m.compose_ms, 1) + " ms";
    $("srcInfo").textContent =
      (head.resolution || "—") + (head.tiled ? " · tiled" : " · one pass");
  }

  /* ---- measurement -----------------------------------------------------
     Mirrors ui/server.py's measure(). MaxCLL and MaxFALL are CTA-861.3 on
     max(R,G,B) and come from an exact GPU reduction, never from the sampled
     readback -- a downsample would miss the single specular pixel MaxCLL is
     entirely about. Everything distributional runs on the capped sample. */
  function computeStats() {
    if (!ctx || !state.header || !hiMask) { return; }
    var t0 = performance.now();
    var s = ctx.sample();
    var w = s.width, h = s.height, n = w * h;
    var luma = new Float32Array(n), baseLuma = new Float32Array(n);
    var aboveDW = 0, above1k = 0, channels = n * 3;
    var CH_BINS = 2048, chHist = new Float32Array(CH_BINS);
    var loLog = Math.log2(SCOPE_LO), hiLog = Math.log2(SCOPE_HI), spanLog = hiLog - loLog;

    for (var i = 0; i < n; i++) {
      var o = i * 4;
      var r = s.model[o] * PEAK_NITS, g = s.model[o + 1] * PEAK_NITS,
          b = s.model[o + 2] * PEAK_NITS;
      luma[i] = Math.max(r, Math.max(g, b));
      baseLuma[i] = Math.max(s.baseline[o], Math.max(s.baseline[o + 1],
                                                     s.baseline[o + 2])) * PEAK_NITS;
      for (var c = 0; c < 3; c++) {
        var v = c === 0 ? r : (c === 1 ? g : b);
        if (v > DIFFUSE_WHITE * (1 + 1e-6)) { aboveDW++; }
        if (v > 1000) { above1k++; }
        var t = (Math.log2(Math.min(Math.max(v, SCOPE_LO), SCOPE_HI)) - loLog) / spanLog;
        chHist[Math.min(CH_BINS - 1, Math.max(0, Math.floor(t * CH_BINS)))]++;
      }
    }

    function pct(p) {
      var want = channels * p / 100, acc = 0;
      for (var k = 0; k < CH_BINS; k++) {
        acc += chHist[k];
        if (acc >= want) { return Math.pow(2, loLog + ((k + 0.5) / CH_BINS) * spanLog); }
      }
      return SCOPE_HI;
    }

    var hiSum = 0, hiBase = 0, hiCount = 0, shSum = 0, shBase = 0, shCount = 0, rms = 0;
    for (var j = 0; j < n; j++) {
      var src = s.index[j];
      if (hiMask[src] > 0.5) { hiSum += luma[j]; hiBase += baseLuma[j]; hiCount++; }
      if (shMask[src] > 0.5) { shSum += luma[j]; shBase += baseLuma[j]; shCount++; }
      var lr = Math.log2((luma[j] + 1e-4) / (baseLuma[j] + 1e-4));
      rms += lr * lr;
    }
    function stopsIn(sum, base, count) {
      if (!count) { return NaN; }
      return Math.log2(Math.max(sum / count, 1e-6) / Math.max(base / count, 1e-6));
    }

    var peak = ctx.peakNits(), basePeak = ctx.basePeakNits();
    var m = {
      maxcll: Math.ceil(peak), maxfall: Math.ceil(ctx.meanNits()),
      peak_nits: peak, baseline_peak_nits: basePeak,
      headroom_stops: Math.log2(Math.max(peak, 1e-6) / Math.max(basePeak, 1e-6)),
      headroom_highlight_stops: stopsIn(hiSum, hiBase, hiCount),
      headroom_shadow_stops: stopsIn(shSum, shBase, shCount),
      departure_rms_stops: Math.sqrt(rms / n),
      p99_nits: pct(99), median_nits: pct(50),
      above_diffuse_white_pct: 100 * aboveDW / channels,
      above_1000_nits_pct: 100 * above1k / channels,
      highlight_mask_pct: state.maskPct.highlight,
      shadow_mask_pct: state.maskPct.shadow
    };
    m.compose_ms = performance.now() - t0;
    state.metrics = m;
    var frame = current();
    if (frame) { frame.peak = peak; frame.aboveDW = m.above_diffuse_white_pct; }
    showMetrics(m);
    state.scopeData = buildScopes(luma, w, h);
    drawScopes(state.scopeData, m);
    drawFrames();
  }

  /* ---- scopes ---------------------------------------------------------- */
  var W = 460, H = 132, HW = 304, HH = 96;
  var COLUMNS = 230, HIST_BINS = 76, COL_BINS = 512;

  function buildScopes(luma, w, h) {
    var loLog10 = Math.log10(SCOPE_LO), span10 = Math.log10(SCOPE_HI / SCOPE_LO);
    var cols = new Float32Array(COLUMNS * COL_BINS);
    var counts = new Float32Array(COLUMNS);
    var hist = new Float32Array(HIST_BINS);
    var loLog2 = Math.log2(SCOPE_LO), span2 = Math.log2(SCOPE_HI) - loLog2;

    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var v = Math.min(Math.max(luma[y * w + x], SCOPE_LO), SCOPE_HI);
        var t = (Math.log10(v) - loLog10) / span10;
        var col = Math.min(COLUMNS - 1, Math.floor(x / w * COLUMNS));
        cols[col * COL_BINS + Math.min(COL_BINS - 1, Math.floor(t * COL_BINS))]++;
        counts[col]++;
        hist[Math.min(HIST_BINS - 1,
                      Math.floor(((Math.log2(v) - loLog2) / span2) * HIST_BINS))]++;
      }
    }

    var out = {lo: [], q1: [], mid: [], q3: [], hi: [], histogram: [],
               floor_nits: SCOPE_LO, ceiling_nits: SCOPE_HI};
    var wanted = [2, 25, 50, 75, 98], keys = ["lo", "q1", "mid", "q3", "hi"];
    for (var c = 0; c < COLUMNS; c++) {
      var total = counts[c] || 1, acc = 0, next = 0, base = c * COL_BINS;
      var got = [0, 0, 0, 0, 0];
      for (var k = 0; k < COL_BINS && next < 5; k++) {
        acc += cols[base + k];
        while (next < 5 && acc >= total * wanted[next] / 100) {
          got[next] = (k + 0.5) / COL_BINS; next++;
        }
      }
      while (next < 5) { got[next] = 1; next++; }
      for (var q = 0; q < 5; q++) { out[keys[q]].push(got[q]); }
    }
    var peak = 1;
    for (var i = 0; i < HIST_BINS; i++) { peak = Math.max(peak, hist[i]); }
    for (var b = 0; b < HIST_BINS; b++) { out.histogram.push(hist[b] / peak); }
    return out;
  }

  function ladder() {
    var lo = SCOPE_LO, span = Math.log10(SCOPE_HI / lo), out = "";
    [[4000, "4000"], [1000, "1000"], [203, "203"], [10, "10"], [1, "1"], [0.05, "0.05"]]
      .forEach(function (t) {
        var y = (H - (Math.log10(t[0] / lo) / span) * H).toFixed(1);
        out += '<line x1="0" y1="' + y + '" x2="' + W + '" y2="' + y +
               '" stroke="#232323" stroke-width="1"/>' +
               '<text x="' + (W + 4) + '" y="' + (Number(y) + 3).toFixed(1) +
               '" font-family="IBM Plex Mono, monospace" font-size="8.5" fill="#5c5c5c">' +
               t[1] + "</text>";
      });
    return out;
  }

  /* Scopes.

     The old pair were three grey percentile strokes and a row of flat grey
     bars. Both drew the data correctly and neither let you READ it: nothing
     marked diffuse white, nothing marked the clip, and highlights and shadows
     were the same colour as everything else -- on a tool whose whole subject is
     what happens at the two ends of the range.

     So: one hue per zone, carried identically across both scopes. Blue below
     -4 stops from diffuse white, neutral through the middle, gold above +2.
     A dashed line on 203 nits in both. And the clip gets its own band, because
     "did this frame clip, and where" is the question the tool exists to
     answer. */
  var Z_SHADOW = "#4d8fd6", Z_MID = "#9fb0c0", Z_HI = "#e8b07a", Z_HI_BRIGHT = "#f8d8b8";

  function nitsToY(n) {
    var span = Math.log10(SCOPE_HI / SCOPE_LO);
    var v = Math.min(Math.max(n, SCOPE_LO), SCOPE_HI);
    return H - (Math.log10(v / SCOPE_LO) / span) * H;
  }

  function drawScopes(s, m) {
    var n = s.mid.length, step = W / n;
    var i, x, out = "";

    /* gridlines first, so the trace sits on top of them */
    var grid = "";
    [[0.05, "0.05"], [1, "1"], [10, "10"], [100, "100"], [1000, "1k"], [4000, "4k"]]
      .forEach(function (t2) {
        var y = nitsToY(t2[0]);
        grid += '<line x1="0" y1="' + y.toFixed(1) + '" x2="' + W + '" y2="' + y.toFixed(1) +
                '" stroke="#1e2c3a" stroke-width="0.6"/>' +
                '<text x="' + (W + 3) + '" y="' + (y + 3).toFixed(1) +
                '" font-family="IBM Plex Mono, monospace" font-size="8" fill="#5b6b7a">' +
                t2[1] + "</text>";
      });
    var dwY = nitsToY(DIFFUSE_WHITE);
    grid += '<line x1="0" y1="' + dwY.toFixed(1) + '" x2="' + W + '" y2="' + dwY.toFixed(1) +
            '" stroke="#6f7f8f" stroke-width="0.8" stroke-dasharray="3 3" opacity="0.8"/>' +
            '<text x="3" y="' + (dwY - 4).toFixed(1) +
            '" font-family="IBM Plex Mono, monospace" font-size="8" fill="#8fa2b4">203 diffuse</text>';

    /* the envelope as filled bands, which reads far better than hairlines */
    var top = [], bot = [], q3 = [], q1 = [], spine = [];
    for (i = 0; i < n; i++) {
      x = (i * step + 0.5);
      top.push(x.toFixed(1) + "," + ((1 - s.hi[i]) * H).toFixed(1));
      bot.push(x.toFixed(1) + "," + ((1 - s.lo[i]) * H).toFixed(1));
      q3.push(x.toFixed(1) + "," + ((1 - s.q3[i]) * H).toFixed(1));
      q1.push(x.toFixed(1) + "," + ((1 - s.q1[i]) * H).toFixed(1));
      spine.push(x.toFixed(1) + "," + ((1 - s.mid[i]) * H).toFixed(1));
    }
    var envelope = '<polygon points="' + top.join(" ") + " " + bot.reverse().join(" ") +
                   '" fill="' + Z_MID + '" opacity="0.13"/>';
    var iqr = '<polygon points="' + q3.join(" ") + " " + q1.reverse().join(" ") +
              '" fill="' + Z_MID + '" opacity="0.30"/>';
    var mid = '<polyline points="' + spine.join(" ") +
              '" fill="none" stroke="' + Z_HI_BRIGHT + '" stroke-width="1.1" opacity="0.92"/>';

    /* how much of the frame is sitting on the ceiling */
    var clipY = nitsToY(SCOPE_HI), clipped = 0;
    for (i = 0; i < n; i++) { if (s.hi[i] >= 0.999) { clipped++; } }
    var clip = "";
    if (clipped) {
      clip = '<rect x="0" y="0" width="' + W + '" height="' + (clipY + 3).toFixed(1) +
             '" fill="' + Z_HI + '" opacity="0.10"/>' +
             '<line x1="0" y1="' + clipY.toFixed(1) + '" x2="' + W + '" y2="' + clipY.toFixed(1) +
             '" stroke="' + Z_HI + '" stroke-width="1"/>' +
             '<text x="3" y="' + (clipY + 9).toFixed(1) +
             '" font-family="IBM Plex Mono, monospace" font-size="8" fill="' + Z_HI +
             '">at ceiling  ' + (clipped * 100 / n).toFixed(1) + "% of columns</text>";
    }

    var cll = "";
    if (m && isFinite(m.maxcll)) {
      var y = nitsToY(m.maxcll);
      cll = '<line x1="0" y1="' + y.toFixed(1) + '" x2="' + W + '" y2="' + y.toFixed(1) +
            '" stroke="' + Z_HI_BRIGHT + '" stroke-width="1" stroke-dasharray="2 3" opacity="0.7"/>' +
            '<text x="' + (W - 3) + '" y="' + (y - 4).toFixed(1) + '" text-anchor="end"' +
            ' font-family="IBM Plex Mono, monospace" font-size="8" fill="' + Z_HI_BRIGHT +
            '">MaxCLL ' + Math.round(m.maxcll) + "</text>";
    }

    /* The top and bottom gridlines sit exactly on y=0 and y=H, so their
       labels fall outside a tight box and get clipped. Pad the viewBox rather
       than inset the plot -- the trace keeps the full height either way. */
    $("wave").setAttribute("viewBox", "0 -9 " + (W + 30) + " " + (H + 18));
    $("wave").innerHTML = grid + envelope + iqr + mid + clip + cll;

    /* ---- histogram ---- */
    var bins = s.histogram.length, bw = HW / bins, bars = "";
    var lo2 = Math.log2(SCOPE_LO), span2 = Math.log2(SCOPE_HI / SCOPE_LO);
    for (var j = 0; j < bins; j++) {
      var v = s.histogram[j];
      if (v <= 0.004) { continue; }
      var stopsFromWhite = (lo2 + (j + 0.5) / bins * span2) - Math.log2(DIFFUSE_WHITE);
      var c = stopsFromWhite < -4 ? Z_SHADOW : (stopsFromWhite > 2 ? Z_HI : Z_MID);
      var h = Math.max(1, v * HH);
      bars += '<rect x="' + (j * bw).toFixed(2) + '" y="' + (HH - h).toFixed(1) +
              '" width="' + (bw * 0.78).toFixed(2) + '" height="' + h.toFixed(1) +
              '" fill="' + c + '" opacity="0.88"/>';
    }
    var marks = "";
    [[0.05, "0.05"], [1, "1"], [10, "10"], [203, "203"], [1000, "1k"], [4000, "4k"]]
      .forEach(function (t2) {
        var mx = (Math.log2(t2[0] / SCOPE_LO) / span2) * HW;
        marks += '<text x="' + mx.toFixed(0) + '" y="110" font-family="IBM Plex Mono, monospace"' +
                 ' font-size="8" fill="#5b6b7a" text-anchor="middle">' + t2[1] + "</text>";
      });
    var dwX = (Math.log2(DIFFUSE_WHITE / SCOPE_LO) / span2) * HW;
    $("hist").innerHTML = bars +
      '<line x1="0" y1="' + HH + '" x2="' + HW + '" y2="' + HH + '" stroke="#1e2c3a"/>' +
      '<line x1="' + dwX.toFixed(1) + '" y1="0" x2="' + dwX.toFixed(1) + '" y2="' + HH +
      '" stroke="#6f7f8f" stroke-dasharray="3 3" opacity="0.8"/>' + marks;
  }

  /* ---- the viewer ------------------------------------------------------ */
  function shown() {
    return (state.show === "baseline") !== state.flipHeld ? "baseline" : "model";
  }

  function graded() {
    return state.regions.some(function (r) { return Math.abs(r.ev) > 1e-9; });
  }

  function present() {
    if (!ctx || !state.header) { return; }
    var wiping = state.wipe !== null;
    ctx.setParams({displayNits: displayNits(), show: shown(),
                   view: state.view,
                   wipe: wiping ? state.wipe : -1});
    ctx.present();
    $("peakBadge").textContent = state.view === 1
      ? "False colour \u00b7 nits"
      : (state.view === 2 ? "Difference \u00b7 |RUDRA \u2212 baseline|"
                          : "Display peak " + Math.round(displayNits()) + " nits");
    var leg = $("fcLegend");
    if (leg) { leg.hidden = state.view !== 1; }
    var isBase = !wiping && shown() === "baseline";
    $("plateLabel").textContent = wiping
      ? ("Baseline \u2502 RUDRA" + (graded() ? " + region EV" : ""))
      : (isBase ? "Inverse-ACES baseline"
                : ("RUDRA reconstruction" + (graded() ? " + region EV" : "")));
    $("plateLabel").classList.toggle("base", isBase);
    var hint = $("plateHint");
    if (hint) {
      hint.textContent = wiping ? "Drag to move the wipe \u00b7 W to exit"
                                : "Hold B, or the image, to flip";
    }
    $("gl").title = wiping ? "Drag to move the wipe" : "Hold to see the baseline";
    [].forEach.call($("viewMode").children, function (b) {
      if (b.id === "wipeBtn") { b.classList.toggle("on", wiping); }
      else { b.classList.toggle("on", !wiping && b.dataset.view === shown()); }
    });
  }

  function recompose() {
    if (!ctx || !state.header) { return; }
    var changed = ctx.setParams({strength: state.strength, mode: state.mode,
                                 preserve: state.preserve, regions: state.regions});
    present();
    if (changed) { scheduleStats(); }
  }

  /* Measurement is the expensive half, so it runs on a trailing edge: the
     picture never waits for it, and the numbers always end up describing the
     composite that is actually on screen. */
  function scheduleStats() {
    statsPending = true;
    if (statsTimer) { return; }
    statsTimer = setTimeout(function () {
      statsTimer = null;
      if (!statsPending) { return; }
      statsPending = false;
      try { computeStats(); } catch (e) { log("measure failed: " + e, "err"); }
      if (statsPending) { scheduleStats(); }
    }, 90);
  }

  /* ---- undo ------------------------------------------------------------ */
  function snapshot() {
    return JSON.stringify({mode: state.mode, strength: state.strength,
                           preserve: state.preserve, regions: state.regions});
  }
  function pushUndo() {
    state.undo.push(snapshot());
    while (state.undo.length > 60) { state.undo.shift(); }
    state.redo.length = 0;
  }
  function restore(json) {
    var s = JSON.parse(json);
    state.mode = s.mode; state.strength = s.strength;
    state.preserve = s.preserve; state.regions = s.regions;
    syncControls();
    recompose();
  }
  function undo() {
    if (!state.undo.length) { log("nothing to undo"); return; }
    state.redo.push(snapshot());
    restore(state.undo.pop());
    log("undo");
  }
  function redo() {
    if (!state.redo.length) { log("nothing to redo"); return; }
    state.undo.push(snapshot());
    restore(state.redo.pop());
    log("redo");
  }

  function syncControls() {
    [].forEach.call($("mode").children, function (b) {
      b.classList.toggle("on", b.dataset.mode === state.mode);
    });
    $("strength").value = String(state.strength);
    $("strengthVal").textContent = state.strength.toFixed(2);
    $("peak").value = String(state.peakEv);
    $("peakVal").textContent = Math.round(displayNits()).toLocaleString("en-US");
    $("preserve").classList.toggle("on", state.preserve);
    $("preserveHint").textContent = state.preserve ? "do-no-harm" : "raw prediction";
    drawRegions();
  }

  /* ---- region EV -------------------------------------------------------
     Drag a value to scrub it, double-click to zero it. These are not a
     preview: Master applies the identical qualifier and gain to the file. */
  function nitsLabel(v) {
    return v >= 1000 ? (v / 1000) + "k" : String(v);
  }

  function drawRegions() {
    $("regions").innerHTML = state.regions.map(function (r, i) {
      var live = Math.abs(r.ev) > 1e-9;
      return '<div class="region' + (i === state.regionSel ? " sel" : "") +
             '" data-i="' + i + '">' +
             '<span class="sw"></span>' +
             '<span class="q">' + nitsLabel(r.low_nits) + " – " + nitsLabel(r.high_nits) +
             ' nits</span>' +
             '<span class="ev' + (live ? " live" : "") + '" data-i="' + i + '">' +
             signed(r.ev, 2) + "</span>" +
             '<span class="u">EV</span></div>';
    }).join("");
    $("regionCount").textContent = String(state.regions.length);
  }

  function bindRegions() {
    var dragging = null, startX = 0, startEv = 0, moved = false;
    $("regions").addEventListener("pointerdown", function (e) {
      var ev = e.target.closest(".ev");
      var row = e.target.closest(".region");
      if (row) { state.regionSel = Number(row.dataset.i); drawRegions(); }
      if (!ev) { return; }
      dragging = Number(ev.dataset.i);
      startX = e.clientX; startEv = state.regions[dragging].ev; moved = false;
      pushUndo();
      try { $("regions").setPointerCapture(e.pointerId); } catch (err) { /* fine */ }
      e.preventDefault();
    });
    $("regions").addEventListener("pointermove", function (e) {
      if (dragging === null) { return; }
      var step = e.shiftKey ? 0.002 : 0.01;
      var value = Math.max(-4, Math.min(4, startEv + (e.clientX - startX) * step));
      if (Math.abs(value - state.regions[dragging].ev) < 1e-6) { return; }
      moved = true;
      state.regions[dragging].ev = Math.round(value * 100) / 100;
      drawRegions();
      recompose();
    });
    window.addEventListener("pointerup", function () {
      if (dragging !== null && !moved) { state.undo.pop(); }
      dragging = null;
    });
    $("regions").addEventListener("dblclick", function (e) {
      var ev = e.target.closest(".ev");
      if (!ev) { return; }
      pushUndo();
      state.regions[Number(ev.dataset.i)].ev = 0;
      drawRegions(); recompose();
    });
  }

  /* ---- frames ----------------------------------------------------------- */
  function drawFrames() {
    var rows = state.frames.map(function (f, i) {
      var size = f.header ? f.header.resolution : "—";
      var peak = f.peak ? Math.round(f.peak).toLocaleString("en-US") : "—";
      var bar = f.aboveDW ? Math.min(1, f.aboveDW / 20) : 0;
      return '<div class="shot' + (i === state.index ? " on" : "") +
             (f.loading ? " loading" : "") + '" data-i="' + i + '">' +
             '<span class="mark"></span><span class="name">' + esc(f.name) + "</span>" +
             '<span class="frames">' + esc(size) + "</span>" +
             '<span class="peak">' + esc(peak) + "</span>" +
             '<span class="bar"><i style="width:' + (bar * 100).toFixed(0) + '%"></i></span></div>';
    }).join("");
    $("shots").innerHTML = rows;
    $("shotCount").textContent = String(state.frames.length);
    $("framesEmpty").hidden = state.frames.length > 0;
    $("tc").textContent = state.frames.length
      ? String(state.index + 1).padStart(3, "0") + " / " +
        String(state.frames.length).padStart(3, "0")
      : "000 / 000";
    var frac = state.frames.length > 1 ? state.index / (state.frames.length - 1) : 0;
    $("scrubHead").style.left = (frac * 100) + "%";
    var single = state.frames.length < 2;
    $("btnPrev").disabled = single;
    $("btnNext").disabled = single;
    $("btnPlay").disabled = single;
  }

  /* Read-ahead. Deliberately separate from fetchFrame(): that one aborts the
     inflight request because the user moved on, which is right for a click and
     fatal for a queue. These run on their own controllers, never abort each
     other, and never touch `inflight` or busy() -- so the interactive path
     behaves exactly as before. */
  var prefetching = 0;

  function prefetchFrom(index) {
    if (!state.live || !ctx) { return; }
    for (var k = 1; k <= PREFETCH_AHEAD && prefetching < PREFETCH_PARALLEL; k++) {
      var f = state.frames[(index + k) % state.frames.length];
      if (!f || f.buf || f.loading) { continue; }
      loadAhead(f);
    }
  }

  function loadAhead(frame) {
    frame.loading = true;
    prefetching++;
    var request = frame.src
      ? {method: "GET", headers: {"X-Rudra-Params": JSON.stringify(params())}}
      : {method: "POST",
         headers: {"Content-Type": "application/octet-stream",
                   "X-Rudra-Params": JSON.stringify(params())},
         body: frame.file};
    fetch(frame.src || "/api/frame", request).then(function (r) {
      var head = r.headers.get("X-Rudra-Frame");
      if (!r.ok || !head) { throw new Error("HTTP " + r.status); }
      return r.arrayBuffer().then(function (buf) {
        frame.header = JSON.parse(head);
        frame.buf = buf;
        if (frame.header.fps) { state.fps = Number(frame.header.fps) || state.fps; }
      });
    }).catch(function () {
      /* A read-ahead miss is not an error the user needs to see. The frame is
         simply not warm, and the clock will show it late or skip it. */
    }).then(function () {
      frame.loading = false;
      prefetching--;
      drawFrames();
      if (state.playing) { prefetchFrom(state.index); }
    });
  }

  function cachedAhead(index) {
    var n = 0;
    for (var k = 1; k <= PREFETCH_AHEAD; k++) {
      var f = state.frames[(index + k) % state.frames.length];
      if (!f || !f.buf) { break; }
      n++;
    }
    return n;
  }

  function evictCache() {
    var loaded = state.frames.filter(function (f) { return f.buf; });
    loaded.sort(function (a, b) { return (a.touched || 0) - (b.touched || 0); });
    while (loaded.length > CACHE_FRAMES) {
      var drop = loaded.shift();
      if (drop !== current()) { drop.buf = null; }
    }
  }

  function adopt(frame) {
    /* Put an already-fetched frame on the GPU. No network, no forward pass. */
    var head = frame.header, off = head.offsets, n = head.width * head.height;
    var fieldsU16 = new Uint16Array(frame.buf, off.fields, n * 4);
    var shadowU16 = new Uint16Array(frame.buf, off.shadow, n);
    var sdr = new Uint8Array(frame.buf, off.sdr, n * 3);

    hiMask = new Float32Array(n);
    shMask = new Float32Array(n);
    var hiCount = 0, shCount = 0;
    for (var i = 0; i < n; i++) {
      hiMask[i] = HALF[fieldsU16[i * 4 + 3]];
      shMask[i] = HALF[shadowU16[i]];
      if (hiMask[i] > 0.5) { hiCount++; }
      if (shMask[i] > 0.5) { shCount++; }
    }
    state.maskPct = {highlight: 100 * hiCount / n, shadow: 100 * shCount / n};
    state.header = head;
    frame.touched = performance.now();

    // The model's own judgement about this frame's shadows, computed server-side
    // from the whole frame. Missing on a checkpoint without the gate, and 1.0 is
    // the ungated behaviour, so an old checkpoint composes exactly as before.
    ctx.setParams({shadowWeight: head.shadow_weight === undefined
                                 ? 1.0 : Number(head.shadow_weight)});
    ctx.setFrame({width: head.width, height: head.height,
                  sdr: sdr, fields: fieldsU16, shadow: shadowU16,
                  log_scale: head.log_scale, max_hdr: head.max_hdr,
                  corpus_ev: head.corpus_ev});
    ctx.setParams({strength: state.strength, mode: state.mode,
                   preserve: state.preserve, regions: state.regions});
    $("empty").hidden = true;
    $("plate").hidden = false;
    present();
    computeStats();
    evictCache();
  }

  function params(extra) {
    var p = {
      strength: state.strength,
      display_nits: displayNits(),
      recovery_mode: state.mode,
      preserve_outside: state.preserve,
      regions: state.regions,
      region_softness_stops: 1.0,
      tile_size: 0, tile_overlap: 64, max_side: 1600
    };
    if (extra) { Object.keys(extra).forEach(function (k) { p[k] = extra[k]; }); }
    return p;
  }

  function busy(on, label) {
    state.busy = on;
    var ready = !on && state.frames.length > 0 && state.live;
    $("btnMaster").disabled = !ready;
    $("btnReprocess").disabled = !ready;
    $("btnMaster").textContent = label || "Master EXR";
  }

  function select(index, force) {
    if (!state.frames.length) { return; }
    index = Math.max(0, Math.min(state.frames.length - 1, index));
    state.index = index;
    var frame = state.frames[index];
    drawFrames();
    evictCache();
    if (frame.buf && !force) { adopt(frame); return; }
    fetchFrame(frame);
  }

  function fetchFrame(frame) {
    if (!state.live) { log("no model loaded", "err"); return; }
    if (!ctx) { log("no WebGL2 with float render targets in this browser", "err"); return; }
    if (frame.loading) { return; }
    if (inflight) { inflight.controller.abort(); inflight.frame.loading = false; }
    var controller = new AbortController();
    inflight = {controller: controller, frame: frame};
    frame.loading = true;
    drawFrames();
    busy(true);
    var started = performance.now();
    log("forward pass " + frame.name);
    /* A dropped file travels as bytes; a frame of an opened shot does not.
       The server already has that footage on disk, so it is fetched by index
       and comes back in exactly the same format -- everything downstream of
       here, adopt() included, cannot tell the two apart. */
    var request = frame.src
      ? {method: "GET", signal: controller.signal,
         headers: {"X-Rudra-Params": JSON.stringify(params())}}
      : {method: "POST", signal: controller.signal,
         headers: {"Content-Type": "application/octet-stream",
                   "X-Rudra-Params": JSON.stringify(params())},
         body: frame.file};
    fetch(frame.src || "/api/frame", request).then(function (r) {
      var head = r.headers.get("X-Rudra-Frame");
      if (!r.ok || !head) {
        return r.json().then(function (d) { throw new Error(d.error || ("HTTP " + r.status)); });
      }
      return r.arrayBuffer().then(function (buf) {
        return {header: JSON.parse(head), buf: buf};
      });
    }).then(function (d) {
      if (inflight && inflight.controller === controller) { inflight = null; }
      frame.loading = false;
      frame.header = d.header;
      frame.buf = d.buf;
      busy(false);
      if (state.frames[state.index] !== frame) { drawFrames(); return; }
      adopt(frame);
      if (d.header.fps) { state.fps = Number(d.header.fps) || state.fps; }
      prefetchFrom(state.index);
      log("  " + d.header.resolution + "  net " + d.header.elapsed_s + " s  transfer " +
          (d.buf.byteLength / 1048576).toFixed(1) + " MB  total " +
          ((performance.now() - started) / 1000).toFixed(2) + " s");
    }).catch(function (e) {
      if (inflight && inflight.controller === controller) { inflight = null; }
      frame.loading = false;
      // A superseded frame is not a failure: the user simply moved on.
      if (e.name === "AbortError") { drawFrames(); return; }
      busy(false); drawFrames();
      log("frame failed: " + e.message, "err");
    });
  }

  function addFiles(files) {
    var start = state.frames.length;
    [].forEach.call(files, function (f) {
      state.frames.push({file: f, name: f.name, header: null, buf: null,
                         loading: false, peak: null, aboveDW: null});
    });
    log("added " + files.length + " frame" + (files.length === 1 ? "" : "s"));
    drawFrames();
    select(state.frames.length === files.length ? 0 : start);
  }

  /* ---- open a shot by path ---------------------------------------------
     The rail could already PLAY a sequence -- scrubber, transport, wipe, all
     of it worked on a list. What it could not do was acquire one: every frame
     had to be dragged on, and a dropped .mov did nothing because the page
     expects images. The server sits on the same machine as the footage, so
     the page sends a path and the server reads it where it is. A 1.4 GB
     ProRes never crosses the socket. */
  function seqNote(text, isError) {
    var node = $("seqNote");
    if (!node) { return; }
    node.textContent = text || "";
    node.classList.toggle("err", !!isError);
    node.hidden = !text;
  }

  /* A dropped frame carries its own bytes; a frame of an opened shot is a
     reference into footage the server already has. Anything that POSTs the
     picture -- Master EXR above all -- has to name it instead. */
  function seqRef() {
    var f = current();
    if (!f || !f.src) { return {}; }
    var query = f.src.split("?")[1] || "";
    var out = {};
    query.split("&").forEach(function (pair) {
      var kv = pair.split("=");
      out[decodeURIComponent(kv[0])] = decodeURIComponent(kv[1] || "");
    });
    return {seq_job: out.job, seq_index: Number(out.i)};
  }

  function openSequence(path) {
    if (!path || !path.trim()) { seqNote("Type a folder or a video file first.", true); return; }
    seqNote("opening " + path + " ...");
    fetch("/api/sequence/open", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path: path})
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok || !d.ok) { throw new Error(d.error || ("HTTP " + r.status)); }
        return d;
      });
    }).then(function (d) {
      closeAll();
      /* Stubs, not frames. Nothing is decoded until the transport asks for
         it, so a 900-frame plate opens as fast as a 3-frame one. */
      for (var i = 0; i < d.count; i++) {
        state.frames.push({
          src: "/api/sequence/frame?job=" + encodeURIComponent(d.job) + "&i=" + i,
          /* The server's own name for the frame: the file name for a folder,
             shot_000123 for a video. It is what the rail shows and what a
             master is named after, so a made-up "folder 12" would put twelve
             different frames in one EXR file name. */
          name: (d.names && d.names[i]) || (d.name + " " + (i + 1)),
          file: null, header: null, buf: null,
          loading: false, peak: null, aboveDW: null
        });
      }
      seqNote(d.kind + " · " + d.count + " frame" + (d.count === 1 ? "" : "s")
              + (d.fps ? " · " + Number(d.fps).toFixed(2) + " fps" : ""));
      log("opened " + d.kind + " " + d.name + " -- " + d.count + " frames");
      drawFrames();
      select(0);
    }).catch(function (e) {
      seqNote(e.message, true);
      log("open shot failed: " + e.message, "err");
    });
  }

  function closeAll() {
    stopPlay();
    if (inflight) { inflight.controller.abort(); inflight = null; }
    state.frames = []; state.index = -1; state.header = null;
    hiMask = shMask = null;
    $("plate").hidden = true; $("empty").hidden = false;
    $("measA").innerHTML = ""; $("measB").innerHTML = "";
    $("wave").innerHTML = ""; $("hist").innerHTML = "";
    $("statusMask").textContent = "—"; $("statusTime").textContent = "—";
    $("srcInfo").textContent = "—";
    busy(false); drawFrames();
    log("closed all frames");
  }

  /* ---- transport -------------------------------------------------------- */
  function step(delta) {
    if (state.frames.length < 2) { return; }
    select((state.index + delta + state.frames.length) % state.frames.length);
  }
  function stopPlay() {
    if (state.playTimer) { clearInterval(state.playTimer); state.playTimer = null; }
    if (state.playRaf) { cancelAnimationFrame(state.playRaf); state.playRaf = null; }
    state.playing = false;
    $("btnPlay").classList.remove("on");
  }

  /* The clock. Which frame is due is a function of elapsed wall-clock time and
     the footage's own frame rate -- never of how long the last one took. If a
     frame is not warm when it comes due it is SKIPPED, not waited for: that is
     the difference between playing at real time and playing in slow motion.
     The count of skips is reported rather than hidden, because a shot that
     drops half its frames is telling you the read-ahead cannot keep up. */
  function playTick() {
    if (!state.playing) { return; }
    var n = state.frames.length;
    var elapsed = (performance.now() - state.playT0) / 1000;
    var want = (state.playBase + Math.floor(elapsed * state.fps)) % n;

    if (want !== state.index) {
      var f = state.frames[want];
      if (f && f.buf) {
        state.index = want;
        drawFrames();
        evictCache();
        adopt(f);
        state.shownCount++;
      } else {
        state.dropped++;
      }
      prefetchFrom(want);

      var dt = (performance.now() - state.shownT0) / 1000;
      if (dt >= 0.5) {
        state.measuredFps = state.shownCount / dt;
        state.shownCount = 0; state.shownT0 = performance.now();
        var el = $("srcInfo");
        if (el) {
          el.textContent = (state.header && state.header.resolution ? state.header.resolution + "  ·  " : "")
            + state.measuredFps.toFixed(1) + " / " + state.fps.toFixed(0) + " fps"
            + (state.dropped ? "  ·  " + state.dropped + " dropped" : "")
            + "  ·  " + cachedAhead(state.index) + " ahead";
        }
      }
    }
    state.playRaf = requestAnimationFrame(playTick);
  }

  function togglePlay() {
    if (state.frames.length < 2) { return; }
    if (state.playing) {
      stopPlay();
      log("stop  ·  " + state.measuredFps.toFixed(1) + " fps measured"
          + (state.dropped ? ", " + state.dropped + " frames dropped" : ""));
      return;
    }
    if (state.header && state.header.fps) { state.fps = Number(state.header.fps) || state.fps; }
    state.playing = true;
    state.playT0 = performance.now();
    state.playBase = state.index < 0 ? 0 : state.index;
    state.shownCount = 0; state.shownT0 = performance.now();
    state.dropped = 0;
    $("btnPlay").classList.add("on");
    log("play " + state.frames.length + " frames at " + state.fps.toFixed(0) + " fps");
    prefetchFrom(state.index);
    state.playRaf = requestAnimationFrame(playTick);
  }

  /* ---- delivery --------------------------------------------------------- */
  function deliveryRecord() {
    var head = state.header || {};
    return {
      checkpoint: head.checkpoint || null,
      step: head.step === undefined ? null : head.step,
      frame: current() ? current().name : null,
      resolution: head.resolution || null,
      container: state.container === "aces" ? "ACES 2065-1 (AP0)" : "linear Rec.2020",
      transfer: "linear",
      diffuse_white_nits: DIFFUSE_WHITE,
      recovery_mode: state.mode,
      residual_strength: state.strength,
      preserve_outside: state.preserve,
      region_ev: graded() ? state.regions : null,
      measurements: state.metrics || null
    };
  }

  function copy(label, value) {
    var text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { log("copied " + label + " to the clipboard"); },
        function () { sheet(label, "<pre>" + esc(text) + "</pre>"); });
    } else {
      sheet(label, "<pre>" + esc(text) + "</pre>");
    }
  }

  function master() {
    if (!state.live || !current() || state.busy) { return; }
    busy(true, "Mastering…");
    log("master " + current().name + " → " +
        (state.container === "aces" ? "ACES 2065-1" : "linear Rec.2020") +
        " EXR, full resolution" + (graded() ? ", with region EV" : ""));
    fetch("/api/master", {
      method: "POST",
      headers: {"Content-Type": "application/octet-stream",
                "X-Rudra-Params": JSON.stringify(params(Object.assign(
                    {name: current().name,
                     container: state.container,
                     master_max_side: 4096,
                     anchor: state.anchor,
                     carry_chroma: state.carryChroma}, seqRef())))},
      body: current().file || null
    }).then(function (r) { return r.json(); }).then(function (d) {
      busy(false);
      if (!d.ok) { log("master failed: " + (d.error || "unknown"), "err"); return; }
      state.master = d;
      log("  " + d.file + "  " + (d.bytes / 1048576).toFixed(2) + " MB  " +
          d.resolution + "  " + d.container + (d.tiled ? "  tiled" : "  one pass") +
          "  " + d.elapsed_s + " s");
      log("  MaxCLL " + d.maxcll + "  MaxFALL " + d.maxfall + " nits  sidecar " + d.sidecar);
      var a = document.createElement("a");
      a.href = "/api/master/download?f=" + encodeURIComponent(d.file);
      a.download = d.file;
      document.body.appendChild(a); a.click(); a.remove();
    }).catch(function (e) { busy(false); log("master failed: " + e, "err"); });
  }

  /* ---- overlay sheet ---------------------------------------------------- */
  function sheet(title, html) {
    $("overlaySheet").innerHTML = "<h3>" + esc(title) + "</h3>" + html +
      '<div class="close">Esc, or click anywhere, to close</div>';
    $("overlay").hidden = false;
  }
  function closeSheet() { $("overlay").hidden = true; }

  var SHORTCUTS = [
    ["B (hold)", "Flip to the inverse-ACES baseline"],
    ["W", "Wipe: baseline left, RUDRA right. Drag the image to move it."],
    ["\u2190 \u2192", "Nudge the wipe (Shift for fine)"],
    ["O", "Open frames"],
    ["M", "Master EXR"],
    [", / .", "Previous / next frame"],
    ["Home / End", "First / last frame"],
    ["Space", "Play / pause"],
    ["[ / ]", "Weaker / stronger residual"],
    ["1 2 3 4", "Recovery: all, highlights, shadows, off"],
    ["P", "Preserve outside masks"],
    ["Z / Y", "Undo / redo"],
    ["?", "This list"]
  ];

  /* ---- viewer geometry ---------------------------------------------------
     Zoom and pan are a CSS transform on the plate, not a resize of the canvas:
     the canvas stays at frame resolution, so zooming never re-rasterises and
     never costs a recomposite. It also keeps every client-to-image mapping
     honest for free -- getBoundingClientRect() already reports the transformed
     box, so the wipe and the probe need no scale maths of their own. */
  function fitScale() {
    var v = $("viewer"), f = ctx && ctx.size();
    if (!f || !v.clientWidth) { return 1; }
    var pad = 28;
    return Math.min((v.clientWidth - pad) / f.width, (v.clientHeight - pad) / f.height);
  }

  function applyViewport() {
    var plate = $("plate");
    if (!plate) { return; }
    if (state.scale === null) {
      state.panX = state.panY = 0;
      plate.style.transform = "";
      plate.style.maxWidth = "100%";
      plate.style.maxHeight = "100%";
    } else {
      plate.style.maxWidth = "none";
      plate.style.maxHeight = "none";
      plate.style.transform = "translate(" + state.panX.toFixed(1) + "px," +
                             state.panY.toFixed(1) + "px) scale(" + state.scale + ")";
    }
    var el = $("zoomVal");
    if (el) {
      var s = state.scale === null ? fitScale() : state.scale;
      el.textContent = Math.round(s * 100) + "%";
    }
    [].forEach.call($("zoomSeg").children, function (b) {
      if (!b.dataset.zoom) { return; }
      b.classList.toggle("on", b.dataset.zoom === "fit" ? state.scale === null
                                                        : state.scale === 1);
    });
  }

  function zoomAbout(clientX, clientY, factor) {
    var plate = $("plate");
    if (!plate || plate.hidden) { return; }
    var from = state.scale === null ? fitScale() : state.scale;
    var to = Math.max(0.05, Math.min(32, from * factor));
    var r = plate.getBoundingClientRect();
    // keep the point under the cursor fixed
    var cx = clientX - (r.left + r.width / 2);
    var cy = clientY - (r.top + r.height / 2);
    var k = to / from;
    state.panX = (state.panX - cx) * k + cx;
    state.panY = (state.panY - cy) * k + cy;
    state.scale = to;
    applyViewport();
  }

  /* ---- probe -------------------------------------------------------------
     The frame-wide numbers answer "what is in this shot". Nothing answered
     "what is THIS pixel", which is the question asked at a specular -- and it
     is the claim itself, measured one pixel at a time: what the baseline had
     there, what the network put there, and whether the SDR was clipped at that
     point at all. A lift where the SDR never clipped is invention, not
     reconstruction, and this is the only view that tells them apart. */
  function probeAt(clientX, clientY) {
    if (!ctx || !state.header || !$("plate") || $("plate").hidden) { return null; }
    var cv = $("gl"), r = cv.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) { return null; }
    var f = ctx.size();
    var x = Math.floor((clientX - r.left) / r.width * f.width);
    var y = Math.floor((clientY - r.top) / r.height * f.height);
    if (x < 0 || y < 0 || x >= f.width || y >= f.height) { return null; }
    var s = ctx.probe(x, y);
    if (!s) { return null; }
    var i = y * f.width + x;
    s.hiMask = hiMask ? hiMask[i] : null;
    s.shMask = shMask ? shMask[i] : null;
    var frame = current();
    if (frame && frame.buf && frame.header && frame.header.offsets) {
      var sdr = new Uint8Array(frame.buf, frame.header.offsets.sdr, f.width * f.height * 3);
      s.sdr = [sdr[i * 3], sdr[i * 3 + 1], sdr[i * 3 + 2]];
    }
    return s;
  }

  function stops(nits) {
    return Math.log2(Math.max(nits, 1e-6) / DIFFUSE_WHITE);
  }

  function showProbe(s, clientX, clientY) {
    var box = $("probeBox");
    if (!box) { return; }
    if (!s) { box.hidden = true; return; }
    var clipped = s.sdr && Math.max(s.sdr[0], s.sdr[1], s.sdr[2]) >= 254;
    var d = stops(s.model.nits) - stops(s.baseline.nits);
    function row(k, v, cls) {
      return '<div class="pr"><span class="k">' + k + '</span><span class="v' +
             (cls ? " " + cls : "") + '">' + v + "</span></div>";
    }
    box.innerHTML =
      row("x,y", s.x + ", " + s.y) +
      row("baseline", Math.round(s.baseline.nits).toLocaleString("en-US") + " nits  " +
          (stops(s.baseline.nits) >= 0 ? "+" : "") + stops(s.baseline.nits).toFixed(2) + " st") +
      row("RUDRA", Math.round(s.model.nits).toLocaleString("en-US") + " nits  " +
          (stops(s.model.nits) >= 0 ? "+" : "") + stops(s.model.nits).toFixed(2) + " st", "hi") +
      row("delta", (d >= 0 ? "+" : "") + d.toFixed(2) + " stops", Math.abs(d) > 0.01 ? "hi" : "") +
      (s.sdr ? row("SDR", s.sdr.join(",") + (clipped ? "  clipped" : ""),
                   clipped ? "bad" : "") : "") +
      row("mask", "hi " + (s.hiMask === null ? "-" : s.hiMask.toFixed(2)) +
                  "  sh " + (s.shMask === null ? "-" : s.shMask.toFixed(2)));
    box.hidden = false;
    var vr = $("viewer").getBoundingClientRect();
    var bw = 196, bh = box.offsetHeight || 118;
    var lx = clientX - vr.left + 16, ly = clientY - vr.top + 16;
    if (lx + bw > vr.width) { lx = clientX - vr.left - bw - 16; }
    if (ly + bh > vr.height) { ly = clientY - vr.top - bh - 16; }
    box.style.left = Math.max(4, lx) + "px";
    box.style.top = Math.max(4, ly) + "px";
  }

  /* ---- window ----------------------------------------------------------- */
  function applyWindow() {
    $("railLeft").classList.toggle("hidden", !state.railLeft);
    $("railRight").classList.toggle("hidden", !state.railRight);
    $("scopes").classList.toggle("hidden", !state.scopesOpen);
    applyViewport();
  }

  /* ---- menus ------------------------------------------------------------ */
  function closeMenus() {
    [].forEach.call(document.querySelectorAll(".menu.open"), function (m) {
      m.classList.remove("open");
    });
  }

  function checkState(key) {
    if (key.indexOf("mode:") === 0) { return state.mode === key.slice(5); }
    if (key.indexOf("container:") === 0) { return state.container === key.slice(10); }
    if (key.indexOf("zoom:") === 0) { return state.zoom === key.slice(5); }
    return !!state[key];
  }

  function refreshMenu(menu) {
    [].forEach.call(menu.querySelectorAll("button[data-check]"), function (b) {
      b.classList.toggle("checked", checkState(b.dataset.check));
    });
    var many = state.frames.length > 1, any = state.frames.length > 0;
    var flags = {
      "close": any, "master": any && state.live && !state.busy,
      "undo": state.undo.length > 0, "redo": state.redo.length > 0,
      "first": many, "prev": many, "next": many, "last": many, "play": many,
      "copy-metrics": !!state.metrics, "copy-scopes": !!state.scopeData,
      "copy-delivery": any, "remeasure": any
    };
    [].forEach.call(menu.querySelectorAll("button[data-act]"), function (b) {
      var f = flags[b.dataset.act];
      b.disabled = f === undefined ? false : !f;
    });
  }

  var ACTIONS = {
    "open": function () { $("file").click(); },
    "close": closeAll,
    "master": master,
    "undo": undo,
    "redo": redo,
    "reset-recon": function () {
      pushUndo();
      state.mode = "all"; state.strength = 1; state.preserve = true;
      syncControls(); recompose(); log("reconstruction reset");
    },
    "reset-regions": function () {
      pushUndo(); state.regions = defaultRegions();
      drawRegions(); recompose(); log("region EV reset");
    },
    "first": function () { select(0); },
    "prev": function () { step(-1); },
    "next": function () { step(1); },
    "last": function () { select(state.frames.length - 1); },
    "play": togglePlay,
    "mode-all": function () { setMode("all"); },
    "mode-highlights": function () { setMode("highlights"); },
    "mode-shadows": function () { setMode("shadows"); },
    "mode-off": function () { setMode("off"); },
    "preserve": function () { $("preserve").click(); },
    "strength-down": function () { nudgeStrength(-0.1); },
    "strength-up": function () { nudgeStrength(0.1); },
    "copy-metrics": function () { copy("measurements", state.metrics); },
    "copy-scopes": function () { copy("scope data", state.scopeData); },
    "copy-delivery": function () { copy("delivery metadata", deliveryRecord()); },
    "remeasure": function () { computeStats(); log("re-measured"); },
    "container-aces": function () { setContainer("aces"); },
    "container-linear": function () { setContainer("linear"); },
    "rail-left": function () { state.railLeft = !state.railLeft; applyWindow(); },
    "rail-right": function () { state.railRight = !state.railRight; applyWindow(); },
    "scopes": function () { state.scopesOpen = !state.scopesOpen; applyWindow(); },
    "zoom-fit": function () { state.scale = null; state.zoom = "fit"; applyViewport(); },
    "zoom-actual": function () {
      state.scale = 1; state.zoom = "actual"; state.panX = state.panY = 0; applyViewport();
    },
    "shortcuts": function () {
      sheet("Keyboard", SHORTCUTS.map(function (s) {
        return '<div class="row"><span class="k">' + s[0] + "</span><span>" + s[1] + "</span></div>";
      }).join(""));
    },
    "about": function () {
      var head = state.header || {};
      sheet("RUDRA Studio", [
        ["Checkpoint", head.checkpoint || "—"],
        ["Step", head.step === undefined || head.step === null ? "—" : head.step],
        ["Device", $("device").textContent],
        ["Composite", "GPU, WebGL2 — ui/compositor.js"],
        ["Storage", "log2_extended, 0.005–1 000 000 nits"],
        ["Master", "ACES 2065-1 (AP0), ST 2065-4 chromaticities"],
        ["Built by", "FXTD Studios / Radiance Research"]
      ].map(function (r) {
        return '<div class="row"><span class="k">' + esc(r[0]) + "</span><span>" +
               esc(r[1]) + "</span></div>";
      }).join(""));
    }
  };

  function setContainer(kind) {
    state.container = kind;
    $("containerField").textContent = kind === "aces"
      ? "OpenEXR — ACES 2065-1" : "OpenEXR — linear Rec.2020";
    $("primariesField").textContent = kind === "aces"
      ? "AP0 (ST 2065-4)" : "Rec.2020";
    log("container: " + $("containerField").textContent);
  }

  function nudgeStrength(delta) {
    pushUndo();
    state.strength = Math.max(0, Math.min(2, Math.round((state.strength + delta) * 100) / 100));
    syncControls(); recompose();
  }

  function setMode(mode) {
    if (state.mode === mode) { return; }
    pushUndo(); state.mode = mode; syncControls(); recompose();
  }

  /* ---- wiring ----------------------------------------------------------- */
  function bind() {
    /* menus */
    [].forEach.call(document.querySelectorAll(".menu"), function (menu) {
      menu.querySelector(".mtitle").addEventListener("click", function (e) {
        e.stopPropagation();
        var open = menu.classList.contains("open");
        closeMenus();
        if (!open) { refreshMenu(menu); menu.classList.add("open"); }
      });
      menu.addEventListener("pointerenter", function () {
        if (document.querySelector(".menu.open") && !menu.classList.contains("open")) {
          closeMenus(); refreshMenu(menu); menu.classList.add("open");
        }
      });
    });
    $("menubar").addEventListener("click", function (e) {
      var b = e.target.closest("button[data-act]");
      if (!b || b.disabled) { return; }
      closeMenus();
      var fn = ACTIONS[b.dataset.act];
      if (fn) { fn(); } else { log("no action for " + b.dataset.act, "err"); }
    });
    window.addEventListener("click", closeMenus);

    /* files */
    $("file").addEventListener("change", function (e) {
      if (e.target.files.length) { addFiles(e.target.files); }
      e.target.value = "";
    });
    ["dragover", "drop"].forEach(function (t) {
      window.addEventListener(t, function (e) { e.preventDefault(); });
    });
    window.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files.length) { addFiles(e.dataTransfer.files); }
    });
    $("viewer").addEventListener("dblclick", function () { $("file").click(); });

    /* frames rail */
    $("shots").addEventListener("click", function (e) {
      var row = e.target.closest(".shot");
      if (row) { stopPlay(); select(Number(row.dataset.i)); }
    });

    /* transport */
    $("seqOpen").addEventListener("click", function () {
      openSequence($("seqPath").value);
    });
    $("seqPath").addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); openSequence($("seqPath").value); }
      // The window listens for single-key shortcuts; a path is full of them.
      e.stopPropagation();
    });

    $("btnPrev").addEventListener("click", function () { stopPlay(); step(-1); });
    $("btnNext").addEventListener("click", function () { stopPlay(); step(1); });
    $("btnPlay").addEventListener("click", togglePlay);
    (function () {
      var scrubbing = false;
      function seek(clientX) {
        if (state.frames.length < 2) { return; }
        var r = $("scrub").getBoundingClientRect();
        var frac = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
        select(Math.round(frac * (state.frames.length - 1)));
      }
      $("scrub").addEventListener("pointerdown", function (e) {
        scrubbing = true; stopPlay();
        $("scrub").setPointerCapture(e.pointerId); seek(e.clientX);
      });
      $("scrub").addEventListener("pointermove", function (e) {
        if (scrubbing) { seek(e.clientX); }
      });
      window.addEventListener("pointerup", function () { scrubbing = false; });
    }());

    /* reconstruction */
    $("mode").addEventListener("click", function (e) {
      var b = e.target.closest("button[data-mode]");
      if (b) { setMode(b.dataset.mode); }
    });
    $("viewMode").addEventListener("click", function (e) {
      var b = e.target.closest("button[data-view]");
      if (b) { state.show = b.dataset.view; state.wipe = null; present(); }
    });
    function wipeFromEvent(e) {
      var r = $("gl").getBoundingClientRect();
      if (r.width <= 0) { return state.wipe; }
      return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    }
    ACTIONS.wipe = function () {
      state.wipe = state.wipe === null ? 0.5 : null;
      state.flipHeld = false;
      present();
    };

    $("plate").addEventListener("pointerdown", function (e) {
      if (e.button !== 0) { return; }
      // While the wipe is up, dragging moves the seam. Hold-to-flip would fight
      // it for the same gesture, so only one of the two is ever live.
      if (state.wipe !== null) {
        state.wipeDragging = true;
        if ($("plate").setPointerCapture) { $("plate").setPointerCapture(e.pointerId); }
        state.wipe = wipeFromEvent(e);
        present();
        return;
      }
      state.flipHeld = true; present();
    });
    $("plate").addEventListener("pointermove", function (e) {
      if (state.wipeDragging) { state.wipe = wipeFromEvent(e); present(); }
    });
    window.addEventListener("pointerup", function () {
      if (state.wipeDragging) { state.wipeDragging = false; }
      if (state.flipHeld) { state.flipHeld = false; present(); }
    });
    $("wipeBtn").addEventListener("click", function () { ACTIONS.wipe(); });

    /* view layers */
    $("viewLayer").addEventListener("click", function (e) {
      var b = e.target.closest("button[data-layer]");
      if (!b) { return; }
      state.view = parseInt(b.dataset.layer, 10) || 0;
      [].forEach.call($("viewLayer").children, function (c) {
        c.classList.toggle("on", c === b);
      });
      present();
    });

    /* probe */
    $("probeBtn").addEventListener("click", function () {
      state.probeOn = !state.probeOn;
      this.classList.toggle("on", state.probeOn);
      $("viewer").classList.toggle("probing", state.probeOn);
      if (!state.probeOn) { showProbe(null); }
    });
    $("plate").addEventListener("pointermove", function (e) {
      if (!state.probeOn || state.panning || state.wipeDragging) { return; }
      showProbe(probeAt(e.clientX, e.clientY), e.clientX, e.clientY);
    });
    $("plate").addEventListener("pointerleave", function () {
      if (state.probeOn) { showProbe(null); }
    });

    /* zoom and pan. Scroll zooms about the cursor, middle-drag pans -- the
       Nuke and Resolve convention, and it leaves left-drag alone so the wipe
       seam and hold-to-flip keep the gesture they already had. */
    $("viewer").addEventListener("wheel", function (e) {
      if (!ctx || !state.header) { return; }
      e.preventDefault();
      zoomAbout(e.clientX, e.clientY, e.deltaY < 0 ? 1.12 : 1 / 1.12);
    }, {passive: false});

    $("viewer").addEventListener("pointerdown", function (e) {
      if (e.button !== 1) { return; }        // middle only
      e.preventDefault();
      if (state.scale === null) { state.scale = fitScale(); }
      state.panning = {x: e.clientX, y: e.clientY, px: state.panX, py: state.panY};
      $("viewer").classList.add("panning");
      if ($("viewer").setPointerCapture) { $("viewer").setPointerCapture(e.pointerId); }
    });
    $("viewer").addEventListener("pointermove", function (e) {
      if (!state.panning) { return; }
      state.panX = state.panning.px + (e.clientX - state.panning.x);
      state.panY = state.panning.py + (e.clientY - state.panning.y);
      applyViewport();
    });
    window.addEventListener("pointerup", function () {
      if (state.panning) { state.panning = false; $("viewer").classList.remove("panning"); }
    });
    $("viewer").addEventListener("dblclick", function () {
      state.scale = null; applyViewport();
    });
    window.addEventListener("resize", function () {
      if (state.scale === null) { applyViewport(); }
    });
    $("strength").addEventListener("pointerdown", pushUndo);
    $("strength").addEventListener("input", function () {
      state.strength = parseFloat(this.value);
      $("strengthVal").textContent = state.strength.toFixed(2);
      recompose();
    });
    $("peak").addEventListener("input", function () {
      state.peakEv = parseFloat(this.value);
      $("peakVal").textContent = Math.round(displayNits()).toLocaleString("en-US");
      present();
    });
    /* Delivery-time only. The viewer composes on the GPU from raw fields and
       does not apply the anchor yet, so the master can differ from what is on
       screen by the source's own exposure -- the hint says so rather than
       letting someone discover it in Resolve. */
    $("anchor").addEventListener("click", function () {
      state.anchor = !state.anchor;
      this.classList.toggle("on", state.anchor);
      $("anchorHint").textContent = state.anchor ? "conform" : "raw ITM level";
      log("master will " + (state.anchor ? "anchor to the source exposure"
                                         : "keep the inverse tone map's own level"));
    });

    $("carryChroma").addEventListener("click", function () {
      state.carryChroma = !state.carryChroma;
      this.classList.toggle("on", state.carryChroma);
      $("chromaHint").textContent = state.carryChroma ? "below the clip" : "per-channel";
      log("master will " + (state.carryChroma
          ? "take hue from the source below the clip"
          : "keep the per-channel expansion's own hue"));
    });

    $("preserve").addEventListener("click", function () {
      pushUndo();
      state.preserve = !state.preserve;
      this.classList.toggle("on", state.preserve);
      $("preserveHint").textContent = state.preserve ? "do-no-harm" : "raw prediction";
      recompose();
    });
    $("btnReprocess").addEventListener("click", function () {
      if (current()) { select(state.index, true); }
    });
    $("btnMaster").addEventListener("click", master);
    bindRegions();

    /* overlay */
    $("overlay").addEventListener("click", closeSheet);

    /* keyboard */
    window.addEventListener("keydown", function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) { return; }
      var k = e.key;
      if (k === "b" || k === "B") {
        if (!e.repeat) { state.flipHeld = true; present(); }
        return;
      }
      if (k === "w" || k === "W") { ACTIONS.wipe(); return; }
      if (state.wipe !== null && (k === "ArrowLeft" || k === "ArrowRight")) {
        e.preventDefault();
        var step = e.shiftKey ? 0.01 : 0.05;
        state.wipe = Math.max(0, Math.min(1,
          state.wipe + (k === "ArrowRight" ? step : -step)));
        present();
        return;
      }
      if (k === "Escape") {
        if (state.wipe !== null) { state.wipe = null; present(); return; }
        closeSheet(); closeMenus(); return;
      }
      var map = {
        "o": "open", "O": "open", "m": "master", "M": "master",
        "z": "undo", "Z": "undo", "y": "redo", "Y": "redo",
        ",": "prev", ".": "next", "Home": "first", "End": "last",
        "[": "strength-down", "]": "strength-up", "?": "shortcuts",
        "p": "preserve", "P": "preserve"
      };
      if (k === " ") { e.preventDefault(); togglePlay(); return; }
      if (k === "1" || k === "2" || k === "3" || k === "4") {
        setMode(["all", "highlights", "shadows", "off"][Number(k) - 1]);
        return;
      }
      var act = map[k];
      if (act && ACTIONS[act]) { e.preventDefault(); ACTIONS[act](); }
    });
    window.addEventListener("keyup", function (e) {
      if ((e.key === "b" || e.key === "B") && state.flipHeld) {
        state.flipHeld = false; present();
      }
    });
  }

  function auditMenus() {
    var orphans = [];
    [].forEach.call(document.querySelectorAll(".menu .drop button[data-act]"), function (b) {
      if (!ACTIONS[b.dataset.act] && orphans.indexOf(b.dataset.act) < 0) {
        orphans.push(b.dataset.act);
      }
    });
    if (orphans.length) {
      log("menu items with no action: " + orphans.join(", "), "err");
    }
    return orphans;
  }

  /* The README screenshot is taken by a headless browser, which cannot drop a
     file on the window. ?demo=1 loads the bundled frame instead, and
     ?frame=<url> loads any other, so ui/capture_shot.py captures the app doing
     its job rather than an empty "Drop an SDR frame" panel -- which is exactly
     what it would have captured between the UI rebuild and 28 Aug 2026. */
  var DEMO_FRAME = "assets/cinematic_hdr_sunset.png";

  function autoload() {
    var query = new URLSearchParams(location.search);
    var url = query.get("frame") || (query.get("demo") ? DEMO_FRAME : null);
    if (!url) { return; }
    log("autoloading " + url);
    fetch(url, {cache: "no-store"}).then(function (r) {
      if (!r.ok) { throw new Error("HTTP " + r.status); }
      return r.blob();
    }).then(function (blob) {
      var name = url.split("/").pop().split("?")[0] || "frame.png";
      addFiles([new File([blob], name, {type: blob.type || "image/png"})]);
    }).catch(function (e) {
      log("autoload failed: " + e.message, "err");
    });
  }

  function boot() {
    ctx = window.RudraGL ? window.RudraGL.create($("gl")) : null;
    bind();
    auditMenus();
    syncControls();
    applyWindow();
    drawFrames();
    setContainer("aces");
    if (!ctx) {
      log("this browser has no WebGL2 with float render targets — the viewer " +
          "needs both", "err");
    }
    fetch("/api/model", {cache: "no-store"}).then(function (r) { return r.json(); })
      .then(function (info) {
        state.live = !!info.loaded;
        $("ckpt").textContent = info.name || info.reason || "no model";
        $("device").textContent = info.gpu || info.device || "—";
        $("lamp").classList.toggle("off", !info.loaded);
        if (info.loaded) {
          log("model " + info.name + (info.step ? "  step " + info.step : ""));
          log("device " + (info.gpu || info.device));
          log("drop frames — the network runs once each, then the grade is local");
          autoload();
        } else {
          log("no model loaded: " + (info.reason || "unknown"), "err");
        }
        busy(false);
      }).catch(function (e) { log("cannot reach the server: " + e, "err"); });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
}());
