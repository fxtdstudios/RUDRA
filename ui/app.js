/* RUDRA Studio — the whole client.

   The network runs once per frame, on the server, and hands back its raw
   fields. Everything after that — residual strength, recovery mode, preserve,
   display peak, and the A/B flip — is composed on this machine's GPU by
   ui/compositor.js, which is why the controls move at frame rate instead of at
   one HTTP round trip each.

   Nothing here simulates. Every number on screen is measured from the
   composite the viewer is actually showing. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var state = {
    live: false, file: null, busy: false,
    mode: "all", strength: 1, peakEv: 0, preserve: true,
    show: "model", flipHeld: false,
    header: null, master: null, metrics: null,
    maskPct: {highlight: 0, shadow: 0}
  };

  var ctx = null;              // the GPU compositor
  var fields = null;           // Float32Array, highlight mask at full res
  var shadowField = null;      // Float32Array, shadow mask at full res
  var statsTimer = null, statsPending = false;

  var SHOTS = [
    ["carousel_fireworks", "4181", "4000", 1.00],
    ["smith_hammering", "702", "4000", 1.00],
    ["fireplace", "1392", "4000", 1.00],
    ["beerfest_lightshow", "2757", "4000", 1.00],
    ["showgirl_01", "1164", "4000", 1.00],
    ["bistro", "1455", "4000", 1.00],
    ["poker_travelling", "2922", "4000", 1.00],
    ["fishing_longshot", "1251", "4000", 1.00],
    ["Chimera_DCI4k_2398p", "3126", "10000", 1.00],
    ["Bar-Scene_PQ-1K", "6201", "991", 0.25]
  ];

  var PEAK_NITS = 10000, DIFFUSE_WHITE = 203;
  var SCOPE_LO = 0.05, SCOPE_HI = 4000;

  function log(line, kind) {
    var box = $("log");
    var row = document.createElement("div");
    if (kind) { row.className = kind; }
    row.textContent = line;
    box.appendChild(row);
    box.scrollTop = box.scrollHeight;
  }

  function displayNits() { return DIFFUSE_WHITE * Math.pow(2, state.peakEv); }

  /* ---- half float ------------------------------------------------------ */
  /* A 64K lookup beats decoding 1.4 million values by hand every frame. */
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

  /* ---- readouts -------------------------------------------------------- */
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
      head.source_resolution + " · net " + fmt(head.elapsed_s, 2) + " s · " +
      "grade " + fmt(m.compose_ms, 1) + " ms";
    $("srcInfo").textContent = head.resolution + (head.tiled ? " · tiled" : " · one pass");
  }

  /* ---- measurement ------------------------------------------------------
     Mirrors ui/server.py's measure(): MaxCLL and MaxFALL are CTA-861.3 on
     max(R,G,B), and they come from an exact GPU reduction, not from the
     sampled readback -- a downsample would miss the single specular pixel
     MaxCLL is entirely about. Everything distributional is measured on the
     sample, which is capped at 768 on the long side. */
  function computeStats() {
    if (!ctx || !state.header) { return; }
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
      var ch = [r, g, b];
      for (var c = 0; c < 3; c++) {
        var v = ch[c];
        if (v > DIFFUSE_WHITE * (1 + 1e-6)) { aboveDW++; }
        if (v > 1000) { above1k++; }
        var t = (Math.log2(Math.min(Math.max(v, SCOPE_LO), SCOPE_HI)) - loLog) / spanLog;
        chHist[Math.min(CH_BINS - 1, Math.max(0, Math.floor(t * CH_BINS)))]++;
      }
    }

    function pct(p) {                       // percentile from the channel histogram
      var want = channels * p / 100, acc = 0;
      for (var k = 0; k < CH_BINS; k++) {
        acc += chHist[k];
        if (acc >= want) {
          return Math.pow(2, loLog + ((k + 0.5) / CH_BINS) * spanLog);
        }
      }
      return SCOPE_HI;
    }

    /* Where the model was ALLOWED to act. The residual is gated to the masks,
       so this is the honest "is it doing anything". */
    var hiSum = 0, hiBase = 0, hiCount = 0, shSum = 0, shBase = 0, shCount = 0;
    var rms = 0;
    for (var j = 0; j < n; j++) {
      var src = s.index[j];
      if (fields[src * 4 + 3] > 0.5) { hiSum += luma[j]; hiBase += baseLuma[j]; hiCount++; }
      if (shadowField[src] > 0.5) { shSum += luma[j]; shBase += baseLuma[j]; shCount++; }
      var lr = Math.log2((luma[j] + 1e-4) / (baseLuma[j] + 1e-4));
      rms += lr * lr;
    }
    function stopsIn(sum, base, count) {
      if (!count) { return NaN; }
      return Math.log2(Math.max(sum / count, 1e-6) / Math.max(base / count, 1e-6));
    }

    var peak = ctx.peakNits(), basePeak = ctx.basePeakNits();
    var m = {
      maxcll: Math.ceil(peak),
      maxfall: Math.ceil(ctx.meanNits()),
      peak_nits: peak,
      baseline_peak_nits: basePeak,
      headroom_stops: Math.log2(Math.max(peak, 1e-6) / Math.max(basePeak, 1e-6)),
      headroom_highlight_stops: stopsIn(hiSum, hiBase, hiCount),
      headroom_shadow_stops: stopsIn(shSum, shBase, shCount),
      departure_rms_stops: Math.sqrt(rms / n),
      p99_nits: pct(99),
      median_nits: pct(50),
      above_diffuse_white_pct: 100 * aboveDW / channels,
      above_1000_nits_pct: 100 * above1k / channels,
      highlight_mask_pct: state.maskPct.highlight,
      shadow_mask_pct: state.maskPct.shadow
    };
    m.compose_ms = performance.now() - t0;
    state.metrics = m;
    showMetrics(m);
    drawScopes(buildScopes(luma, w, h), m);
  }

  /* ---- scopes ---------------------------------------------------------- */
  var W = 460, H = 132, HW = 304, HH = 96;
  var COLUMNS = 230, HIST_BINS = 76, COL_BINS = 512;

  /* Per-column percentiles from a log histogram: O(pixels), not O(n log n),
     which is what lets the waveform keep up with a slider drag. */
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
        var t2 = (Math.log2(v) - loLog2) / span2;
        hist[Math.min(HIST_BINS - 1, Math.floor(t2 * HIST_BINS))]++;
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
          got[next] = (k + 0.5) / COL_BINS;
          next++;
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
    var lo = SCOPE_LO, hi = SCOPE_HI, span = Math.log10(hi / lo), out = "";
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

  function drawScopes(s, m) {
    var n = s.mid.length, step = W / n, outer = "", inner = "", spine = [];
    for (var i = 0; i < n; i++) {
      var x = (i * step + 0.5).toFixed(1);
      outer += "M" + x + " " + ((1 - s.lo[i]) * H).toFixed(1) + "V" + ((1 - s.hi[i]) * H).toFixed(1);
      inner += "M" + x + " " + ((1 - s.q1[i]) * H).toFixed(1) + "V" + ((1 - s.q3[i]) * H).toFixed(1);
      spine.push(x + "," + ((1 - s.mid[i]) * H).toFixed(1));
    }
    var cll = "";
    if (m && isFinite(m.maxcll)) {
      var span = Math.log10(SCOPE_HI / SCOPE_LO);
      var y = H - (Math.log10(Math.min(Math.max(m.maxcll, SCOPE_LO), SCOPE_HI) / SCOPE_LO) / span) * H;
      cll = '<line x1="0" y1="' + y.toFixed(1) + '" x2="' + W + '" y2="' + y.toFixed(1) +
            '" stroke="#cfcfcf" stroke-width="1" stroke-dasharray="2 3" opacity="0.55"/>' +
            '<text x="4" y="' + (y - 4).toFixed(1) +
            '" font-family="IBM Plex Mono, monospace" font-size="8.5" fill="#a8a8a8">MaxCLL ' +
            Math.round(m.maxcll) + "</text>";
    }
    $("wave").innerHTML = ladder() +
      '<path d="' + outer + '" stroke="#8f8f8f" stroke-width="1.6" opacity="0.30"/>' +
      '<path d="' + inner + '" stroke="#d8d8d8" stroke-width="1.6" opacity="0.62"/>' +
      '<polyline points="' + spine.join(" ") +
      '" fill="none" stroke="#ffffff" stroke-width="0.9" opacity="0.5"/>' + cll;

    var bins = s.histogram.length, bw = HW / bins, bars = "";
    for (var j = 0; j < bins; j++) {
      var v = s.histogram[j];
      if (v > 0.004) {
        bars += '<rect x="' + (j * bw).toFixed(2) + '" y="' + (HH - v * HH).toFixed(1) +
                '" width="' + (bw * 0.72).toFixed(2) + '" height="' + (v * HH).toFixed(1) + '"/>';
      }
    }
    var marks = "";
    [[0.05, "0.05"], [1, "1"], [10, "10"], [203, "203"], [1000, "1k"], [4000, "4k"]]
      .forEach(function (t) {
        var x = (Math.log2(t[0] / SCOPE_LO) / Math.log2(SCOPE_HI / SCOPE_LO)) * HW;
        marks += '<text x="' + x.toFixed(0) + '" y="110" font-family="IBM Plex Mono, monospace"' +
                 ' font-size="8.5" fill="#5c5c5c" text-anchor="middle">' + t[1] + "</text>";
      });
    var dw = (Math.log2(DIFFUSE_WHITE / SCOPE_LO) / Math.log2(SCOPE_HI / SCOPE_LO)) * HW;
    $("hist").innerHTML = '<g fill="#b4b4b4" opacity="0.78">' + bars + "</g>" +
      '<line x1="0" y1="' + HH + '" x2="' + HW + '" y2="' + HH + '" stroke="#2c2c2c"/>' +
      '<line x1="' + dw.toFixed(1) + '" y1="0" x2="' + dw.toFixed(1) + '" y2="' + HH +
      '" stroke="#4a4a4a" stroke-dasharray="2 3"/>' + marks;
  }

  /* ---- the viewer ------------------------------------------------------ */
  function shown() {
    return (state.show === "baseline") !== state.flipHeld ? "baseline" : "model";
  }

  function present() {
    if (!ctx || !state.header) { return; }
    ctx.setParams({displayNits: displayNits(), show: shown()});
    ctx.present();
    $("peakBadge").textContent = "Display peak " + Math.round(displayNits()) + " nits";
    var isBase = shown() === "baseline";
    $("plateLabel").textContent = isBase ? "INVERSE-ACES BASELINE" : "RUDRA RECONSTRUCTION";
    $("plateLabel").classList.toggle("base", isBase);
    [].forEach.call($("viewMode").children, function (b) {
      b.classList.toggle("on", b.dataset.view === shown());
    });
  }

  function recompose() {
    if (!ctx || !state.header) { return; }
    var changed = ctx.setParams({strength: state.strength, mode: state.mode,
                                 preserve: state.preserve});
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
      try { computeStats(); }
      catch (e) { log("measure failed: " + e, "err"); }
      if (statsPending) { scheduleStats(); }
    }, 90);
  }

  /* ---- requests -------------------------------------------------------- */
  function params(extra) {
    var p = {
      strength: state.strength,
      display_nits: displayNits(),
      recovery_mode: state.mode,
      preserve_outside: state.preserve,
      tile_size: 0, tile_overlap: 64, max_side: 1600
    };
    if (extra) { Object.keys(extra).forEach(function (k) { p[k] = extra[k]; }); }
    return p;
  }

  function busy(on, label) {
    state.busy = on;
    $("btnMaster").disabled = on || !state.file || !state.live;
    $("btnReprocess").disabled = on || !state.file || !state.live;
    $("btnMaster").textContent = label || "Master EXR";
  }

  function loadFrame() {
    if (!state.live || !state.file || state.busy) { return; }
    if (!ctx) { log("no WebGL2 with float targets in this browser", "err"); return; }
    busy(true);
    log("forward pass " + state.file.name);
    var started = performance.now();
    fetch("/api/frame", {
      method: "POST",
      headers: {"Content-Type": "application/octet-stream",
                "X-Rudra-Params": JSON.stringify(params())},
      body: state.file
    }).then(function (r) {
      var head = r.headers.get("X-Rudra-Frame");
      if (!r.ok || !head) {
        return r.json().then(function (d) {
          throw new Error(d.error || ("HTTP " + r.status));
        });
      }
      return r.arrayBuffer().then(function (buf) {
        return {header: JSON.parse(head), buf: buf};
      });
    }).then(function (d) {
      busy(false);
      var head = d.header, off = head.offsets, n = head.width * head.height;
      state.header = head;

      var fieldsU16 = new Uint16Array(d.buf, off.fields, n * 4);
      var shadowU16 = new Uint16Array(d.buf, off.shadow, n);
      var sdr = new Uint8Array(d.buf, off.sdr, n * 3);

      /* Decoded once, kept: the masks drive the headroom numbers on every
         later measurement, and re-decoding 1.4 M halves per slider move
         would be the new bottleneck. */
      fields = new Float32Array(n * 4);
      for (var i = 0; i < n * 4; i++) { fields[i] = HALF[fieldsU16[i]]; }
      shadowField = new Float32Array(n);
      var hiCount = 0, shCount = 0;
      for (var j = 0; j < n; j++) {
        shadowField[j] = HALF[shadowU16[j]];
        if (fields[j * 4 + 3] > 0.5) { hiCount++; }
        if (shadowField[j] > 0.5) { shCount++; }
      }
      state.maskPct = {highlight: 100 * hiCount / n, shadow: 100 * shCount / n};

      ctx.setFrame({width: head.width, height: head.height,
                    sdr: sdr, fields: fieldsU16, shadow: shadowU16});
      ctx.setParams({strength: state.strength, mode: state.mode,
                     preserve: state.preserve});
      $("empty").hidden = true;
      $("plate").hidden = false;
      present();
      computeStats();
      log("  " + head.resolution + "  net " + head.elapsed_s + " s  transfer " +
          ((d.buf.byteLength / 1048576).toFixed(1)) + " MB  total " +
          ((performance.now() - started) / 1000).toFixed(2) + " s");
      log("  controls are local from here — no round trip per change");
    }).catch(function (e) {
      busy(false);
      log("frame failed: " + e.message, "err");
    });
  }

  function master() {
    if (!state.live || !state.file || state.busy) { return; }
    busy(true, "Mastering…");
    log("master " + state.file.name + " → ACES 2065-1 EXR, full resolution");
    fetch("/api/master", {
      method: "POST",
      headers: {"Content-Type": "application/octet-stream",
                "X-Rudra-Params": JSON.stringify(params({name: state.file.name,
                                                         container: "aces",
                                                         tile_size: 512,
                                                         master_max_side: 4096}))},
      body: state.file
    }).then(function (r) { return r.json(); }).then(function (d) {
      busy(false);
      if (!d.ok) { log("master failed: " + (d.error || "unknown"), "err"); return; }
      state.master = d;
      log("  " + d.file + "  " + (d.bytes / 1048576).toFixed(2) + " MB  " +
          d.resolution + "  " + d.container + "  " + d.elapsed_s + " s");
      log("  MaxCLL " + d.maxcll + "  MaxFALL " + d.maxfall + " nits  sidecar " + d.sidecar);
      var a = document.createElement("a");
      a.href = "/api/master/download?f=" + encodeURIComponent(d.file);
      a.download = d.file;
      document.body.appendChild(a); a.click(); a.remove();
    }).catch(function (e) { busy(false); log("master failed: " + e, "err"); });
  }

  /* ---- wiring ---------------------------------------------------------- */
  function take(file) {
    state.file = file;
    $("tc").textContent = "00:00:00:01";
    loadFrame();
  }

  function bind() {
    $("shots").innerHTML = SHOTS.map(function (s) {
      return '<div class="shot"><span class="mark"></span>' +
             '<span class="name">' + s[0] + '</span>' +
             '<span class="frames">' + s[1] + '</span>' +
             '<span class="peak">' + s[2] + '</span>' +
             '<span class="bar"><i style="width:' + (s[3] * 100) + '%"></i></span></div>';
    }).join("");
    $("shotCount").textContent = String(SHOTS.length);

    $("file").addEventListener("change", function (e) {
      if (e.target.files.length) { take(e.target.files[0]); }
    });
    ["dragover", "drop"].forEach(function (t) {
      window.addEventListener(t, function (e) { e.preventDefault(); });
    });
    window.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files.length) { take(e.dataTransfer.files[0]); }
    });
    $("viewer").addEventListener("dblclick", function () { $("file").click(); });

    $("mode").addEventListener("click", function (e) {
      var b = e.target.closest("button[data-mode]");
      if (!b) { return; }
      state.mode = b.dataset.mode;
      [].forEach.call(this.children, function (c) { c.classList.toggle("on", c === b); });
      recompose();
    });

    /* The compare. Click either side to hold it, or hold B for a momentary
       flip -- the eye is far better at spotting a change in one place than at
       tracking a seam travelling across the frame. */
    $("viewMode").addEventListener("click", function (e) {
      var b = e.target.closest("button[data-view]");
      if (!b) { return; }
      state.show = b.dataset.view;
      present();
    });
    $("plate").addEventListener("pointerdown", function (e) {
      if (e.button !== 0) { return; }
      state.flipHeld = true; present();
    });
    window.addEventListener("pointerup", function () {
      if (state.flipHeld) { state.flipHeld = false; present(); }
    });
    window.addEventListener("keydown", function (e) {
      if (e.repeat) { return; }
      if (e.key === "b" || e.key === "B") { state.flipHeld = true; present(); }
    });
    window.addEventListener("keyup", function (e) {
      if (e.key === "b" || e.key === "B") { state.flipHeld = false; present(); }
    });

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
    $("preserve").addEventListener("click", function () {
      state.preserve = !state.preserve;
      this.classList.toggle("on", state.preserve);
      $("preserveHint").textContent = state.preserve ? "do-no-harm" : "raw prediction";
      recompose();
    });
    $("btnReprocess").addEventListener("click", loadFrame);
    $("btnMaster").addEventListener("click", master);
  }

  function boot() {
    var canvas = $("gl");
    ctx = window.RudraGL ? window.RudraGL.create(canvas) : null;
    bind();
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
          log("drop a frame — the network runs once, then the grade is local");
        } else {
          log("no model loaded: " + (info.reason || "unknown"), "err");
        }
        busy(false);
      }).catch(function (e) {
        log("cannot reach the server: " + e, "err");
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
}());
