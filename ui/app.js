/* RUDRA Studio — the whole client.
   Replaces the earlier mock (app.js) plus the patch layer (live.js) that fixed
   it at runtime. Nothing here simulates: every number on screen came from
   /api/infer, and every label describes what is actually in the pane. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var state = {
    live: false, file: null, busy: false,
    mode: "all", strength: 1, peakEv: 0, preserve: true,
    view: "ab", wipe: 0.5, last: null, master: null
  };

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

  function log(line, kind) {
    var box = $("log");
    var row = document.createElement("div");
    if (kind) { row.className = kind; }
    row.textContent = line;
    box.appendChild(row);
    box.scrollTop = box.scrollHeight;
  }

  function displayNits() { return 203 * Math.pow(2, state.peakEv); }

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
    $("statusTime").textContent = m.source_resolution + " · " + m.elapsed_s + " s";
    $("srcInfo").textContent = m.source_resolution + " · " + m.elapsed_s + " s";
  }

  /* ---- scopes ---------------------------------------------------------- */
  var W = 460, H = 132, HW = 304, HH = 96;

  function ladder() {
    var lo = 0.05, hi = 4000, span = Math.log10(hi / lo), out = "";
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
      var span = Math.log10(4000 / 0.05);
      var y = H - (Math.log10(Math.min(Math.max(m.maxcll, 0.05), 4000) / 0.05) / span) * H;
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
        var x = (Math.log2(t[0] / 0.05) / Math.log2(4000 / 0.05)) * HW;
        marks += '<text x="' + x.toFixed(0) + '" y="110" font-family="IBM Plex Mono, monospace"' +
                 ' font-size="8.5" fill="#5c5c5c" text-anchor="middle">' + t[1] + "</text>";
      });
    var dw = (Math.log2(203 / 0.05) / Math.log2(4000 / 0.05)) * HW;
    $("hist").innerHTML = '<g fill="#b4b4b4" opacity="0.78">' + bars + "</g>" +
      '<line x1="0" y1="' + HH + '" x2="' + HW + '" y2="' + HH + '" stroke="#2c2c2c"/>' +
      '<line x1="' + dw.toFixed(1) + '" y1="0" x2="' + dw.toFixed(1) + '" y2="' + HH +
      '" stroke="#4a4a4a" stroke-dasharray="2 3"/>' + marks;
  }

  /* ---- viewer ---------------------------------------------------------- */
  function layoutWipe() {
    var plate = $("plate");
    if (plate.hidden) { return; }
    var top = $("imgHdr");
    var w = top.clientWidth, h = top.clientHeight;
    if (!w || !h) { return; }
    var img = $("imgBase");
    img.style.width = w + "px";
    img.style.height = h + "px";
    var x = state.view === "ab" ? state.wipe * w : (state.view === "base" ? w : 0);
    $("under").style.width = x + "px";
    $("wipe").style.left = x + "px";
    $("wipe").style.display = state.view === "ab" ? "block" : "none";
    $("grip").style.display = state.view === "ab" ? "flex" : "none";
    $("grip").style.left = x + "px";
    $("grip").style.top = (h / 2) + "px";
  }

  function bindWipe() {
    var plate = $("plate"), dragging = false;
    function move(clientX) {
      var r = plate.getBoundingClientRect();
      state.wipe = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
      layoutWipe();
    }
    plate.addEventListener("pointerdown", function (e) {
      if (state.view !== "ab") { return; }
      dragging = true; plate.setPointerCapture(e.pointerId); move(e.clientX);
    });
    plate.addEventListener("pointermove", function (e) { if (dragging) { move(e.clientX); } });
    window.addEventListener("pointerup", function () { dragging = false; });
    window.addEventListener("resize", layoutWipe);
  }

  /* ---- requests -------------------------------------------------------- */
  function params(extra) {
    var p = {
      strength: state.strength,
      display_nits: displayNits(),
      recovery_mode: state.mode,
      preserve_outside: state.preserve,
      tile_size: 512, tile_overlap: 64, max_side: 1600
    };
    if (extra) { Object.keys(extra).forEach(function (k) { p[k] = extra[k]; }); }
    return p;
  }

  function busy(on, label) {
    state.busy = on;
    $("btnMaster").disabled = on || !state.file || !state.live;
    $("btnReprocess").disabled = on || !state.file || !state.live;
    if (label) { $("btnMaster").textContent = label; }
    else { $("btnMaster").textContent = "Master EXR"; }
  }

  function infer() {
    if (!state.live || !state.file || state.busy) { return; }
    busy(true);
    log("reconstruct " + state.file.name + "  ×" + state.strength.toFixed(2) +
        "  " + state.mode + "  peak " + Math.round(displayNits()) + " nits");
    fetch("/api/infer", {
      method: "POST",
      headers: {"Content-Type": "application/octet-stream",
                "X-Rudra-Params": JSON.stringify(params())},
      body: state.file
    }).then(function (r) { return r.json(); }).then(function (d) {
      busy(false);
      if (!d.ok) { log("failed: " + (d.error || "unknown"), "err"); return; }
      state.last = d;
      $("empty").hidden = true;
      $("plate").hidden = false;
      $("imgHdr").onload = layoutWipe;
      $("imgHdr").src = d.hdr_png;
      $("imgBase").src = d.baseline_png;
      $("peakBadge").textContent = "Display peak " + Math.round(d.metrics.display_nits) + " nits";
      showMetrics(d.metrics);
      drawScopes(d.scopes, d.metrics);
      setTimeout(layoutWipe, 0);
      log("  MaxCLL " + Math.round(d.metrics.maxcll) + "  MaxFALL " +
          Math.round(d.metrics.maxfall) + " nits  headroom " +
          signed(d.metrics.headroom_highlight_stops, 2) + " st in mask");
    }).catch(function (e) { busy(false); log("request failed: " + e, "err"); });
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
    infer();
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
      infer();
    });
    $("viewMode").addEventListener("click", function (e) {
      var b = e.target.closest("button[data-view]");
      if (!b) { return; }
      state.view = b.dataset.view;
      [].forEach.call(this.children, function (c) { c.classList.toggle("on", c === b); });
      layoutWipe();
    });
    $("strength").addEventListener("input", function () {
      state.strength = parseFloat(this.value);
      $("strengthVal").textContent = state.strength.toFixed(2);
    });
    $("strength").addEventListener("change", infer);
    $("peak").addEventListener("input", function () {
      state.peakEv = parseFloat(this.value);
      $("peakVal").textContent = Math.round(displayNits()).toLocaleString("en-US");
    });
    $("peak").addEventListener("change", infer);
    $("preserve").addEventListener("click", function () {
      state.preserve = !state.preserve;
      this.classList.toggle("on", state.preserve);
      $("preserveHint").textContent = state.preserve ? "do-no-harm" : "unclamped";
      infer();
    });
    $("btnReprocess").addEventListener("click", infer);
    $("btnMaster").addEventListener("click", master);
    bindWipe();
  }

  // ?demo=1 runs the bundled frame through the loaded checkpoint as soon as the
  // page is live, so ui/capture_shot.py photographs the product working rather
  // than an empty drop target. It moved here when live.js was folded in; losing
  // it would have made the next README capture a picture of a placeholder.
  function autorun() {
    if (!/[?&]demo=1/.test(location.search)) { return; }
    fetch("assets/cinematic_hdr_sunset.png")
      .then(function (r) { return r.blob(); })
      .then(function (b) {
        take(new File([b], "cinematic_hdr_sunset.png", {type: "image/png"}));
      })
      .catch(function (e) { log("demo frame unavailable: " + e, "err"); });
  }

  function boot() {
    bind();
    fetch("/api/model").then(function (r) { return r.json(); }).then(function (info) {
      if (info.loaded) {
        state.live = true;
        $("ckpt").textContent = info.name + " · step " + info.step;
        $("device").textContent = info.gpu;
        $("lamp").classList.remove("off");
        log("model " + info.checkpoint);
        log("base_channels " + info.base_channels + "  max_hdr " + info.max_hdr +
            " (" + (info.max_hdr * 10000).toLocaleString("en-US") + " nits)");
        log("drop an SDR frame to reconstruct");
        autorun();
      } else {
        $("ckpt").textContent = "NO MODEL";
        $("ckpt").className = "pill warn";
        $("device").textContent = String(info.reason || "no checkpoint");
        log("no model loaded: " + (info.reason || "no checkpoint") +
            ". Nothing on this page is a measurement.", "err");
      }
    }).catch(function () {
      $("ckpt").textContent = "BACKEND UNREACHABLE";
      $("ckpt").className = "pill warn";
      log("no /api backend — run: python ui/server.py", "err");
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else { boot(); }
})();
