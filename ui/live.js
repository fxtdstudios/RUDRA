/* =============================================================================
   RUDRA Studio -- live backend bridge                      27 Aug 2026
   -----------------------------------------------------------------------------
   app.js drives a mock: it has no network calls, its "metrics" are arithmetic
   on slider positions, and its progress bars are setTimeout chains. This file
   loads AFTER it and replaces the parts that were pretending, without editing
   22 KB of working interaction code. Delete the <script> tag in index.html and
   the page reverts to the mock exactly as it was.

   What it does:
     - asks /api/model what checkpoint is really loaded, and says so in the
       header instead of the hard-coded "Flux 1 Dev"
     - POSTs the uploaded file to /api/infer and puts the REAL prediction in
       the right-hand pane of the compare slider
     - relabels the two dials, which claimed LPIPS and HDR-VDP JOD. Both need
       the ground-truth HDR to compute, and a file you just dragged in does not
       have one. They now show MaxCLL and peak nits, which are measurable.
   ============================================================================= */
(function () {
  "use strict";

  var state = { file: null, busy: false, live: false, lastMetrics: null };

  function $(id) { return document.getElementById(id); }

  function log(line, kind) {
    var body = $("consoleBody");
    if (!body) { return; }
    var row = document.createElement("div");
    row.className = "console-line" + (kind ? " " + kind : "");
    row.textContent = line;
    body.appendChild(row);
    body.scrollTop = body.scrollHeight;
  }

  function setPill(id, text, ok) {
    var pill = $(id);
    if (!pill) { return; }
    var span = pill.querySelector("span");
    if (span) { span.textContent = text; }
    pill.classList.toggle("active", ok !== false);
  }

  /* ---- dials -------------------------------------------------------------
     Relabelled, not just re-fed. Leaving "LPIPS" over a number that is not
     LPIPS is how the mock misled in the first place.                        */
  // The left pane carries the analytic inverse-ACES baseline, not the SDR --
  // that is the comparison worth showing. The markup's badge said "DISPLAY SDR
  // (CLIPPED)", which described the mock's behaviour and is now simply wrong.
  function relabelPanes() {
    var sdr = document.querySelector(".badge-sdr");
    var hdr = document.querySelector(".badge-hdr");
    if (sdr) { sdr.textContent = "INVERSE-ACES BASELINE"; }
    if (hdr) { hdr.textContent = "RUDRA RECONSTRUCTION"; }
  }

  function relabelDials() {
    var boxes = document.querySelectorAll(".dial-box");
    if (boxes.length < 2) { return; }
    var labels = [
      ["MaxCLL", "nits, max(R,G,B) -- CTA-861.3"],
      ["Peak", "nits in this frame"]
    ];
    for (var i = 0; i < 2; i++) {
      var label = boxes[i].querySelector(".dial-label");
      var desc = boxes[i].querySelector(".dial-description");
      if (label) { label.textContent = labels[i][0]; }
      if (desc) { desc.textContent = labels[i][1]; }
    }
    var heading = document.querySelector(".dials-container");
    if (heading && heading.previousElementSibling) {
      var span = heading.previousElementSibling.querySelector("span");
      if (span) { span.textContent = "Measured HDR Statistics"; }
    }
  }

  function showDials(metrics) {
    // Both dials are log-scaled against 10,000 nits: a 4,000-nit peak should
    // not read as 40% of the way round when the codes above it are so few.
    function frac(nits) {
      return Math.max(0, Math.min(1, Math.log10(Math.max(nits, 1)) / 4));
    }
    if (typeof setDialValue === "function") {
      setDialValue("dialLpips", "textLpips", Math.round(metrics.maxcll), 1.0);
      setDialValue("dialJod", "textJod", Math.round(metrics.peak_nits), 1.0);
    }
    var a = $("dialLpips"), b = $("dialJod");
    var c = 282.74;
    if (a) { a.style.strokeDashoffset = String(c - frac(metrics.maxcll) * c); }
    if (b) { b.style.strokeDashoffset = String(c - frac(metrics.peak_nits) * c); }
  }

  /* ---- parameters --------------------------------------------------------
     Only the controls that map onto something the model actually has:
       Gating          -> residual_strength (per-pixel ITM aggressiveness)
       Exposure        -> display peak for the preview, NOT a change to the
                          prediction. Raising it lifts the clip point so
                          reconstructed highlights become visible on an SDR
                          monitor. The HDR itself is identical.
     Patch size and projection belong to the old Flux backbone and do nothing
     here; they are left alone rather than faked.                             */
  function currentParams() {
    var gating = parseFloat((($("sliderGating") || {}).value) || "1");
    var exposure = parseFloat((($("sliderExposure") || {}).value) || "0");
    return {
      strength: isNaN(gating) ? 1.0 : gating,
      display_nits: 203.0 * Math.pow(2.0, isNaN(exposure) ? 0 : exposure),
      recovery_mode: "all",
      preserve_outside: true,
      tile_size: 512,
      tile_overlap: 64,
      max_side: 1600
    };
  }

  function busy(on, text) {
    state.busy = on;
    var loader = $("btnLoader"), label = $("btnText"), button = $("btnMaster");
    if (loader) { loader.style.display = on ? "inline-block" : "none"; }
    if (label) { label.textContent = text || (on ? "Reconstructing..." : "Master HDR Image"); }
    if (button) { button.disabled = on; button.style.opacity = on ? "0.6" : "1"; }
  }

  function infer() {
    if (!state.live || !state.file || state.busy) { return; }
    var params = currentParams();
    busy(true);
    log("infer: " + state.file.name + "  strength " + params.strength.toFixed(2) +
        "  display peak " + Math.round(params.display_nits) + " nits");

    fetch("/api/infer", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Rudra-Params": JSON.stringify(params)
      },
      body: state.file
    }).then(function (r) { return r.json(); }).then(function (data) {
      busy(false);
      if (!data.ok) {
        log("FAILED: " + (data.error || "unknown"), "error");
        return;
      }
      var m = data.metrics;
      state.lastMetrics = m;
      var sdrImg = $("sdrImage"), hdrImg = $("hdrImage");
      // Left of the slider: the analytic inverse-ACES baseline, displayed at
      // the same peak. So the wipe compares the model against what you get for
      // free -- not against the SDR, which would flatter it.
      if (sdrImg) { sdrImg.src = data.baseline_png; sdrImg.style.filter = "none"; }
      if (hdrImg) { hdrImg.src = data.hdr_png; hdrImg.style.filter = "none"; }
      showDials(m);
      log("  " + m.resolution + " in " + m.elapsed_s + "s   MaxCLL " + m.maxcll +
          "  MaxFALL " + m.maxfall + " nits");
      log("  peak " + m.peak_nits + " nits vs baseline " + m.baseline_peak_nits +
          "  (+" + m.headroom_stops + " stops of headroom)");
      log("  above diffuse white " + m.above_diffuse_white_pct + "%   above 1000 nits " +
          m.above_1000_nits_pct + "%");
      log("  masks fired on " + m.highlight_mask_pct + "% highlight / " +
          m.shadow_mask_pct + "% shadow");
    }).catch(function (err) {
      busy(false);
      log("request failed: " + err, "error");
    });
  }

  /* ---- wiring ----------------------------------------------------------- */
  function hookUploads() {
    var input = $("fileInput");
    if (input) {
      input.addEventListener("change", function (e) {
        if (e.target.files && e.target.files.length) {
          state.file = e.target.files[0];
          infer();
        }
      });
    }
    ["uploadCard", "viewerCard"].forEach(function (id) {
      var card = $(id);
      if (!card) { return; }
      card.addEventListener("drop", function (e) {
        if (e.dataTransfer && e.dataTransfer.files.length) {
          state.file = e.dataTransfer.files[0];
          infer();
        }
      });
    });
    var master = $("btnMaster");
    if (master) {
      // app.js binds triggerMastering() inline, which fakes a progress bar.
      master.removeAttribute("onclick");
      master.addEventListener("click", function () {
        if (!state.file) { log("load an SDR image first"); return; }
        infer();
      });
    }
    // Re-run when a control that actually affects the result settles.
    ["sliderGating", "sliderExposure"].forEach(function (id) {
      var slider = $(id);
      if (!slider) { return; }
      slider.addEventListener("change", function () { infer(); });
    });
  }

  // ?demo=1 runs the bundled asset through the model as soon as the page is
  // live. Exists so a headless screenshot captures the product doing its job --
  // real prediction, real MaxCLL -- instead of an empty upload card.
  function autorun() {
    if (!/[?&]demo=1/.test(location.search)) { return; }
    fetch("assets/cinematic_hdr_sunset.png")
      .then(function (r) { return r.blob(); })
      .then(function (b) {
        state.file = new File([b], "cinematic_hdr_sunset.png", {type: "image/png"});
        var card = $("uploadCard"), viewer = $("viewerCard");
        if (card) { card.style.display = "none"; }
        if (viewer) { viewer.style.display = "block"; }
        var handle = $("sliderHandle"), layer = $("hdrLayer");
        if (layer) { layer.style.clipPath = "polygon(50% 0, 100% 0, 100% 100%, 50% 100%)"; }
        if (handle) { handle.style.left = "50%"; }
        infer();
      });
  }

  function boot() {
    relabelDials();
    fetch("/api/model").then(function (r) { return r.json(); }).then(function (info) {
      if (info.loaded) {
        state.live = true;
        relabelPanes();
        setPill("cudaStatus", info.gpu + "  (" + info.device + ")", true);
        setPill("backboneStatus", "SDR2HDRNet " + (info.name || ""), true);
        log("model: " + info.checkpoint);
        log("step " + info.step + ", base_channels " + info.base_channels +
            ", max_hdr " + info.max_hdr + " (" + (info.max_hdr * 10000) +
            " nits), preserve_outside on");
        log("drop an SDR image anywhere to reconstruct it");
      } else {
        setPill("cudaStatus", "NO MODEL -- demo mode", false);
        setPill("backboneStatus", String(info.reason || "no checkpoint"), false);
        log("demo mode: " + (info.reason || "no checkpoint found") +
            ". Numbers on this page are NOT measurements.", "error");
      }
      hookUploads();
      autorun();
    }).catch(function () {
      setPill("cudaStatus", "backend not reachable -- demo mode", false);
      log("no /api backend: serving statically. Run 'python ui/server.py'.", "error");
      // Hook the handlers anyway. Without this a page loaded while the server
      // was down stays inert even after it comes back, and every drop silently
      // falls through to app.js's simulated path -- which is how a capture on
      // 28 Aug 2026 came back showing the mock's LPIPS 0.082 / JOD 9.85 dials.
      hookUploads();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
