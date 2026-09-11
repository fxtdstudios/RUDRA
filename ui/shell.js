/* RUDRA Studio — shell behaviour.

   Everything in here is chrome: workspace switch, inspector tabs, the icon
   rail, framing guides, and the viewer toolbar's zoom segment. It deliberately
   owns no state that app.js owns. Where a shell control does something the
   application already does, it CLICKS THE EXISTING CONTROL rather than
   reimplementing it, so there is exactly one code path for every action and
   the two cannot drift.

   Loaded after app.js. If app.js failed, the shell still runs and the page is
   still legible, which is the point of keeping them apart. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  /* ---- workspace ------------------------------------------------------
     Simple hides the scopes dock, the log and the explanatory notes. It
     hides nothing that is the only way to reach a behaviour: every control
     Simple removes is also on a menu. */
  function workspace(mode) {
    document.body.classList.toggle("ws-simple", mode === "simple");
    document.body.classList.toggle("ws-full", mode !== "simple");
    var simple = $("wsSimple"), full = $("wsFull");
    if (simple) { simple.classList.toggle("on", mode === "simple"); }
    if (full) { full.classList.toggle("on", mode !== "simple"); }
    if ($("iScopes")) { $("iScopes").classList.toggle("on", mode !== "simple"); }
    window.dispatchEvent(new Event("resize"));
  }
  if ($("wsSimple")) { $("wsSimple").addEventListener("click", function () { workspace("simple"); }); }
  if ($("wsFull")) { $("wsFull").addEventListener("click", function () { workspace("full"); }); }

  /* ---- inspector tabs -------------------------------------------------- */
  var tabs = $("itabs");
  if (tabs) {
    tabs.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-tab]");
      if (!b) { return; }
      showTab(b.dataset.tab);
    });
  }
  function showTab(name) {
    if (!tabs) { return; }
    Array.prototype.forEach.call(tabs.querySelectorAll("button"), function (x) {
      x.classList.toggle("on", x.dataset.tab === name);
    });
    Array.prototype.forEach.call(document.querySelectorAll(".ipanel"), function (p) {
      p.classList.toggle("on", p.dataset.panel === name);
    });
  }
  /* The menus reach controls that may be on a tab that is not showing. A
     menu item that changes something invisible is a menu item the user will
     believe did nothing, so bring its tab forward. */
  var TAB_FOR_ACT = {
    "reset-regions": "grade",
    "container-aces": "deliver",
    "container-linear": "deliver",
    "copy-delivery": "deliver",
    "mode-all": "rec", "mode-highlights": "rec", "mode-shadows": "rec",
    "mode-off": "rec", "preserve": "rec",
    "strength-up": "rec", "strength-down": "rec", "reset-recon": "rec"
  };
  document.addEventListener("click", function (e) {
    var b = e.target.closest(".drop button[data-act]");
    if (b && TAB_FOR_ACT[b.dataset.act]) { showTab(TAB_FOR_ACT[b.dataset.act]); }
  }, true);

  /* ---- icon rail ------------------------------------------------------
     Drives the same menu actions, so the check marks in Window stay right. */
  function menuAct(act) {
    var b = document.querySelector('.drop button[data-act="' + act + '"]');
    if (b) { b.click(); }
  }
  function railToggle(id, act, watch) {
    var btn = $(id);
    if (!btn) { return; }
    btn.addEventListener("click", function () {
      menuAct(act);
      var target = $(watch);
      if (target) { btn.classList.toggle("on", !target.classList.contains("hidden")); }
    });
  }
  railToggle("iMedia", "rail-left", "railLeft");
  railToggle("iInspector", "rail-right", "railRight");
  railToggle("iScopes", "scopes", "scopes");
  if ($("iHelp")) { $("iHelp").addEventListener("click", function () { menuAct("shortcuts"); }); }

  /* ---- open ------------------------------------------------------------ */
  if ($("browseBtn")) { $("browseBtn").addEventListener("click", function () { menuAct("open"); }); }
  var dz = $("dropzone");
  if (dz) {
    ["dragenter", "dragover"].forEach(function (t) {
      window.addEventListener(t, function () { dz.classList.add("hot"); });
    });
    ["dragleave", "drop"].forEach(function (t) {
      window.addEventListener(t, function () { dz.classList.remove("hot"); });
    });
  }

  /* ---- the drop zone earns its space only while it is needed ----------
     A permanent 90px invitation to open something, sitting above the list of
     things you already opened, is the commonest way a left rail turns into
     wasted rail. It comes back when the last frame is closed. */
  var shotCount = $("shotCount");
  if (dz && shotCount) {
    var syncDrop = function () {
      var n = Number(shotCount.textContent || "0");
      dz.classList.toggle("tucked", n > 0);
    };
    syncDrop();
    new MutationObserver(syncDrop).observe(shotCount, {childList: true, characterData: true, subtree: true});
  }

  /* ---- framing guides -------------------------------------------------- */
  var guideBtn = $("guideBtn"), plate = $("plate");
  if (guideBtn && plate) {
    guideBtn.addEventListener("click", function () {
      var on = plate.classList.toggle("guides-on");
      guideBtn.classList.toggle("on", on);
    });
  }

  /* ---- zoom segment ---------------------------------------------------
     Mirrors Window ▸ Fit / Actual pixels. The class on #viewer is the truth;
     this only reflects it. */
  var zoomSeg = $("zoomSeg"), viewer = $("viewer");
  if (zoomSeg && viewer) {
    zoomSeg.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-zoom]");
      if (!b) { return; }
      menuAct(b.dataset.zoom === "actual" ? "zoom-actual" : "zoom-fit");
    });
    new MutationObserver(function () {
      var actual = viewer.classList.contains("actual");
      Array.prototype.forEach.call(zoomSeg.querySelectorAll("button"), function (x) {
        x.classList.toggle("on", (x.dataset.zoom === "actual") === actual);
      });
    }).observe(viewer, {attributes: true, attributeFilter: ["class"]});
  }

  /* ---- view transform read-out ---------------------------------------
     Says what the picture on screen is being shown as, taken from the
     delivery container the app already tracks. It reports; it does not set. */
  var cf = $("containerField"), vt = $("viewTransform");
  if (cf && vt) {
    var readTransform = function () {
      var t = (cf.textContent || "").toLowerCase();
      vt.textContent = t.indexOf("aces") >= 0 ? "linear · AP0" : "linear · Rec.2020";
    };
    readTransform();
    new MutationObserver(readTransform).observe(cf, {childList: true, characterData: true, subtree: true});
  }
})();
