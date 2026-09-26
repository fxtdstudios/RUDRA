/* Scene-linear Rec.2020 luminance, matching server.py scopes(). */
(function (root) {
  "use strict";
  function luminanceSamples(rgba, peakNits) {
    if (rgba.length % 4 !== 0 || !Number.isFinite(peakNits) || peakNits <= 0) {
      throw new Error("Expected RGBA samples and a positive nits scale");
    }
    var out = new Float32Array(rgba.length / 4);
    for (var i = 0; i < out.length; i++) {
      var o = i * 4;
      out[i] = (0.2627 * rgba[o] + 0.6780 * rgba[o + 1] + 0.0593 * rgba[o + 2]) * peakNits;
    }
    return out;
  }
  function waveformDensity(luma, width, height, columns, bins, floor, ceiling) {
    if (luma.length !== width * height || width < 1 || height < 1 ||
        !Number.isInteger(columns) || !Number.isInteger(bins) || columns < 1 || bins < 1 ||
        !(floor > 0 && ceiling > floor)) { throw new Error("Invalid waveform geometry or scale"); }
    var counts = new Uint32Array(columns * bins), peak = 0, invalid = 0;
    var span = Math.log10(ceiling / floor);
    for (var y = 0; y < height; y++) for (var x = 0; x < width; x++) {
      var value = luma[y * width + x];
      if (!Number.isFinite(value)) { invalid++; continue; }
      var level = Math.log10(Math.min(ceiling, Math.max(floor, value)) / floor) / span;
      var col = Math.min(columns - 1, Math.floor(x * columns / width));
      var bin = Math.min(bins - 1, Math.floor(level * bins));
      var count = ++counts[col * bins + bin];
      peak = Math.max(peak, count);
    }
    return {counts: counts, columns: columns, bins: bins, peak: peak, invalid: invalid};
  }
  // ST 2084 inverse EOTF, absolute nits; BT.2100 non-constant-luminance chroma.
  function pqSignal(nits) {
    var p = Math.pow(Math.max(0, Math.min(10000, nits)) / 10000, 2610 / 16384);
    return Math.pow((3424 / 4096 + (2413 / 128) * p) / (1 + (2392 / 128) * p), 2523 / 32);
  }
  function signalChroma(r, g, b) {
    var y = .2627 * r + .6780 * g + .0593 * b;
    return {cb: (b - y) / 1.8814, cr: (r - y) / 1.4746};
  }
  function vectorTargets() {
    return [['R',1,0,0],['Y',1,1,0],['G',0,1,0],['C',0,1,1],['B',0,0,1],['M',1,0,1]].map(function (t) {
      var c = signalChroma(t[1] * .75, t[2] * .75, t[3] * .75);
      c.label = t[0]; return c;
    });
  }
  function vectorDensity(rgba, peakNits, bins) {
    if (rgba.length % 4 || !(peakNits > 0 && Number.isFinite(peakNits)) || !Number.isInteger(bins) || bins < 1) {
      throw new Error('Invalid vectorscope samples or geometry');
    }
    var counts = new Uint32Array(bins * bins), peak = 0, invalid = 0, clipped = 0;
    for (var i = 0; i < rgba.length; i += 4) {
      var r = rgba[i] * peakNits, g = rgba[i+1] * peakNits, b = rgba[i+2] * peakNits;
      if (![r,g,b].every(Number.isFinite)) { invalid++; continue; }
      if (Math.min(r,g,b) < 0 || Math.max(r,g,b) > 10000) { clipped++; }
      var c = signalChroma(pqSignal(r), pqSignal(g), pqSignal(b));
      var x = Math.max(0, Math.min(bins-1, Math.floor((c.cb + .5) * bins)));
      var y = Math.max(0, Math.min(bins-1, Math.floor((.5 - c.cr) * bins)));
      peak = Math.max(peak, ++counts[y * bins + x]);
    }
    return {counts: counts, peak: peak, bins: bins, invalid: invalid, clipped: clipped};
  }
  var api = {luminanceSamples: luminanceSamples, waveformDensity: waveformDensity,
    pqSignal: pqSignal, signalChroma: signalChroma, vectorTargets: vectorTargets, vectorDensity: vectorDensity};
  if (typeof module !== "undefined" && module.exports) { module.exports = api; }
  else { root.RudraScopeMath = api; }
}(typeof globalThis !== "undefined" ? globalThis : this));
