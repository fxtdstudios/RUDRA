/* RUDRA Studio — the GPU compositor.

   The network runs ONCE per frame on the server and hands back three raw
   fields: the residual, and the highlight and shadow masks. Everything the
   controls do after that — residual strength, recovery mode, preserve, display
   peak, the A/B flip — is arithmetic on those fields, so it belongs on the
   viewer's GPU, not on a round trip.

   The composite here is a line-for-line port of SDR2HDRNet.forward's tail
   (rudra/sdr2hdr.py). It has to be: what you see must be what
   `Master EXR` writes. tests/test_frame_fields_2026_08_28.py checks the two
   against each other on the torch side, and tests/webgl_parity checks this
   shader against a numpy reference.

   Nothing in here talks to the network or the DOM beyond its canvas. */
(function () {
  "use strict";

  var LOG_SCALE = 16.0;
  var MAX_HDR = 4.0;
  var PEAK_NITS = 10000.0;
  var MODES = {all: 0, highlights: 1, shadows: 2, off: 3};

  var VERT = [
    "#version 300 es",
    "in vec2 aPos;",
    "out vec2 vUV;",
    "void main(){ vUV = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }"
  ].join("\n");

  /* Shared maths. log1p/expm1 are not in GLSL ES, and the naive forms lose
     the deep shadows: log(1+x) for x ~ 1e-7 is where a 0.05-nit floor lives. */
  var COMMON = [
    "precision highp float;",
    "precision highp sampler2D;",
    "const float LOG_SCALE = 16.0;",
    "const float MAX_HDR = 4.0;",
    "vec3 log1p3(vec3 x){",
    "  vec3 big = log(1.0 + x);",
    "  vec3 small = x - x*x*0.5 + x*x*x*(1.0/3.0);",
    "  return mix(big, small, lessThan(abs(x), vec3(1e-4)));",
    "}",
    "vec3 expm13(vec3 x){",
    "  vec3 big = exp(x) - 1.0;",
    "  vec3 small = x + x*x*0.5 + x*x*x*(1.0/6.0);",
    "  return mix(big, small, lessThan(abs(x), vec3(1e-4)));",
    "}",
    "vec3 srgbToLinear(vec3 c){",
    "  c = clamp(c, 0.0, 1.0);",
    "  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)),",
    "             greaterThan(c, vec3(0.04045)));",
    "}",
    "vec3 linearToSrgb(vec3 x){",
    "  x = clamp(x, 0.0, 1.0);",
    "  return mix(12.92 * x, 1.055 * pow(x, vec3(1.0 / 2.4)) - 0.055,",
    "             greaterThan(x, vec3(0.0031308)));",
    "}",
    /* inverse_aces_approx, including the two clamps that keep it finite:
       display linear is capped just under one because a clipped pixel is
       unknowable, and the denominator is held strictly negative. */
    "vec3 inverseAces(vec3 displayLinear){",
    "  vec3 y = clamp(displayLinear, 0.0, 0.995);",
    "  vec3 qa = y * 2.43 - 2.51;",
    "  vec3 qb = y * 0.59 - 0.03;",
    "  vec3 qc = y * 0.14;",
    "  vec3 disc = max(qb * qb - 4.0 * qa * qc, 0.0);",
    "  vec3 den = min(2.0 * qa, vec3(-1e-7));",
    "  vec3 s = sqrt(disc);",
    "  return max(max((-qb - s) / den, (-qb + s) / den), vec3(0.0));",
    "}",
    "vec3 baselineOf(vec3 sdr){",
    "  return inverseAces(srgbToLinear(sdr)) * (2.0 * 203.0 / 10000.0);",
    "}",
    /* Region EV. A port of rudra/delivery/controls.py's qualifier_mask:
       a soft window in log luminance, feathered symmetrically in stops so a
       qualifier never puts a hard edge through a gradient, with the offsets
       adding in stops. The identical numbers reach the EXR writer, so what
       you grade here is what Master delivers. */
    "uniform vec3 uRegionLo;",
    "uniform vec3 uRegionHi;",
    "uniform vec3 uRegionEv;",
    "uniform float uRegionSoft;",
    "vec3 applyRegions(vec3 pred){",
    "  float logY = log2(max(dot(pred * 10000.0, vec3(0.2627, 0.6780, 0.0593)), 1e-6));",
    "  float total = 0.0;",
    "  for (int i = 0; i < 3; i++) {",
    "    float rise = clamp((logY - (uRegionLo[i] - uRegionSoft)) / uRegionSoft, 0.0, 1.0);",
    "    float fall = clamp(((uRegionHi[i] + uRegionSoft) - logY) / uRegionSoft, 0.0, 1.0);",
    "    float m = min(rise, fall);",
    "    total += uRegionEv[i] * m * m * (3.0 - 2.0 * m);",
    "  }",
    "  return clamp(pred * exp2(total), 0.0, MAX_HDR);",
    "}"
  ].join("\n");

  var COMPOSITE = ["#version 300 es", COMMON,
    "in vec2 vUV;",
    "out vec4 oCol;",
    "uniform sampler2D uSdr;",
    "uniform sampler2D uFields;",   // rgb = residual, a = highlight mask
    "uniform sampler2D uShadow;",   // r   = shadow mask
    "uniform float uStrength;",
    "uniform int uMode;",
    "uniform bool uPreserve;",
    "uniform bool uBaselineOnly;",
    "uniform float uShadowWeight;",
    "void main(){",
    "  vec3 sdr = texture(uSdr, vUV).rgb;",
    "  vec3 base = baselineOf(sdr);",
    "  if (uBaselineOnly) { oCol = vec4(base, 1.0); return; }",
    "  vec4 f = texture(uFields, vUV);",
    "  float shadow = texture(uShadow, vUV).r;",
    "  float y = dot(clamp(sdr, 0.0, 1.0), vec3(0.2126, 0.7152, 0.0722));",
    "  float hp = 1.0 / (1.0 + exp(-((y - 0.82) * 24.0)));",
    "  float sp = 1.0 / (1.0 + exp(-((0.10 - y) * 24.0)));",
    /* The learned shadow weight. It multiplies the shadow PRIOR, never the
       learned masks -- exactly what SDR2HDRNet.forward does, and exactly what
       `--recovery-mode highlights` measured. 1.0 is the shipped behaviour, so a
       checkpoint without the gate composes as it always did. Without this the
       page would show the shadow arm always on while the Master EXR applied the
       model's own judgement: the two would disagree, which is the class of bug
       tests/webgl_parity exists to prevent. */
    "  sp *= uShadowWeight;",
    "  float gate = uMode == 0 ? max(hp, sp) : (uMode == 1 ? hp : (uMode == 2 ? sp : 0.0));",
    "  gate *= uStrength;",
    "  vec3 predLog = clamp(log1p3(base * LOG_SCALE) + f.rgb * gate,",
    "                       0.0, log(1.0 + MAX_HDR * LOG_SCALE));",
    "  vec3 pred = expm13(predLog) / LOG_SCALE;",
    "  if (uPreserve) { pred = base + max(f.a, shadow) * (pred - base); }",
    "  oCol = vec4(applyRegions(pred), 1.0);",
    "}"].join("\n");

  /* Exposure and clip, no tone curve — the same display_map() the server used
     to do. Any S-curve here would hide the highlights the page exists to show. */
  var DISPLAY = ["#version 300 es", COMMON,
    "in vec2 vUV;",
    "out vec4 oCol;",
    "uniform sampler2D uHdr;",
    /* The BEFORE side of the wipe: the analytic baseline, always bound, so
       toggling the wipe costs no rebind and no recomposite. */
    "uniform sampler2D uHdrBefore;",
    "uniform float uScale;",
    /* Wipe position in [0,1] across the canvas, or a negative number for off.
       Left of it is the baseline, right of it is the reconstruction, which is
       the order the words "before and after" are read in. */
    "uniform float uWipe;",
    "uniform float uWipeHalfWidth;",
    "void main(){",
    /* The one flip in the whole pipeline, and it belongs here. Every texture
       is uploaded in array order -- row 0 is the TOP of the image -- and the
       compositor keeps that order end to end, which is what readComposite(),
       sample()/sourceIndex() and the Master EXR all assume. But the default
       framebuffer puts row 0 at the BOTTOM of the canvas, so presenting with
       vUV unchanged shows the frame upside down. Flip V here, at the last
       step, and nothing upstream has to know. Do NOT "fix" this with
       UNPACK_FLIP_Y_WEBGL at upload: that inverts the model/base targets too
       and silently flips the readback paths, which no numeric test would
       catch because they all compare the float buffer, never the canvas. */
    "  vec2 uv = vec2(vUV.x, 1.0 - vUV.y);",
    "  vec3 hdr = (uWipe >= 0.0 && vUV.x < uWipe) ? texture(uHdrBefore, uv).rgb",
    "                                             : texture(uHdr, uv).rgb;",
    "  vec3 col = linearToSrgb(clamp(hdr * uScale, 0.0, 1.0));",
    /* The handle is drawn here rather than as a DOM overlay so it cannot drift
       from the split it marks: one pixel of disagreement between the line and
       the seam is exactly the artefact a wipe exists to rule out. */
    "  if (uWipe >= 0.0 && abs(vUV.x - uWipe) < uWipeHalfWidth) {",
    "    col = vec3(1.0) - col;",
    "  }",
    "  oCol = vec4(col, 1.0);",
    "}"].join("\n");

  /* Point-sample into a smaller float target. The scopes and the
     distribution numbers do not need every pixel, and reading 23 MB back per
     slider move would undo the whole point of doing this on the GPU. The
     exact peak still comes from the reduction below. */
  var COPY = ["#version 300 es", "precision highp float; precision highp sampler2D;",
    "in vec2 vUV;",
    "out vec4 oCol;",
    "uniform sampler2D uSrc;",
    "void main(){ oCol = texture(uSrc, vUV); }"].join("\n");

  /* One reduction step: a 2x2 box of the source, combined by max or sum.
     Run to 1x1 it is exact, which is what MaxCLL and MaxFALL need. */
  var REDUCE = ["#version 300 es", "precision highp float; precision highp sampler2D;",
    "in vec2 vUV;",
    "out vec4 oCol;",
    "uniform sampler2D uSrc;",
    "uniform vec2 uSrcSize;",
    "uniform vec2 uDstSize;",
    "uniform int uOp;",              // 0 = max, 1 = sum
    "uniform bool uFirst;",          // first pass reads RGB and takes max(R,G,B)
    "float take(vec2 p){",
    "  vec4 t = texelFetch(uSrc, ivec2(p), 0);",
    "  return uFirst ? max(max(t.r, t.g), t.b) : t.r;",
    "}",
    "void main(){",
    "  vec2 d = floor(vUV * uDstSize);",
    "  vec2 s = d * 2.0;",
    "  float acc = uOp == 0 ? -1e30 : 0.0;",
    "  for (int j = 0; j < 2; j++) {",
    "    for (int i = 0; i < 2; i++) {",
    "      vec2 p = s + vec2(float(i), float(j));",
    "      if (p.x >= uSrcSize.x || p.y >= uSrcSize.y) { continue; }",
    "      float v = take(p);",
    "      acc = uOp == 0 ? max(acc, v) : acc + v;",
    "    }",
    "  }",
    "  oCol = vec4(acc, 0.0, 0.0, 1.0);",
    "}"].join("\n");

  function compile(gl, type, src) {
    var sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error("shader: " + gl.getShaderInfoLog(sh) + "\n" + src);
    }
    return sh;
  }

  function program(gl, fragSrc) {
    var p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fragSrc));
    gl.bindAttribLocation(p, 0, "aPos");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error("link: " + gl.getProgramInfoLog(p));
    }
    return p;
  }

  /* getUniformLocation is a string lookup into the linked program. The
     reduction ladder called it once per uniform per level -- about 55 lookups
     per reduce, three reduces per measurement -- and the composite called ten
     more on every slider move. They never change for a linked program. */
  function locator(gl) {
    var cache = new Map();
    return function (program, name) {
      var byName = cache.get(program);
      if (!byName) { byName = new Map(); cache.set(program, byName); }
      if (!byName.has(name)) { byName.set(name, gl.getUniformLocation(program, name)); }
      return byName.get(name);
    };
  }

  function texture(gl, unit) {
    var t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  var SCRATCH_UNIT = 7;   // never holds a frame texture

  function target(gl, width, height, internal) {
    var t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + SCRATCH_UNIT);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, gl.RGBA,
                  internal === gl.RGBA8 ? gl.UNSIGNED_BYTE : gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    var fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return {tex: t, fb: fb, w: width, h: height};
  }

  function create(canvas) {
    var gl = canvas.getContext("webgl2", {alpha: false, antialias: false,
                                          preserveDrawingBuffer: true});
    if (!gl) { return null; }
    if (!gl.getExtension("EXT_color_buffer_float")) { return null; }

    var uniform = locator(gl);
    var quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER,
                  new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    var progComposite = program(gl, COMPOSITE);
    var progDisplay = program(gl, DISPLAY);
    var progReduce = program(gl, REDUCE);
    var progCopy = program(gl, COPY);

    var texSdr = texture(gl, 0);
    var texFields = texture(gl, 1);
    var texShadow = texture(gl, 2);

    var frame = null;          // {w, h}
    var model = null;          // RGBA32F target, RUDRA composite
    var base = null;           // RGBA32F target, inverse-ACES baseline
    var chain = [];            // reduction ladder
    var small = null;          // capped-size sample of the model composite
    var smallBase = null;      // and of the baseline, which never changes
    var baseSample = null;     // cached readback of smallBase
    var params = {strength: 1, mode: "all", preserve: true,
                  displayNits: 203, show: "model", shadowWeight: 1.0,
                  wipe: -1,          // <0 is off; otherwise 0..1 across the canvas
                  wipeHalfWidth: 0.0012,
                  regions: [{label: "highlights", low_nits: 400, high_nits: 2000, ev: 0},
                            {label: "speculars", low_nits: 2000, high_nits: 8000, ev: 0},
                            {label: "shadows", low_nits: 0.05, high_nits: 12, ev: 0}],
                  regionSoft: 1.0};

    function draw(prog, w, h, fb) {
      gl.useProgram(prog);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb || null);
      gl.viewport(0, 0, w, h);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    function buildChain() {
      chain.forEach(function (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); });
      chain = [];
      var w = frame.w, h = frame.h;
      while (w > 1 || h > 1) {
        w = Math.max(1, Math.ceil(w / 2));
        h = Math.max(1, Math.ceil(h / 2));
        chain.push(target(gl, w, h, gl.RGBA32F));
      }
      if (!chain.length) { chain.push(target(gl, 1, 1, gl.RGBA32F)); }
    }

    /* Exact max and sum of max(R,G,B), by reduction to a single texel.
       A readback of a downsampled image would miss the one specular pixel
       MaxCLL is entirely about. */
    function reduce(sourceTarget, op) {
      gl.useProgram(progReduce);
      gl.uniform1i(uniform(progReduce, "uSrc"), 7);
      gl.uniform1i(uniform(progReduce, "uOp"), op);
      var srcTex = sourceTarget.tex, sw = sourceTarget.w, sh = sourceTarget.h, first = true;
      for (var i = 0; i < chain.length; i++) {
        var dst = chain[i];
        gl.activeTexture(gl.TEXTURE7);
        gl.bindTexture(gl.TEXTURE_2D, srcTex);
        gl.uniform2f(uniform(progReduce, "uSrcSize"), sw, sh);
        gl.uniform2f(uniform(progReduce, "uDstSize"), dst.w, dst.h);
        gl.uniform1i(uniform(progReduce, "uFirst"), first ? 1 : 0);
        draw(progReduce, dst.w, dst.h, dst.fb);
        srcTex = dst.tex; sw = dst.w; sh = dst.h; first = false;
      }
      var out = new Float32Array(4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, chain[chain.length - 1].fb);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, out);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return out[0];
    }

    function setFrame(f) {
      frame = {w: f.width, h: f.height};
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texSdr);
      var precise = f.sdr instanceof Float32Array;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, precise ? gl.RGB32F : gl.RGB8, f.width, f.height, 0,
                    gl.RGB, precise ? gl.FLOAT : gl.UNSIGNED_BYTE, f.sdr);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, texFields);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, f.width, f.height, 0,
                    gl.RGBA, gl.HALF_FLOAT, f.fields);

      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, texShadow);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, f.width, f.height, 0,
                    gl.RED, gl.HALF_FLOAT, f.shadow);

      [model, base].forEach(function (t) {
        if (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); }
      });
      model = target(gl, f.width, f.height, gl.RGBA32F);
      base = target(gl, f.width, f.height, gl.RGBA32F);
      [small, smallBase].forEach(function (t) {
        if (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); }
      });
      var s = sampleSize();
      small = target(gl, s.width, s.height, gl.RGBA32F);
      smallBase = target(gl, s.width, s.height, gl.RGBA32F);
      baseSample = null;
      buildChain();
      compositeBaseline();
      composite();
    }

    function bindComposite(baselineOnly) {
      gl.useProgram(progComposite);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texSdr);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texFields);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, texShadow);
      gl.uniform1i(uniform(progComposite, "uSdr"), 0);
      gl.uniform1i(uniform(progComposite, "uFields"), 1);
      gl.uniform1i(uniform(progComposite, "uShadow"), 2);
      gl.uniform1f(uniform(progComposite, "uStrength"), params.strength);
      gl.uniform1i(uniform(progComposite, "uMode"),
                   MODES[params.mode] === undefined ? 0 : MODES[params.mode]);
      gl.uniform1i(uniform(progComposite, "uPreserve"), params.preserve ? 1 : 0);
      gl.uniform1i(uniform(progComposite, "uBaselineOnly"), baselineOnly ? 1 : 0);
      gl.uniform1f(uniform(progComposite, "uShadowWeight"), params.shadowWeight);
      var lo = [], hi = [], ev = [];
      for (var i = 0; i < 3; i++) {
        var band = params.regions[i] || {low_nits: 1, high_nits: 1, ev: 0};
        lo.push(Math.log2(Math.max(band.low_nits, 1e-6)));
        hi.push(Math.log2(Math.max(band.high_nits, 1e-6)));
        ev.push(band.ev || 0);
      }
      gl.uniform3fv(uniform(progComposite, "uRegionLo"), lo);
      gl.uniform3fv(uniform(progComposite, "uRegionHi"), hi);
      gl.uniform3fv(uniform(progComposite, "uRegionEv"), ev);
      gl.uniform1f(uniform(progComposite, "uRegionSoft"), params.regionSoft);
    }

    function compositeBaseline() {
      if (!frame) { return; }
      bindComposite(true);
      draw(progComposite, frame.w, frame.h, base.fb);
    }

    function composite() {
      if (!frame) { return; }
      bindComposite(false);
      draw(progComposite, frame.w, frame.h, model.fb);
    }

    function present() {
      if (!frame) { return; }
      var wiping = params.wipe >= 0.0;
      // While wiping, the right-hand side is always the reconstruction: a wipe
      // of the baseline against itself is a blank comparison, and B-to-flip
      // already covers "show me the baseline full frame".
      var source = (!wiping && params.show === "baseline") ? base : model;
      if (canvas.width !== frame.w || canvas.height !== frame.h) {
        canvas.width = frame.w; canvas.height = frame.h;
      }
      gl.useProgram(progDisplay);
      gl.activeTexture(gl.TEXTURE6);
      gl.bindTexture(gl.TEXTURE_2D, source.tex);
      gl.uniform1i(uniform(progDisplay, "uHdr"), 6);
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D, base.tex);
      gl.uniform1i(uniform(progDisplay, "uHdrBefore"), 7);
      gl.uniform1f(uniform(progDisplay, "uScale"),
                   PEAK_NITS / Math.max(params.displayNits, 1e-3));
      gl.uniform1f(uniform(progDisplay, "uWipe"), wiping ? params.wipe : -1.0);
      gl.uniform1f(uniform(progDisplay, "uWipeHalfWidth"), params.wipeHalfWidth);
      draw(progDisplay, frame.w, frame.h, null);
    }

    function readComposite(which) {
      var t = which === "baseline" ? base : model;
      var out = new Float32Array(frame.w * frame.h * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
      gl.readPixels(0, 0, frame.w, frame.h, gl.RGBA, gl.FLOAT, out);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return out;
    }

    /* A capped-resolution sample for the scopes and the distribution numbers.
       The exact peak comes from the reduction above, not from here. */
    var SAMPLE_MAX_SIDE = 768;

    function sampleSize() {
      var scale = Math.min(1, SAMPLE_MAX_SIDE / Math.max(frame.w, frame.h));
      return {width: Math.max(1, Math.round(frame.w * scale)),
              height: Math.max(1, Math.round(frame.h * scale))};
    }

    function blitInto(sourceTarget, dst) {
      gl.useProgram(progCopy);
      gl.activeTexture(gl.TEXTURE0 + SCRATCH_UNIT);
      gl.bindTexture(gl.TEXTURE_2D, sourceTarget.tex);
      gl.uniform1i(uniform(progCopy, "uSrc"), SCRATCH_UNIT);
      draw(progCopy, dst.w, dst.h, dst.fb);
      var out = new Float32Array(dst.w * dst.h * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb);
      gl.readPixels(0, 0, dst.w, dst.h, gl.RGBA, gl.FLOAT, out);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return out;
    }

    /* Which source pixel each sampled pixel came from, so a caller can pull
       the mask values it already holds at exactly the same places. */
    function sourceIndex(sw, sh) {
      var idx = new Int32Array(sw * sh);
      for (var y = 0; y < sh; y++) {
        var sy = Math.min(frame.h - 1, Math.floor((y + 0.5) / sh * frame.h));
        for (var x = 0; x < sw; x++) {
          var sx = Math.min(frame.w - 1, Math.floor((x + 0.5) / sw * frame.w));
          idx[y * sw + x] = sy * frame.w + sx;
        }
      }
      return idx;
    }

    function sample() {
      if (!frame) { return null; }
      var sw = small.w, sh = small.h;
      if (!baseSample) { baseSample = blitInto(base, smallBase); }
      return {width: sw, height: sh,
              model: blitInto(model, small),
              baseline: baseSample,
              index: sourceIndex(sw, sh)};
    }

    return {
      gl: gl,
      setFrame: setFrame,
      setParams: function (p) {
        var recompose = false;
        ["strength", "mode", "preserve", "shadowWeight"].forEach(function (k) {
          if (p[k] !== undefined && p[k] !== params[k]) { params[k] = p[k]; recompose = true; }
        });
        if (p.regions !== undefined) {
          if (JSON.stringify(p.regions) !== JSON.stringify(params.regions)) {
            params.regions = p.regions; recompose = true;
          }
        }
        if (p.displayNits !== undefined) { params.displayNits = p.displayNits; }
        if (p.show !== undefined) { params.show = p.show; }
        // The wipe only chooses which of two finished buffers each pixel reads,
        // so it never triggers a recomposite -- dragging it is free.
        if (p.wipe !== undefined) {
          params.wipe = p.wipe < 0 ? -1 : Math.max(0, Math.min(1, p.wipe));
        }
        if (p.wipeHalfWidth !== undefined) { params.wipeHalfWidth = p.wipeHalfWidth; }
        if (recompose) { composite(); }
        return recompose;
      },
      present: present,
      readComposite: readComposite,
      sample: sample,
      peakNits: function () { return reduce(model, 0) * PEAK_NITS; },
      basePeakNits: function () { return reduce(base, 0) * PEAK_NITS; },
      meanNits: function () {
        return reduce(model, 1) * PEAK_NITS / (frame.w * frame.h);
      },
      size: function () { return frame ? {width: frame.w, height: frame.h} : null; },
      wipe: function () { return params.wipe; }
    };
  }

  window.RudraGL = {create: create, LOG_SCALE: LOG_SCALE, MAX_HDR: MAX_HDR,
                    PEAK_NITS: PEAK_NITS};
}());
