"""Does the wipe actually show two different images, on the sides it claims?

The wipe is a DISPLAY-only feature: it picks, per pixel, which of two finished
float buffers to read. Nothing upstream changes, so the composite parity test
cannot see it at all -- the same blind spot that let the viewer present every
frame upside down for two days. So this test looks at the canvas.

Three things have to hold, and each has failed in some viewer somewhere:

  1. Left of the seam is the BASELINE and right of it is the RECONSTRUCTION,
     not the other way round. "Before and after" is read left to right.
  2. The two sides genuinely differ. A wipe that shows the same image twice
     looks exactly like a working wipe on most frames.
  3. Moving the seam moves the boundary, and turning the wipe off restores the
     full reconstruction -- otherwise the control is decoration.

The frame is DARK rather than mid grey, which matters. The composite gates the
residual by a per-pixel luminance prior, so on mid grey neither arm fires and
the reconstruction is within one code value of the baseline -- a wipe over that
frame looks broken whether it works or not. At code 20 the shadow arm fires and
the two sides are ~50 against ~200, which is decidable from one row of pixels.

    python tests/webgl_parity/wipe.py
"""
import pathlib

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
W, H = 32, 4

JS = r"""
() => {
  const W = %d, H = %d;
  const gl = window.RudraGL.create(document.getElementById("c"));
  if (!gl) { return {error: "no webgl2 / no EXT_color_buffer_float"}; }

  // Dark everywhere, so the shadow arm of the luminance gate is open, plus a
  // positive residual. Mid grey would gate the residual to nothing.
  const sdr = new Uint8Array(W * H * 3).fill(20);
  const fields = new Float32Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    fields[i * 4] = fields[i * 4 + 1] = fields[i * 4 + 2] = 1.2;  // log residual
    fields[i * 4 + 3] = 1.0;                                      // highlight mask
  }
  // The compositor takes float16 bits; Float16Array is not universal, so go
  // through a Float32Array view the same way app.js does for real frames.
  const f16 = new Uint16Array(W * H * 4);
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  const toHalf = (v) => {
    f32[0] = v;
    const x = u32[0], s = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112;
    const m = x & 0x7fffff;
    if (e <= 0) { return s; }
    if (e >= 31) { return s | 0x7c00; }
    return s | (e << 10) | (m >>> 13);
  };
  for (let i = 0; i < W * H * 4; i++) { f16[i] = toHalf(fields[i]); }
  const shadow = new Uint16Array(W * H).fill(toHalf(1.0));

  gl.setFrame({width: W, height: H, sdr: sdr, fields: f16, shadow: shadow});

  const scratch = document.createElement("canvas");
  scratch.width = W; scratch.height = H;
  const c2 = scratch.getContext("2d");
  const rowAt = () => {
    c2.clearRect(0, 0, W, H);
    c2.drawImage(document.getElementById("c"), 0, 0);
    const px = c2.getImageData(0, 0, W, H).data;
    const out = [];
    for (let x = 0; x < W; x++) { out.push(px[x * 4]); }   // top row, red channel
    return out;
  };

  const shots = {};
  gl.setParams({wipe: -1});      gl.present(); shots.off      = rowAt();
  gl.setParams({wipe: 0.5});     gl.present(); shots.half     = rowAt();
  gl.setParams({wipe: 0.25});    gl.present(); shots.quarter  = rowAt();
  gl.setParams({wipe: -1});      gl.present(); shots.offAgain = rowAt();
  return {shots: shots, reported: gl.wipe()};
}
""" % (W, H)


def sides(row, split, half_px=1):
    """Mean of each side, ignoring pixels near the seam (the handle is drawn there)."""
    cut = int(round(split * W))
    left = [v for i, v in enumerate(row) if i < cut - half_px]
    right = [v for i, v in enumerate(row) if i > cut + half_px]
    return sum(left) / len(left), sum(right) / len(right)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader",
                                          "--enable-unsafe-swiftshader",
                                          "--use-angle=swiftshader"])
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((HERE / "wipe.html").as_uri())
        out = page.evaluate(JS)
        browser.close()

    if errors:
        print("page errors:", errors)
        return 1
    if "error" in out:
        print("harness failed:", out["error"])
        return 1

    s = out["shots"]
    full = sum(s["off"]) / len(s["off"])
    print("  wipe off        mean %.1f" % full)

    for name, split in (("half", 0.5), ("quarter", 0.25)):
        left, right = sides(s[name], split)
        print("  wipe at %-5s   left %6.1f   right %6.1f" % (split, left, right))
        assert right > left + 20, (
            "wipe at %s: the right side (%.1f) should be the brighter "
            "reconstruction and the left (%.1f) the baseline" % (split, right, left))
        assert abs(right - full) < 12, (
            "wipe at %s: the right side (%.1f) is not the same image the wipe-off "
            "view shows (%.1f)" % (split, right, full))

    # Moving the seam has to move the boundary, not just redraw the same split.
    a, b = s["half"], s["quarter"]
    moved = sum(1 for i in range(W) if abs(a[i] - b[i]) > 20)
    print("  moving 0.5 -> 0.25 changed %d of %d columns" % (moved, W))
    assert moved >= W // 5, "moving the wipe did not move the boundary"

    assert s["offAgain"] == s["off"], "turning the wipe off did not restore the view"
    assert out["reported"] == -1, "wipe() should report -1 when off, got %r" % out["reported"]
    print("  wipe ok: baseline left, reconstruction right, seam follows the value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
