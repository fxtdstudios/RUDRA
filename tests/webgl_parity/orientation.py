"""Is the frame presented right way up?

Every texture in the compositor is uploaded in array order -- row 0 is the top
of the image -- and every readback path (readComposite, sample/sourceIndex, the
Master EXR) assumes that order. The default framebuffer does not: its row 0 is
the BOTTOM of the canvas, so presenting with vUV unchanged shows the frame
upside down. That is exactly the bug shipped on 29 Aug 2026, and none of the
numeric tests caught it, because all of them compare the float composite -- the
one buffer a presentation flip leaves untouched.

So this test looks at the canvas and nothing else: a frame whose TOP half is
bright must display bright at the top. Zero fields keep it deterministic, so
there is nothing here to flake.

    python tests/webgl_parity/orientation.py
"""
import pathlib
import sys

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
W, H = 8, 4
BRIGHT, DARK = 230, 26

JS = r"""
() => {
  const W = %d, H = %d;
  const gl = window.RudraGL.create(document.getElementById("c"));
  if (!gl) { return {error: "no webgl2 / no EXT_color_buffer_float"}; }
  const sdr = new Uint8Array(W * H * 3);
  for (let y = 0; y < H; y++) {
    const v = y < H / 2 ? %d : %d;        // array order: the TOP half is the bright one
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 3;
      sdr[i] = sdr[i + 1] = sdr[i + 2] = v;
    }
  }
  gl.setFrame({width: W, height: H, sdr: sdr,
               fields: new Uint16Array(W * H * 4), shadow: new Uint16Array(W * H)});
  gl.present();
  // Read through a 2D context rather than gl.readPixels: getImageData row 0 is
  // the top of the canvas as a person sees it, which is the whole question.
  const scratch = document.createElement("canvas");
  scratch.width = W; scratch.height = H;
  const ctx = scratch.getContext("2d");
  ctx.drawImage(document.getElementById("c"), 0, 0);
  const px = ctx.getImageData(0, 0, W, H).data;
  const rows = [];
  for (let y = 0; y < H; y++) {
    let s = 0;
    for (let x = 0; x < W; x++) { s += px[(y * W + x) * 4]; }
    rows.push(s / W);
  }
  return {rows: rows};
}
""" % (W, H, BRIGHT, DARK)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader",
                                          "--enable-unsafe-swiftshader",
                                          "--use-angle=swiftshader"])
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((HERE / "orientation.html").as_uri())
        out = page.evaluate(JS)
        browser.close()

    if errors:
        print("page errors:", errors)
        return 1
    if "error" in out:
        print("harness failed:", out["error"])
        return 1

    rows = out["rows"]
    top = sum(rows[:H // 2]) / (H // 2)
    bottom = sum(rows[H // 2:]) / (H - H // 2)
    print(f"  row means {['%.1f' % r for r in rows]}")
    print(f"  top {top:.1f}   bottom {bottom:.1f}")
    assert top > bottom + 40, (
        f"the canvas is presenting the frame upside down: the bright top half "
        f"reads {top:.1f} and the dark bottom half {bottom:.1f}")
    print("  orientation ok: the bright half displays at the top")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
