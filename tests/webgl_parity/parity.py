"""Check the WebGL composite against a numpy port of SDR2HDRNet.forward's tail.

The shader is the only place the reconstruction is assembled for the viewer, so
if it drifts from the torch maths the page shows something the master EXR does
not contain. Runs headless on SwiftShader; no GPU and no torch required.
"""
import json, pathlib, sys
import numpy as np
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
W, H = 1001, 37                     # odd, and wide enough to force the sample downscale

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from compose_reference import (baseline_of, compose,          # noqa: E402
                               linear_to_srgb, LOG_SCALE, MAX_HDR)


def main():
    rng = np.random.default_rng(20260828)
    sdr_u8 = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    # a strip of pure white and a strip of pure black: the two clamp edges
    sdr_u8[0, :, :] = 255
    sdr_u8[1, :, :] = 0
    residual = (rng.normal(0.0, 1.2, size=(H, W, 3))).astype(np.float16)
    highlight = rng.random((H, W, 1)).astype(np.float16)
    shadow = rng.random((H, W, 1)).astype(np.float16)

    fields = np.concatenate([residual, highlight], axis=-1).astype(np.float16)
    payload = {
        "width": W, "height": H,
        "sdr": sdr_u8.ravel().tolist(),
        "fields": fields.ravel().view(np.uint16).tolist(),
        "shadow": shadow.ravel().view(np.uint16).tolist(),
    }
    (HERE / "payload.json").write_text(json.dumps(payload))

    cases = [
        (1.0, "all", True, 203.0),
        (1.0, "all", False, 203.0),
        (0.0, "all", True, 203.0),
        (2.0, "highlights", True, 1000.0),
        (1.35, "shadows", False, 4000.0),
        (1.0, "off", True, 203.0),
    ]

    import functools, http.server, socketserver, threading
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(REPO))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    httpd.allow_reuse_address = True
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader",
                                          "--enable-unsafe-swiftshader",
                                          "--use-angle=swiftshader"])
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(f"http://127.0.0.1:{port}/tests/webgl_parity/harness.html")
        try:
            page.wait_for_function("window.READY === true || window.FAIL", timeout=30000)
        except Exception:
            print("timeout; errors:", errors); return 1
        if page.evaluate("window.FAIL || null"):
            print("harness failed:", page.evaluate("window.FAIL")); return 1
        if errors:
            print("page errors:", errors); return 1

        worst = 0.0
        for strength, mode, preserve, nits in cases:
            got = page.evaluate(
                "a => window.runCase(a)",
                {"strength": strength, "mode": mode, "preserve": preserve,
                 "displayNits": nits})
            gl_rgb = np.array(got["composite"], dtype=np.float64).reshape(H, W, 4)[..., :3]
            gl_disp = np.array(got["display"], dtype=np.float64).reshape(H, W, 4)[..., :3]

            ref = compose(sdr_u8.astype(np.float64) / 255.0,
                            residual.astype(np.float64), highlight.astype(np.float64),
                            shadow.astype(np.float64), strength, mode, preserve)
            err = np.abs(gl_rgb - ref).max()
            rel = np.abs(gl_rgb - ref).max() / max(float(np.abs(ref).max()), 1e-9)
            worst = max(worst, rel)

            ref_disp = np.round(linear_to_srgb(
                np.clip(ref * (10000.0 / nits), 0.0, 1.0)) * 255.0)
            disp_err = np.abs(gl_disp * 255.0 - ref_disp).max()

            mx = float((ref.max(-1)).max()) * 10000.0
            mean = float((ref.max(-1)).mean()) * 10000.0
            print(f"  strength={strength:<5} mode={mode:<11} preserve={str(preserve):<5} "
                  f"peak={nits:>6.0f}  abs {err:.3e}  rel {rel:.3e}  disp8 {disp_err:.1f} "
                  f"| peak {got['peak']:.4f} vs {mx:.4f}  mean {got['mean']:.4f} vs {mean:.4f}")
            # relative to the frame peak: half-float fields carry ~1e-3 relative
            # precision, and the composite must not amplify that.
            assert rel < 2e-5, f"composite drift {rel} relative"
            assert disp_err <= 1.0, f"display drift {disp_err} codes"
            assert abs(got["peak"] - mx) / max(mx, 1e-6) < 1e-4, "peak reduction wrong"
            assert abs(got["mean"] - mean) / max(mean, 1e-6) < 1e-4, "mean reduction wrong"
        # the sampled readback the scopes and distribution numbers run on
        page.evaluate("() => window.runCase({strength:1.0, mode:'all', "
                      "preserve:true, displayNits:203})")
        s = page.evaluate("() => window.sampleCase()")
        sw, sh = s["width"], s["height"]
        idx = np.array(s["index"], dtype=np.int64)
        gm = np.array(s["model"], dtype=np.float64).reshape(sh, sw, 4)[..., :3]
        gb = np.array(s["baseline"], dtype=np.float64).reshape(sh, sw, 4)[..., :3]
        ref_model = compose(sdr_u8.astype(np.float64) / 255.0,
                              residual.astype(np.float64), highlight.astype(np.float64),
                              shadow.astype(np.float64), 1.0, "all", True)
        ref_base = baseline_of(sdr_u8.astype(np.float64) / 255.0)
        flat_m = ref_model.reshape(-1, 3)[idx].reshape(sh, sw, 3)
        flat_b = ref_base.reshape(-1, 3)[idx].reshape(sh, sw, 3)
        em = np.abs(gm - flat_m).max()
        eb = np.abs(gb - flat_b).max()
        print(f"\n  sample {sw}x{sh} from {W}x{H}: model {em:.3e}  baseline {eb:.3e} "
              f"(index map agrees)")
        scale_m = max(float(np.abs(flat_m).max()), 1e-9)
        assert em / scale_m < 2e-5 and eb / scale_m < 2e-5, \
            "sampled readback does not match its index map"

        browser.close()
    print(f"\nOK  worst relative composite error {worst:.2e} over {len(cases)} cases")
    return 0

if __name__ == "__main__":
    sys.exit(main())
