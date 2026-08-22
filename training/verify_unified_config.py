"""Quick verification script for the unified MODEL_VAE_CONFIG integration.

Run from inside ComfyUI (e.g. via an Execute Script node) or ensure
ComfyUI's folder_paths / comfy modules are on sys.path.
"""
import sys
import os

# Add paths so direct imports work without going through radiance.__init__
_RADIANCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CUST_NODES = os.path.abspath(os.path.join(_RADIANCE_ROOT, ".."))
sys.path.insert(0, _RADIANCE_ROOT)
sys.path.insert(0, _CUST_NODES)


def _import_model_map():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "model_map", os.path.join(_RADIANCE_ROOT, "config", "model_map.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_imports():
    print("=" * 60)
    print("TEST 1: Config imports")
    print("=" * 60)
    mm = _import_model_map()
    MODEL_VAE_CONFIG = mm.MODEL_VAE_CONFIG
    print(f"  MODEL_VAE_CONFIG keys: {list(MODEL_VAE_CONFIG.keys())}")
    assert "flux" in MODEL_VAE_CONFIG
    assert "sdxl" in MODEL_VAE_CONFIG
    assert "wan" in MODEL_VAE_CONFIG
    print("  [PASS] Config imports OK")


def test_config_resolution():
    print("\n" + "=" * 60)
    print("TEST 2: Config resolution")
    print("=" * 60)
    mm = _import_model_map()
    resolve_model_vae_config = mm.resolve_model_vae_config

    tests = [
        ("flux", "scale_factor", 0.3611),
        ("flux", "shift_factor", 0.1159),
        ("flux1", "scale_factor", 0.3611),
        ("wan2.1", "latent_channels", 16),
        ("hunyuan_video", "latent_channels", 16),
        ("sd3.5", "scale_factor", 1.5305),
        ("sd1.5", "latent_channels", 4),
        ("sd15", "noise_schedule", "ddpm"),
        ("ltx", "latent_channels", 128),
        ("lumina2", "latent_channels", 16),
    ]
    for hint, param, expected in tests:
        cfg = resolve_model_vae_config(hint)
        assert cfg is not None, f"Failed to resolve '{hint}'"
        val = cfg[param]
        assert val == expected, f"{hint}.{param} = {val}, expected {expected}"
        print(f"  {hint:20s} -> {param:20s} = {val}")
    print("  [PASS] Config resolution OK")


def _comfyui_available() -> bool:
    return "comfy" in sys.modules or "folder_paths" in sys.modules


def test_model_vae():
    print("\n" + "=" * 60)
    print("TEST 3: model/vae.py uses config")
    print("=" * 60)
    if not _comfyui_available():
        print("  [SKIP] ComfyUI not available in this environment")
        return
    mm = _import_model_map()
    resolve_model_vae_config = mm.resolve_model_vae_config

    import importlib.util
    spec = importlib.util.spec_from_file_location("vae", os.path.join(_RADIANCE_ROOT, "model", "vae.py"))
    vae_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vae_mod)
    load_radiance_decoder_weights = vae_mod.load_radiance_decoder_weights

    for mt in ("flux", "sdxl", "wan", "sd3", "sd15", "ltx-video"):
        cfg = resolve_model_vae_config(mt)
        expected_ch = cfg["latent_channels"]
        model = load_radiance_decoder_weights(model_type=mt, model_size="turbo")
        actual_ch = model.latent_channels
        assert actual_ch == expected_ch, f"{mt}: expected {expected_ch}ch, got {actual_ch}ch"
        print(f"  {mt:12s} -> {actual_ch}ch decoder loaded OK")
    print("  [PASS] model/vae.py OK")


def test_fast_vae():
    print("\n" + "=" * 60)
    print("TEST 4: fast_vae.py uses config")
    print("=" * 60)
    if not _comfyui_available():
        print("  [SKIP] ComfyUI not available in this environment")
        return
    mm = _import_model_map()
    resolve_model_vae_config = mm.resolve_model_vae_config

    import importlib.util
    spec = importlib.util.spec_from_file_location("fast_vae", os.path.join(_RADIANCE_ROOT, "fast_vae.py"))
    fv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fv)

    for mt in ("flux", "sdxl", "wan", "sd3", "sd15", "ltx-video"):
        cfg = resolve_model_vae_config(mt)
        expected_ch = cfg["latent_channels"]
        model = fv.load_radiance_decoder_weights(model_type=mt, model_size="turbo")
        actual_ch = model.latent_channels
        assert actual_ch == expected_ch, f"{mt}: expected {expected_ch}ch, got {actual_ch}ch"
        print(f"  {mt:12s} -> {actual_ch}ch decoder loaded OK")
    print("  [PASS] fast_vae.py OK")


def test_target_modules():
    print("\n" + "=" * 60)
    print("TEST 5: LoRA target modules expanded")
    print("=" * 60)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_hdr_lora", os.path.join(_RADIANCE_ROOT, "training", "train_hdr_lora.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _DEFAULT_TARGET_MODULES = mod._DEFAULT_TARGET_MODULES

    required = {"query", "key", "value", "out_proj", "dense", "fc1", "fc2"}
    missing = required - _DEFAULT_TARGET_MODULES
    assert not missing, f"Missing target modules: {missing}"
    print(f"  Total target modules: {len(_DEFAULT_TARGET_MODULES)}")
    print(f"  Newly added: {sorted(required)}")
    print("  [PASS] Target modules expanded OK")


def test_train_turbo_decoder():
    print("\n" + "=" * 60)
    print("TEST 6: train_turbo_decoder.py imports")
    print("=" * 60)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_turbo_decoder", os.path.join(_RADIANCE_ROOT, "training", "train_turbo_decoder.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import inspect
    sig = inspect.signature(mod.train)
    params = list(sig.parameters.keys())
    assert "model_type" in params
    assert "model_size" in params
    print(f"  train() params: {params}")
    print("  [PASS] train_turbo_decoder.py imports OK")


def main():
    try:
        test_config_imports()
        test_config_resolution()
        test_model_vae()
        test_fast_vae()
        test_target_modules()
        test_train_turbo_decoder()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
