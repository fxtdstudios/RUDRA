"""validate_training_data.py -- Quick sanity check for .npz training pairs.

Usage:
    python validate_training_data.py G:\data\checkpoints\wan_hdr_pairs
    python validate_training_data.py G:\data\checkpoints\wan_lora_cache --mode lora
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np


def validate_pairs(pair_dir: str, mode: str = "decoder", sample_count: int = 20):
    pair_path = Path(pair_dir)
    if not pair_path.exists():
        print(f"[ERROR] Directory not found: {pair_dir}")
        return 1

    pattern = "*.npz"
    files = sorted(pair_path.rglob(pattern))
    if not files:
        print(f"[ERROR] No .npz files found in: {pair_dir}")
        return 1

    print("=" * 60)
    print(f"  VALIDATING: {pair_dir}")
    print(f"  Mode:       {mode}")
    print(f"  Total files: {len(files)}")
    print("=" * 60)

    total_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    print(f"\n  Total disk size: {total_size_mb:.1f} MB")

    n_check = min(sample_count, len(files))
    
    latent_channels_set = set()
    nan_count = 0
    inf_count = 0
    shapes = set()
    value_ranges = {"latent_min": [], "latent_max": [],
                     "target_min": [], "target_max": []}
    meta_keys = set()

    print(f"\n  Checking {n_check} samples...")
    print("-" * 60)

    for f in files[:n_check]:
        try:
            data = np.load(str(f), allow_pickle=False)
            keys = list(data.keys())
            
            if mode == "decoder":
                if "latent" not in keys or "log_coded" not in keys:
                    print(f"  [WARN] {f.name}: missing keys. Got: {keys}")
                    continue
                
                latent = data["latent"]
                target = data["log_coded"]
                
                ch = latent.shape[0]
                latent_channels_set.add(ch)
                shapes.add(str(latent.shape))
                
                has_nan = np.isnan(latent).any() or np.isnan(target).any()
                has_inf = np.isinf(latent).any() or np.isinf(target).any()
                
                if has_nan:
                    nan_count += 1
                    print(f"  [NaN]  {f.name}")
                if has_inf:
                    inf_count += 1
                    print(f"  [Inf]  {f.name}")
                
                value_ranges["latent_min"].append(float(latent.min()))
                value_ranges["latent_max"].append(float(latent.max()))
                value_ranges["target_min"].append(float(target.min()))
                value_ranges["target_max"].append(float(target.max()))
                
                if "meta" in keys:
                    meta = json.loads(str(data["meta"]))
                    meta_keys.update(meta.keys())
                
                print(f"  {f.name}: latent={latent.shape} target={target.shape} "
                      f"ch={ch} range=[{latent.min():.2f},{latent.max():.2f}]")
            
            elif mode == "lora":
                if "clean_latent" not in keys:
                    print(f"  [WARN] {f.name}: missing clean_latent. Got: {keys}")
                    continue
                
                latent = data["clean_latent"]
                ch = latent.shape[0]
                latent_channels_set.add(ch)
                shapes.add(str(latent.shape))
                
                has_nan = np.isnan(latent).any()
                has_inf = np.isinf(latent).any()
                
                if has_nan:
                    nan_count += 1
                if has_inf:
                    inf_count += 1
                
                value_ranges["latent_min"].append(float(latent.min()))
                value_ranges["latent_max"].append(float(latent.max()))
                
                if "meta" in keys:
                    meta = json.loads(str(data["meta"]))
                    meta_keys.update(meta.keys())
                    if "model_name" in meta:
                        print(f"  {f.name}: shape={latent.shape} model={meta.get('model_name','?')} "
                              f"ratio={meta.get('compression_ratio','?')}")
                    else:
                        print(f"  {f.name}: shape={latent.shape}")
                else:
                    print(f"  {f.name}: shape={latent.shape}")

        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total pairs:       {len(files)}")
    print(f"  Disk size:        {total_size_mb:.1f} MB")
    print(f"  Channel counts:   {sorted(latent_channels_set)}")
    print(f"  Unique shapes:    {len(shapes)}")
    print(f"  NaN files:        {nan_count}  {'[WARNING]' if nan_count else '[OK]'}")
    print(f"  Inf files:        {inf_count}  {'[WARNING]' if inf_count else '[OK]'}")
    
    if value_ranges["latent_min"]:
        print(f"  Latent range:     [{min(value_ranges['latent_min']):.3f}, "
              f"{max(value_ranges['latent_max']):.3f}]")
    if mode == "decoder" and value_ranges["target_min"]:
        print(f"  Target range:     [{min(value_ranges['target_min']):.3f}, "
              f"{max(value_ranges['target_max']):.3f}]")
    
    if meta_keys:
        print(f"  Meta keys:        {sorted(meta_keys)}")

    print("\n  HEALTH CHECKS:")
    
    if mode == "decoder":
        if len(latent_channels_set) == 1:
            ch = list(latent_channels_set)[0]
            if ch in (4, 16, 128):
                print(f"  [OK] Consistent channel count: {ch}")
            else:
                print(f"  [WARN] Unexpected channel count: {ch}")
        else:
            print(f"  [ERROR] Inconsistent channel counts: {latent_channels_set}")
        
        target_max = max(value_ranges.get("target_max", [0]))
        if target_max > 1.5:
            print(f"  [OK] Target max={target_max:.3f} (log-coded highlights present)")
        else:
            print(f"  [WARN] Target max={target_max:.3f} (may lack HDR highlights)")
        
        latent_std = np.std([float(v) for v in value_ranges.get("latent_min", [0])])
        if latent_std < 5.0:
            print(f"  [OK] Latent values in reasonable range")
        else:
            print(f"  [WARN] Latent values may be out of expected range")
    
    if nan_count > 0:
        print(f"  [ERROR] {nan_count} files contain NaN!")
        return 1
    if inf_count > 0:
        print(f"  [ERROR] {inf_count} files contain Inf!")
        return 1
    
    print("\n  VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate .npz training pairs")
    parser.add_argument("pair_dir", help="Directory with .npz files")
    parser.add_argument("--mode", choices=["decoder", "lora"], default="decoder",
                       help="decoder=latent+log_coded pairs, lora=clean_latent cache")
    parser.add_argument("--samples", type=int, default=20,
                       help="Number of files to check in detail")
    args = parser.parse_args()
    sys.exit(validate_pairs(args.pair_dir, args.mode, args.samples))