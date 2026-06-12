"""RUDRA training benchmark — measures per-step time and estimates production runs."""
import time, torch, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rudra'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import RUDRA_MODEL_CONFIGS, FORMAT_DIM
from descriptor import RUDRADescriptor, format_onehot
from encoder import RUDRAProjection
from decoder import RUDRADecoder, RUDRAFullDecoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print()

desc = RUDRADescriptor().to(device)
proj = RUDRAProjection(dr_raw_dim=22, format_dim=FORMAT_DIM, proj_dim=64).to(device)

# Warmup
x = torch.randn(1, 3, 512, 512).abs().to(device)
dr = desc(x)
fmt = format_onehot(torch.tensor([4]), device=device)
dp = proj(dr, fmt)

test_configs = [
    # name,          ch,  h,  w, batch
    ('flux',           16, 64, 64,   8),
    ('wan',            16, 64, 64,   4),
    ('hunyuanvideo',   16, 64, 64,   4),
    ('sd3',            16, 64, 64,   8),
    ('sdxl',            4, 64, 64,  16),
    ('sd15',            4, 64, 64,  16),
    ('ltx-video',     128,  4,  4,   2),
    ('cogvideox',      16, 64, 64,   4),
    ('lumina2',        16, 64, 64,   8),
    ('pixart',          4, 64, 64,  16),
]

print(f"{'Model':<15} {'Ch':>3} {'Latent':>8} {'Batch':>5} {'Step ms':>8} {'steps/s':>7} {'VRAM GB':>8} {'50K Turbo':>10} {'200K Full':>10}")
print("-" * 90)

for name, ch, h, w, bs in test_configs:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    turbo = RUDRADecoder(latent_channels=ch, dr_dim=64).to(device)
    optimizer = torch.optim.AdamW(list(turbo.parameters()) + list(proj.parameters()), lr=3e-4)
    target = torch.randn(bs, 512, 512, 3).to(device)
    latent = torch.randn(bs, ch, h, w).to(device)

    # Warmup
    for _ in range(5):
        dr_raw = desc(torch.randn(bs, 3, 512, 512).abs().to(device))
        fmt_ids = torch.randint(0, FORMAT_DIM, (bs,), device=device)
        fmt_oh = format_onehot(fmt_ids, device=device)
        dr_proj = proj(dr_raw, fmt_oh)
        pred = turbo(latent, dr_proj)
        pred_bhwc = pred.permute(0, 2, 3, 1)
        if pred_bhwc.shape[1:3] != target.shape[1:3]:
            pred_bhwc = torch.nn.functional.interpolate(
                pred.permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
        loss = torch.nn.functional.l1_loss(pred_bhwc, target)
        loss.backward()

    torch.cuda.synchronize()

    # Timed run (30 iterations)
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        dr_raw = desc(torch.randn(bs, 3, 512, 512).abs().to(device))
        fmt_ids = torch.randint(0, FORMAT_DIM, (bs,), device=device)
        fmt_oh = format_onehot(fmt_ids, device=device)
        dr_proj = proj(dr_raw, fmt_oh)
        pred = turbo(latent, dr_proj)
        pred_bhwc = pred.permute(0, 2, 3, 1)
        if pred_bhwc.shape[1:3] != target.shape[1:3]:
            pred_bhwc = torch.nn.functional.interpolate(
                pred.permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
        loss = torch.nn.functional.l1_loss(pred_bhwc, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    step_time = sum(times) / len(times)
    steps_per_sec = 1.0 / step_time
    vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    # Estimates (Full decoder is ~3x slower, 200K steps)
    est_50k_min = 50000 * step_time / 60
    est_200k_min = 200000 * step_time * 3 / 60  # Full decoder ~3x slower

    print(f"{name:<15} {ch:>3} {f'{h}x{w}':>8} {bs:>5} {step_time*1000:>7.1f}ms {steps_per_sec:>6.1f}/s {vram_gb:>7.1f}GB {est_50k_min/60:>8.1f}h {est_200k_min/60:>8.1f}h")

    del turbo, optimizer, latent, target, dr_raw, fmt_ids, fmt_oh, dr_proj
    torch.cuda.empty_cache()

print()
print("Estimates: 50K steps (turbo) and 200K steps (full, ~3x slower per step)")
print("Times include multi_curve augmentation overhead")