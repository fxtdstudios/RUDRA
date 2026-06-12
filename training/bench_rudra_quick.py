"""Quick RUDRA benchmark — 3 key model configs, 10 iters each."""
import time, torch, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rudra'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import FORMAT_DIM
from descriptor import RUDRADescriptor, format_onehot
from encoder import RUDRAProjection
from decoder import RUDRADecoder

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

desc = RUDRADescriptor().to(device)
proj = RUDRAProjection(dr_raw_dim=22, format_dim=FORMAT_DIM, proj_dim=64).to(device)

configs = [
    ('flux (16ch)',  16, 64, 8),
    ('sdxl (4ch)',    4, 64, 8),
    ('ltx (128ch)', 128,  4, 2),
]

print(f"\n{'Model':<15} {'Batch':>5} {'Step ms':>8} {'steps/s':>7} {'VRAM':>6} {'50K turbo':>10} {'200K full':>10}")
print("-" * 75)

for name, ch, hw, bs in configs:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    turbo = RUDRADecoder(latent_channels=ch, dr_dim=64).to(device)
    optimizer = torch.optim.AdamW(list(turbo.parameters()) + list(proj.parameters()), lr=3e-4)
    target = torch.randn(bs, 512, 512, 3, device=device)
    latent = torch.randn(bs, ch, hw, hw, device=device)

    # Warmup 3
    for _ in range(3):
        dr_raw = desc(torch.randn(bs, 3, 512, 512, device=device).abs())
        fmt_oh = format_onehot(torch.randint(0, FORMAT_DIM, (bs,), device=device), device=device)
        dr_proj = proj(dr_raw, fmt_oh)
        pred = turbo(latent, dr_proj)
        if pred.shape[2] != 512 or pred.shape[3] != 512:
            pred = torch.nn.functional.interpolate(pred, size=(512, 512), mode='bilinear', align_corners=False)
        pred_bhwc = pred.permute(0, 2, 3, 1)
        loss = torch.nn.functional.l1_loss(pred_bhwc, target)
        loss.backward()

    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        dr_raw = desc(torch.randn(bs, 3, 512, 512, device=device).abs())
        fmt_oh = format_onehot(torch.randint(0, FORMAT_DIM, (bs,), device=device), device=device)
        dr_proj = proj(dr_raw, fmt_oh)
        pred = turbo(latent, dr_proj)
        if pred.shape[2] != 512 or pred.shape[3] != 512:
            pred = torch.nn.functional.interpolate(pred, size=(512, 512), mode='bilinear', align_corners=False)
        pred_bhwc = pred.permute(0, 2, 3, 1)
        loss = torch.nn.functional.l1_loss(pred_bhwc, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    step_time = sum(times) / len(times)
    sps = 1.0 / step_time
    vram = torch.cuda.max_memory_allocated() / 1024**3
    est_50k = 50000 * step_time / 3600
    est_200k = 200000 * step_time * 3 / 3600

    print(f"{name:<15} {bs:>5} {step_time*1000:>7.1f}ms {sps:>6.1f}/s {vram:>5.1f}GB {est_50k:>9.1f}h {est_200k:>9.1f}h")

    del turbo, optimizer, latent, target
    torch.cuda.empty_cache()

print("\n50K = turbo decoder, 200K = full decoder (~3x slower per step)")