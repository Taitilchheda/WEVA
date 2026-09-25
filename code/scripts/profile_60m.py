"""Profile WEVA on the actual RTX 3060 to answer the
'have we really trained 60M locally' question with real numbers.

Reports peak VRAM and step time at multiple batch sizes for all three
model sizes. Output: a clean table you can paste into the paper.
"""
import gc
import time
import torch
import sys
sys.path.insert(0, "E:/dj sanghvi/findings/nicola inspired transformer/code")

from weva.config import WEVA_TINY, WEVA_SMALL, WEVA_MEDIUM
from weva.model import WEVAModel
from weva.losses import MultimodalContrastiveLoss

torch.backends.cuda.matmul.allow_tf32 = True

SIZES = [
    ("WEVA-Tiny   (16M)", WEVA_TINY),
    ("WEVA-Small  (33M)", WEVA_SMALL),
    ("WEVA-Medium (58M)", WEVA_MEDIUM),
]
BATCH_SIZES = [16, 32, 48, 64, 96]


def make_batch(B, cfg):
    """Build a batch of the right shape for this config's max sizes."""
    text_len  = cfg.text_max_len
    img       = cfg.image_size
    vid_fr    = cfg.video_max_frames
    audio_len = cfg.audio_max_frames * cfg.audio_hop   # rough inverse
    return {
        "text":  torch.randint(0, cfg.text_vocab_size, (B, text_len),    device="cuda"),
        "image": torch.randn   (B, 3, img, img,                          device="cuda"),
        "audio": torch.randn   (B, audio_len,                            device="cuda"),
        "video": torch.randn   (B, vid_fr, 3, img, img,                  device="cuda"),
    }


print("=" * 78)
print("WEVA profiling on RTX 3060 12GB — peak VRAM and step time")
print("=" * 78)

results = []
for name, cfg in SIZES:
    print(f"\n--- {name}  d={cfg.d_model}  L={cfg.n_layers}  H={cfg.n_heads} ---")
    torch.cuda.empty_cache()
    model = WEVAModel(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best = None
    for B in BATCH_SIZES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = make_batch(B, cfg)
            # Warmup
            emb = model(batch)
            loss = MultimodalContrastiveLoss()(emb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            torch.cuda.synchronize()
            # Timed
            t0 = time.time()
            for _ in range(3):
                emb = model(batch)
                loss = MultimodalContrastiveLoss()(emb)
                loss.backward()
                opt.step()
                opt.zero_grad()
            torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000 / 3
            gb = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"  batch={B:3d}  step={ms:6.0f}ms  peak={gb:5.2f}GB   OK")
            best = (B, ms, gb)
            results.append((name, B, ms, gb, "OK"))
            # Free activations
            del emb, loss, batch
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                gb = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(f"  batch={B:3d}  OOM at {gb:.2f}GB")
                results.append((name, B, None, gb, "OOM"))
                torch.cuda.empty_cache()
            else:
                print(f"  batch={B:3d}  ERROR: {e}")
                results.append((name, B, None, None, f"ERR: {e}"))
                torch.cuda.empty_cache()
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

print()
print("=" * 78)
print("SUMMARY (RTX 3060 12GB, BF16, ADAM-W + multimodal contrastive loss)")
print("=" * 78)
print(f"{'Model':<22} {'Max batch':>10} {'Step time':>12} {'Peak VRAM':>12}")
print("-" * 78)
for name, B, ms, gb, status in results:
    if status == "OK":
        line = f"{name:<22} {B:>10} {ms:>9.0f}ms {gb:>9.2f}GB"
    else:
        line = f"{name:<22} {'OOM':>10} {'-':>12} {gb:>9.2f}GB"
    print(line)
print()
print("VERDICT: see paper Section 7 (Implementation & Feasibility).")
