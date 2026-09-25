"""Evaluation: retrieval metrics (R@1, R@5, R@10) and modality-ablation
helpers.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .config import WEVA_TINY
from .data import SyntheticMultimodalDataset, multimodal_collate
from .encoder import Modality
from .model import WEVAModel


@torch.no_grad()
def encode_dataset(
    model: WEVAModel, dataset, batch_size: int = 16, device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Run the model on every item in `dataset`, returning per-modality
    embeddings stacked as (N, proj_dim) tensors.
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=multimodal_collate, num_workers=0,
    )
    out: Dict[str, List[torch.Tensor]] = {m.value: [] for m in Modality}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        embeddings = model(batch)
        for k, v in embeddings.items():
            out[k].append(v.cpu())
    return {k: torch.cat(v, dim=0) for k, v in out.items()}


def retrieval_metrics(
    a: torch.Tensor, b: torch.Tensor, ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute R@k for image->text and text->image.

    Args:
        a: (N, d) query embeddings.
        b: (N, d) candidate embeddings (same order as a — index i of a
           corresponds to index i of b).
        ks: tuple of k values for R@k.
    Returns:
        Dict with keys "a_to_b_Rk" and "b_to_a_Rk" for each k.
    """
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    sims = a @ b.T  # (N, N)
    N = sims.size(0)
    targets = torch.arange(N)
    metrics = {}
    for k in ks:
        # a -> b: top-k candidates for each row
        _, topk = sims.topk(k, dim=1)
        hits = (topk == targets.unsqueeze(1)).any(dim=1).float()
        metrics[f"a_to_b_R{k}"] = hits.mean().item()
        # b -> a
        _, topk = sims.T.topk(k, dim=1)
        hits = (topk == targets.unsqueeze(1)).any(dim=1).float()
        metrics[f"b_to_a_R{k}"] = hits.mean().item()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/weva_tiny.pt")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False: our checkpoints embed a full Python dict (config + history).
    # Only load checkpoints produced by this codebase; do not load untrusted files.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_name = ckpt["config_name"]
    cfg_dict = ckpt["config"]
    from .config import WEVAConfig
    cfg = WEVAConfig(**cfg_dict)
    model = WEVAModel(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    dataset = SyntheticMultimodalDataset(n_samples=args.n_samples, config=cfg)
    embeddings = encode_dataset(model, dataset, args.batch_size, device)

    print(f"Evaluated on {args.n_samples} synthetic samples.")
    print("Embedding shapes:")
    for m, e in embeddings.items():
        print(f"  {m}: {tuple(e.shape)}")

    # Pairwise retrieval metrics
    pairs = [
        ("text", "image"),
        ("image", "audio"),
        ("text", "audio"),
        ("text", "video"),
        ("image", "video"),
        ("audio", "video"),
    ]
    results = {}
    for a, b in pairs:
        m = retrieval_metrics(embeddings[a], embeddings[b])
        results[f"{a}-{b}"] = m
        print(
            f"  {a} <-> {b}: "
            + ", ".join(f"{k}={v:.3f}" for k, v in m.items())
        )

    out_path = args.checkpoint.replace(".pt", "_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
