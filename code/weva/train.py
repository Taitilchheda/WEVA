"""Training loop for WEVA.

Designed to run on a single RTX 3060 (12GB) with the synthetic dataset.
For real multimodal training, swap in your own dataset in
`weva.data` that returns matching (text, image, audio, video) tuples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import WEVAConfig, WEVA_TINY, WEVA_SMALL, WEVA_MEDIUM
from .data import SyntheticMultimodalDataset, multimodal_collate
from .encoder import Modality
from .losses import MultimodalContrastiveLoss, OrbitalEnergyRegularizer
from .model import WEVAModel


CONFIGS = {
    "tiny": WEVA_TINY,
    "small": WEVA_SMALL,
    "medium": WEVA_MEDIUM,
}


def _modality_subset(name: str) -> list:
    """Parse a comma-separated list of modalities to use."""
    if name == "all":
        return [m.value for m in Modality]
    return [s.strip() for s in name.split(",") if s.strip()]


def train(
    config_name: str = "tiny",
    n_samples: int = 1024,
    batch_size: int = 16,
    n_epochs: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    use_oer: bool = False,
    oer_lambda: float = 0.001,
    oer_tau: float = 0.1,
    modalities: str = "all",
    save_path: str = "models/weva_tiny.pt",
    log_every: int = 10,
    device: Optional[str] = None,
) -> Dict[str, list]:
    """Train a WEVA model on the synthetic multimodal dataset.

    Returns a dict of training curves.
    """
    cfg = CONFIGS[config_name]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {config_name}, n_layers={cfg.n_layers}, d_model={cfg.d_model}, "
          f"n_heads={cfg.n_heads}")

    model = WEVAModel(cfg).to(device)
    print(f"Total parameters: {model.num_parameters():,}")

    train_set = SyntheticMultimodalDataset(n_samples=n_samples, config=cfg)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=multimodal_collate, num_workers=0,
    )

    modalities_to_run = _modality_subset(modalities)
    print(f"Modalities in use: {modalities_to_run}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9,
        weight_decay=weight_decay,
    )
    n_steps = max(1, n_epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    contrastive = MultimodalContrastiveLoss(temperature=0.07).to(device)
    oer = OrbitalEnergyRegularizer(tau=oer_tau).to(device) if use_oer else None

    history: Dict[str, list] = {"loss": [], "lr": [], "oer": []}

    model.train()
    step = 0
    start = time.time()
    for epoch in range(n_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if k in modalities_to_run}
            optimizer.zero_grad()
            embeddings = model(batch, modalities_to_run=modalities_to_run)
            loss = contrastive(embeddings)
            oer_val = torch.tensor(0.0, device=device)
            if oer is not None:
                # First do a backward pass so gradients exist
                loss.backward(retain_graph=True)
                oer_val = oer_lambda * oer(model)
                (loss + oer_val).backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            history["loss"].append(float(loss.item()))
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))
            history["oer"].append(float(oer_val.item()) if oer is not None else 0.0)
            if step % log_every == 0:
                print(
                    f"  epoch {epoch} step {step:5d} | loss {loss.item():.4f} | "
                    f"oer {history['oer'][-1]:.4f} | lr {history['lr'][-1]:.2e} | "
                    f"elapsed {time.time()-start:.1f}s"
                )
            step += 1

    # Save the trained model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "config_name": config_name,
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "history": history,
    }, save_path)
    print(f"Saved model to {save_path}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tiny", choices=list(CONFIGS))
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--use-oer", action="store_true")
    parser.add_argument("--oer-lambda", type=float, default=0.001)
    parser.add_argument("--oer-tau", type=float, default=0.1)
    parser.add_argument("--modalities", default="all",
                        help="Comma-separated list: text,image,audio,video or 'all'")
    parser.add_argument("--save-path", default="models/weva_tiny.pt")
    args = parser.parse_args()

    history = train(
        config_name=args.config,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_oer=args.use_oer,
        oer_lambda=args.oer_lambda,
        oer_tau=args.oer_tau,
        modalities=args.modalities,
        save_path=args.save_path,
    )

    # Save training curves
    history_path = args.save_path.replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
