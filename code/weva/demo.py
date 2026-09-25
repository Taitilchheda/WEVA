"""Inference demo: encode an image and a text, return their similarity.

Usage:
    python -m weva.demo --image path/to/image.jpg --text "a cat sitting on a couch"
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from .config import WEVAConfig, WEVA_TINY
from .data import _synthetic_audio
from .encoder import Modality
from .model import WEVAModel


def _load_image(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return tf(img)


def _tokenize_text(text: str, vocab_size: int, max_len: int) -> torch.LongTensor:
    """Simple whitespace tokenizer with hash-based ids. Replace with a real
    tokenizer (e.g., GPT-2 BPE) for production use.
    """
    tokens = [hash(w) % vocab_size for w in text.split()[:max_len]]
    tokens = tokens + [0] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def similarity(
    model: WEVAModel,
    config: WEVAConfig,
    image: Optional[torch.Tensor] = None,
    text: Optional[str] = None,
    audio: Optional[torch.Tensor] = None,
    video: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> dict:
    """Compute pairwise cosine similarity between any subset of modalities."""
    batch = {}
    if image is not None:
        batch[Modality.IMAGE.value] = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
    if text is not None:
        batch[Modality.TEXT.value] = _tokenize_text(
            text, config.text_vocab_size, config.text_max_len,
        ).to(device)
    if audio is not None:
        batch[Modality.AUDIO.value] = audio.unsqueeze(0).to(device) if audio.dim() == 1 else audio.to(device)
    if video is not None:
        batch[Modality.VIDEO.value] = video.unsqueeze(0).to(device) if video.dim() == 4 else video.to(device)
    model.eval()
    embeddings = model(batch)
    out = {}
    keys = list(embeddings.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            sim = torch.nn.functional.cosine_similarity(
                embeddings[k1], embeddings[k2], dim=-1,
            )
            out[f"{k1}-{k2}"] = float(sim.item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/weva_tiny.pt")
    parser.add_argument("--image", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False: our checkpoints embed a full Python dict (config + history).
    # Only load checkpoints produced by this codebase; do not load untrusted files.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    from .config import WEVAConfig
    cfg = WEVAConfig(**ckpt["config"])
    model = WEVAModel(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    image = _load_image(args.image, cfg.image_size) if args.image else None
    if not any([args.image, args.text, args.audio, args.video]):
        # Default: synthetic image and text
        from .data import _synthetic_image
        image = _synthetic_image(0, cfg.image_size)
        text = "a synthetic pattern with a sinusoidal component"
    else:
        text = args.text

    sims = similarity(model, cfg, image=image, text=text, device=device)
    print("Cross-modal similarities:")
    for k, v in sims.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
