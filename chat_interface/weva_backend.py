"""WEVA backend wrapper for the chat interface.

Loads a trained WEVA checkpoint, exposes:
  - encode_item(text, image, audio, video) -> 256-d L2-normalised embedding
  - encode_database(items) -> (N, 256) matrix
  - search(query_embedding, db_matrix, top_k) -> ranked list of (idx, score)
  - summarise(matches) -> natural-language summary (template-based)
"""
from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# Make the parent code/ folder importable so we can use the WEVA package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from weva.config import WEVAConfig
from weva.model import WEVAModel  # noqa: E402
from weva.encoder import Modality  # noqa: E402


@dataclass
class IndexedItem:
    """One item in the database."""
    id: str
    text: Optional[str] = None
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    embedding: Optional[np.ndarray] = None  # (256,) float32


class WEVABackend:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """Load WEVA from a checkpoint. CPU is the safe default for the
        chat server (RTX 3060 only has 12 GB and the user might be training
        on it at the same time).
        """
        self.device = device
        # weights_only=False: our checkpoints embed a Python dict.
        # Only load checkpoints produced by this codebase.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = WEVAConfig(**ckpt["config"])
        self.cfg = cfg
        self.model = WEVAModel(cfg).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.proj_dim = cfg.proj_dim
        print(f"[weva] loaded {checkpoint_path}  d={cfg.d_model}  L={cfg.n_layers}  H={cfg.n_heads}")

    # ------------------------------------------------------------------ encode
    def _load_image(self, source) -> Optional[torch.Tensor]:
        if source is None:
            return None
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source)).convert("RGB")
        elif isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
        else:
            img = source
        img = img.resize((self.cfg.image_size, self.cfg.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)                              # (3, H, W)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def _load_audio(self, source) -> Optional[torch.Tensor]:
        # Lazy import — torchaudio is not in our requirements.txt by default
        try:
            import torchaudio  # type: ignore
        except ImportError:
            return None
        if isinstance(source, (str, Path)):
            wav, sr = torchaudio.load(str(source))
        else:
            return None
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(0)  # mono
        target_len = self.cfg.audio_max_frames * self.cfg.audio_hop
        if wav.size(0) > target_len:
            wav = wav[:target_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.size(0)))
        return wav.unsqueeze(0).to(self.device)

    def _load_video(self, source) -> Optional[torch.Tensor]:
        # Lazy: use torchvision.io.read_video if available
        if isinstance(source, (str, Path)):
            try:
                import torchvision.io  # type: ignore
                video, _, _ = torchvision.io.read_video(str(source), pts_unit="sec")
            except Exception:
                return None
        else:
            return None
        n = self.cfg.video_max_frames
        if video.size(0) >= n:
            idx = torch.linspace(0, video.size(0) - 1, n).long()
            video = video[idx]
        else:
            pad = n - video.size(0)
            video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)
        # (T, 3, H, W) -> we resize each frame
        out = []
        for frame in video:
            img = Image.fromarray(frame.numpy()).resize(
                (self.cfg.image_size, self.cfg.image_size)
            )
            arr = np.asarray(img, dtype=np.float32) / 255.0
            out.append(arr.transpose(2, 0, 1))
        arr = np.stack(out, axis=0)                               # (T, 3, H, W)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode(self,
               text: Optional[str] = None,
               image=None,
               audio=None,
               video=None) -> np.ndarray:
        """Encode one (text, image, audio, video) tuple to a 256-d vector.
        At least one modality must be provided.
        """
        batch = {}
        if text is not None:
            # Naive tokenisation: random "fake" token ids from a deterministic
            # hash. Good enough for a demo; real BPE would need a tokenizer.
            ids = [hash(w) % self.cfg.text_vocab_size for w in text.split()[: self.cfg.text_max_len]]
            ids = ids + [0] * (self.cfg.text_max_len - len(ids))
            batch["text"] = torch.tensor([ids], dtype=torch.long, device=self.device)
        if image is not None:
            batch["image"] = self._load_image(image)
        if audio is not None:
            batch["audio"] = self._load_audio(audio)
        if video is not None:
            batch["video"] = self._load_video(video)
        if not batch:
            raise ValueError("At least one modality (text/image/audio/video) must be provided.")
        with torch.no_grad():
            emb = self.model(batch)
        # If multiple modalities are given, average their L2-normalised embeddings
        # and re-normalise. This is a simple late-fusion for the demo.
        vecs = [v.squeeze(0).cpu().numpy() for v in emb.values()]
        avg = np.mean(np.stack(vecs, axis=0), axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        return avg.astype(np.float32)

    @torch.no_grad()
    def encode_database(self, items: List[IndexedItem], show_progress: bool = True) -> np.ndarray:
        """Encode a list of items. Returns (N, proj_dim) L2-normalised matrix."""
        mat = np.zeros((len(items), self.proj_dim), dtype=np.float32)
        for i, item in enumerate(items):
            text = item.text
            image = item.image_path
            audio = item.audio_path
            video = item.video_path
            try:
                v = self.encode(text=text, image=image, audio=audio, video=video)
                mat[i] = v
                item.embedding = v
            except Exception as e:
                print(f"[weva] failed to encode item {item.id}: {e}")
                mat[i] = 0.0
            if show_progress and (i + 1) % 10 == 0:
                print(f"[weva] encoded {i + 1}/{len(items)}")
        return mat

    # ------------------------------------------------------------------ search
    @staticmethod
    def search(query: np.ndarray, db: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Cosine-similarity search. Returns top_k matches as a list of dicts."""
        if db.shape[0] == 0:
            return []
        sims = db @ query                                          # (N,)
        idx = np.argsort(-sims)[:top_k]
        return [
            {"index": int(i), "score": float(sims[i])}
            for i in idx
        ]

    # ------------------------------------------------------------------ summarise
    @staticmethod
    def summarise(matches: List[Dict], items: List[IndexedItem], top_k: int) -> str:
        """Render a natural-language summary of the matches. This is NOT
        generative — it is a deterministic template that picks a
        human-readable phrasing based on the top score."""
        if not matches:
            return "I could not find any matching items in the database."
        top = matches[0]
        item = items[top["index"]]
        score_pct = int(round(max(0.0, min(1.0, top["score"])) * 100))
        text_hint = item.text[:80] + ("..." if item.text and len(item.text) > 80 else "")
        bits = [
            f"Your query matched {len(matches)} item{'s' if len(matches) != 1 else ''} in the database.",
            f"The strongest match is item {item.id} (similarity: {score_pct}%).",
        ]
        if text_hint:
            bits.append(f"That item is described as: \"{text_hint}\".")
        if len(matches) > 1:
            second_pct = int(round(max(0.0, min(1.0, matches[1]["score"])) * 100))
            bits.append(
                f"Other close matches: {', '.join(items[m['index']].id for m in matches[1:min(3, len(matches))])} "
                f"at {second_pct}% or below."
            )
        return " ".join(bits)
