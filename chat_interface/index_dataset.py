"""Build a WEVA retrieval index from a folder of files.

Run:
    python index_dataset.py --checkpoint ../models/weva_tiny_train1.pt --data ./data --out ./index.npz
"""
import argparse
import json
from pathlib import Path
from typing import List

from weva_backend import IndexedItem, WEVABackend


def auto_pair(data_dir: Path) -> List[IndexedItem]:
    """Scan sub-folders and pair files by basename (item_id)."""
    images = {p.stem: p for p in (data_dir / "images").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    audio  = {p.stem: p for p in (data_dir / "audio").rglob("*")  if p.suffix.lower() in {".wav", ".mp3"}}
    videos = {p.stem: p for p in (data_dir / "videos").rglob("*") if p.suffix.lower() in {".mp4", ".avi"}}
    texts  = {p.stem: p for p in (data_dir / "texts").rglob("*")  if p.suffix.lower() == ".txt"}
    ids = sorted(set(images) | set(audio) | set(videos) | set(texts))
    items = []
    for i in ids:
        t = None
        if i in texts:
            t = texts[i].read_text(encoding="utf-8", errors="ignore").strip()
        items.append(IndexedItem(
            id=i,
            text=t,
            image_path=str(images[i]) if i in images else None,
            audio_path=str(audio[i])   if i in audio   else None,
            video_path=str(videos[i])  if i in videos  else None,
        ))
    return items


def load_manifest(data_dir: Path) -> List[IndexedItem]:
    manifest = data_dir / "manifest.json"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    items = []
    for d in data:
        items.append(IndexedItem(
            id=d["id"],
            text=d.get("text"),
            image_path=str(data_dir / d["image"]) if d.get("image") else None,
            audio_path=str(data_dir / d["audio"])  if d.get("audio")  else None,
            video_path=str(data_dir / d["video"])  if d.get("video")  else None,
        ))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./index.npz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    data_dir = Path(args.data)
    items = load_manifest(data_dir) or auto_pair(data_dir)
    print(f"[index] found {len(items)} items in {data_dir}")
    if not items:
        print("[index] nothing to index. Drop files into data/images, data/audio, etc.")
        return

    backend = WEVABackend(args.checkpoint, device=args.device)
    mat = backend.encode_database(items, show_progress=True)

    # Save as npz
    import numpy as np
    np.savez(args.out,
             ids=[it.id for it in items],
             embeddings=mat,
             items=[{
                 "id": it.id,
                 "text": it.text,
                 "image_path": it.image_path,
                 "audio_path": it.audio_path,
                 "video_path": it.video_path,
             } for it in items])
    print(f"[index] saved {len(items)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
