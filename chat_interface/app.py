"""Flask chat-style interface backed by WEVA retrieval.

Run:
    python app.py --checkpoint ../models/weva_tiny_train1.pt --index ./index.npz
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from weva_backend import IndexedItem, WEVABackend


app = Flask(__name__)
STATE = {"backend": None, "index": None, "items": [], "matrix": None}


def load_index(path: str):
    data = np.load(path, allow_pickle=True)
    items = [IndexedItem(
        id=meta["id"],
        text=meta.get("text"),
        image_path=meta.get("image_path"),
        audio_path=meta.get("audio_path"),
        video_path=meta.get("video_path"),
    ) for meta in data["items"]]
    return items, data["embeddings"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    if STATE["backend"] is None or STATE["matrix"] is None:
        return jsonify({"error": "no index loaded; run index_dataset.py first"}), 400

    top_k = int(request.form.get("top_k", 5))
    text  = request.form.get("text") or None
    image_bytes = request.files.get("image") or None
    image_bytes = image_bytes.read() if image_bytes else None
    audio_bytes = request.files.get("audio") or None
    audio_bytes = audio_bytes.read() if audio_bytes else None
    video_bytes = request.files.get("video") or None
    video_bytes = video_bytes.read() if video_bytes else None

    if not any([text, image_bytes, audio_bytes, video_bytes]):
        return jsonify({"error": "send at least one of text / image / audio / video"}), 400

    try:
        query = STATE["backend"].encode(
            text=text,
            image=image_bytes,
            audio=audio_bytes,
            video=video_bytes,
        )
    except Exception as e:
        return jsonify({"error": f"encode failed: {e}"}), 500

    matches = WEVABackend.search(query, STATE["matrix"], top_k=top_k)
    enriched = []
    for m in matches:
        item = STATE["items"][m["index"]]
        enriched.append({
            "id": item.id,
            "score": m["score"],
            "text": item.text,
            "image_path": item.image_path,
        })
    summary = WEVABackend.summarise(matches, STATE["items"], top_k=top_k)
    return jsonify({"matches": enriched, "summary": summary})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--index", default="./index.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    STATE["backend"] = WEVABackend(args.checkpoint, device=args.device)
    STATE["items"], STATE["matrix"] = load_index(args.index)
    print(f"[chat] loaded {len(STATE['items'])} items, ready on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
