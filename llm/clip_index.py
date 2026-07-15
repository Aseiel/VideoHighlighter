"""Encode-once frame index for CLIP visual search.

`clip_prefilter` answers one query in one forward pass: it feeds frames and
prompts through CLIP together and softmaxes the result. That couples the two
halves, so every new query re-encodes every frame — the expensive half — and a
reworded search costs a full rescan.

The image tower doesn't depend on the query. So this module splits them: encode
each sampled frame once into a unit-norm embedding, keep the embeddings, and
answer any later query with a dot product. Re-querying an indexed video is then
arithmetic on a small array (microseconds) instead of a video decode plus a
model pass per frame.

The embeddings are also the substrate for matching by *example* rather than by
wording: an embedding is just a point, so user-picked frames can be averaged
into a query vector. Note `query_vector`'s warning about the modality gap
before doing that.

Storage is cheap: 512 dims x fp16 = 1 KB per frame, so a 2-hour video sampled
once per second is ~7 MB.

Standalone usage:

    python -m llm.clip_index --video "D:\\clips\\a.mp4" --interval 1.0 \
        --query "your wording" --topk 20

Requires the same stack as clip_prefilter: pip install "optimum[openvino]" pillow opencv-python
"""
from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import numpy as np

from llm.clip_prefilter import ClipFramePrefilter

# Bump when the .npz layout changes in a way old caches can't satisfy.
# v2: v1 memos were written by a scan whose grid was offset by start_time, so
# they hold off-lattice timestamps that nothing will ever look up again — dead
# weight that also doubled the file on every search. Discard them.
INDEX_FORMAT_VERSION = 2

# The OpenVINO graph is compiled with both towers wired together, so a forward
# pass needs text and images even when only one side's embedding is wanted. The
# unused side gets a throwaway input; the cost is one extra pass through the
# *small* tower per batch, which is noise next to the vision tower.
_DUMMY_TEXT = "a photo"
_DUMMY_IMAGE_SIZE = 224


def l2_normalize(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)


def score_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                     logit_scale: float = 100.0) -> np.ndarray:
    """P(label 0) for each embedding -> [n] in [0,1].

    `labels` is [m, dim] unit-norm with row 0 the positive and the rest
    contrastive negatives — CLIP's own zero-shot layout, which is why this
    reproduces clip_prefilter's softmax rather than approximating it.
    """
    logits = logit_scale * (np.asarray(embeddings, dtype=np.float32) @ labels.T)
    logits -= logits.max(axis=1, keepdims=True)   # stable softmax
    exp = np.exp(logits)
    return exp[:, 0] / exp.sum(axis=1)


def cache_path_for(video_path: str, cache_dir: str = "./cache") -> str:
    """Where this video's index lives. Mirrors VideoAnalysisCache's key
    (abspath + size + mtime) so a re-encode gets a different file."""
    import hashlib

    st = os.stat(video_path)
    key = f"{os.path.abspath(video_path)}_{st.st_size}_{st.st_mtime}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return os.path.join(cache_dir, "clip", f"{digest}.npz")


class ClipEmbedder(ClipFramePrefilter):
    """Adds raw embedding access to the prefilter's loader.

    Subclasses rather than edits `ClipFramePrefilter` so its hard-won frozen-exe
    IR-loading path (and the file itself) stays untouched.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CLIP's trained temperature. Derived from the first real forward pass
        # rather than hardcoded, so swapping MODEL_ID for a differently-tuned
        # CLIP stays correct. (openai/clip-vit-base-patch32 gives 100.0.)
        self._logit_scale: Optional[float] = None

    def _forward(self, texts: Sequence[str], images: Sequence):
        inputs = self._processor(
            text=list(texts), images=list(images), return_tensors="pt", padding=True,
        )
        return self._model(**inputs)

    def _capture_logit_scale(self, out, img_unit: np.ndarray, txt_unit: np.ndarray):
        """logits_per_image = scale * (img . txt) for unit-norm embeddings, so
        the scale falls out of any pass we've already run. Pick the pair with the
        largest |cosine| so we never divide by a near-zero."""
        if self._logit_scale is not None:
            return
        cos = img_unit @ txt_unit.T
        i, j = np.unravel_index(np.argmax(np.abs(cos)), cos.shape)
        denom = float(cos[i, j])
        if abs(denom) < 1e-6:
            return  # degenerate batch; try again on the next one
        logits = out.logits_per_image.detach().numpy()
        self._logit_scale = float(logits[i, j] / denom)

    @property
    def logit_scale(self) -> float:
        # 100.0 is CLIP's published value; only used if no pass has run yet.
        return self._logit_scale if self._logit_scale is not None else 100.0

    def embed_query_labels(self) -> np.ndarray:
        """Embed the labels set by `set_query` -> [1 + n_negatives, dim].

        Row 0 is the positive prompt. Feed straight to `score_embeddings`.
        """
        if self._labels is None:
            raise RuntimeError("Call set_query() first.")
        return self.embed_texts(self._labels)

    def embed_frames_bgr(self, frames_bgr: Sequence) -> np.ndarray:
        """Unit-norm embeddings for BGR (OpenCV) frames -> [n, dim] float32."""
        import cv2
        from PIL import Image

        if self._model is None:
            raise RuntimeError("Call load() first.")
        images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        out = self._forward([_DUMMY_TEXT], images)
        img = l2_normalize(out.image_embeds.detach().numpy().astype(np.float32))
        self._capture_logit_scale(
            out, img, l2_normalize(out.text_embeds.detach().numpy().astype(np.float32))
        )
        return img

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-norm embeddings for prompts -> [n, dim] float32."""
        from PIL import Image

        if self._model is None:
            raise RuntimeError("Call load() first.")
        dummy = Image.fromarray(
            np.zeros((_DUMMY_IMAGE_SIZE, _DUMMY_IMAGE_SIZE, 3), dtype=np.uint8)
        )
        out = self._forward(texts, [dummy])
        txt = l2_normalize(out.text_embeds.detach().numpy().astype(np.float32))
        self._capture_logit_scale(
            out, l2_normalize(out.image_embeds.detach().numpy().astype(np.float32)), txt
        )
        return txt


class ClipFrameIndex:
    """Timestamps plus their CLIP embeddings, and the queries you can run
    against them. Cheap to keep on disk, instant to re-query.

    Think of it as a memo of "the embedding for frame at time t", not as a
    snapshot of one complete pass. It can be grown a batch at a time
    (`extend`) and asked what it already knows (`lookup`), so a scan that
    stops early still banks the frames it did encode, and the next scan pays
    only for the gaps.
    """

    # Timestamps are floats derived from the same arithmetic each run, but
    # compare them at ms resolution rather than exactly.
    TS_DECIMALS = 3

    def __init__(self, timestamps: np.ndarray, embeddings: np.ndarray,
                 meta: Optional[dict] = None):
        if len(timestamps) != len(embeddings):
            raise ValueError(
                f"{len(timestamps)} timestamps vs {len(embeddings)} embeddings"
            )
        self.timestamps = np.asarray(timestamps, dtype=np.float32)
        # Stored fp16 (1 KB/frame), computed in fp32.
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.meta = dict(meta or {})
        self._rebuild_lookup()

    def __len__(self) -> int:
        return len(self.timestamps)

    # -- memo -------------------------------------------------------------

    def _key(self, ts: float) -> float:
        return round(float(ts), self.TS_DECIMALS)

    def _rebuild_lookup(self) -> None:
        self._by_ts = {self._key(t): i for i, t in enumerate(self.timestamps)}

    def lookup(self, timestamps: Sequence[float]) -> tuple[list[int], list[int], list[float]]:
        """Split `timestamps` into what's already known and what isn't.

        Returns (rows, positions, missing): `rows` indexes into `self.embeddings`
        and `positions` into `timestamps`, aligned; `missing` is the timestamps
        with no embedding yet.
        """
        rows, positions, missing = [], [], []
        for pos, ts in enumerate(timestamps):
            row = self._by_ts.get(self._key(ts))
            if row is None:
                missing.append(ts)
            else:
                rows.append(row)
                positions.append(pos)
        return rows, positions, missing

    def extend(self, timestamps: Sequence[float], embeddings: np.ndarray) -> int:
        """Add embeddings for timestamps not already held. Returns how many
        were actually added (duplicates are ignored, so replaying a scan is
        free rather than corrupting the memo)."""
        emb = np.asarray(embeddings, dtype=np.float32)
        if len(timestamps) != len(emb):
            raise ValueError(f"{len(timestamps)} timestamps vs {len(emb)} embeddings")
        new_ts, new_emb = [], []
        seen = set()
        for ts, e in zip(timestamps, emb):
            k = self._key(ts)
            if k in self._by_ts or k in seen:
                continue
            seen.add(k)
            new_ts.append(float(ts))
            new_emb.append(e)
        if not new_ts:
            return 0
        stacked = np.asarray(new_emb, dtype=np.float32)
        self.timestamps = np.concatenate([self.timestamps, np.asarray(new_ts, dtype=np.float32)])
        self.embeddings = (np.vstack([self.embeddings, stacked])
                           if len(self.embeddings) else stacked)
        self._rebuild_lookup()
        return len(new_ts)

    # -- querying ---------------------------------------------------------

    def query_vector(self, vector: np.ndarray, negatives: Optional[np.ndarray] = None,
                     logit_scale: float = 100.0) -> np.ndarray:
        """Score every frame against one unit-norm vector -> [n] in [0,1].

        With `negatives`, scores are a softmax over [vector, *negatives], which
        reproduces clip_prefilter's numbers to ~1e-6. Without them, the raw
        cosine is rescaled from [-1,1] to [0,1] — monotonic, so ranking is
        unaffected, but the absolute number isn't comparable across queries.

        The negatives must live in the **same modality as `vector`**. CLIP's
        image and text embeddings occupy separate regions of the space (the
        "modality gap"): measured on this model, image<->image cosines run
        ~0.87-0.97 while image<->text run ~0.21-0.31. So a text query needs text
        negatives, and an image-derived query (e.g. averaged example frames)
        needs image negatives — pit an image vector against text negatives and it
        wins by default, handing back 1.0 for every frame. Mixing modalities is
        silent: the numbers look like probabilities, they're just meaningless.
        """
        v = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        if negatives is None or len(negatives) == 0:
            return ((self.embeddings @ v) + 1.0) / 2.0
        labels = np.vstack([v[None, :], l2_normalize(np.asarray(negatives, dtype=np.float32))])
        return score_embeddings(self.embeddings, labels, logit_scale)

    def top_k(self, scores: np.ndarray, k: int = 20) -> list[tuple[float, float]]:
        """The k best (timestamp, score) pairs, best first."""
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx])]
        return [(float(self.timestamps[i]), float(scores[i])) for i in idx]

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            timestamps=self.timestamps,
            embeddings=self.embeddings.astype(np.float16),
            meta=np.array(repr(self.meta), dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "ClipFrameIndex":
        with np.load(path, allow_pickle=True) as z:
            meta = {}
            if "meta" in z:
                try:
                    meta = eval(str(z["meta"].item()), {"__builtins__": {}})  # dict repr we wrote
                except Exception:
                    meta = {}
            return cls(z["timestamps"], z["embeddings"].astype(np.float32), meta)

    def matches(self, video_path: str, model_id: str,
                interval: Optional[float] = None) -> bool:
        """True if this index describes the same video under the same model.
        Size+mtime rather than a hash: cheap, and enough to catch a re-encode.

        `interval` is only checked when given. A memo is keyed per timestamp, so
        it stays valid across sampling intervals — a finer interval just means
        more misses, not a stale cache. Pass it only for a whole-video index
        where "complete" means "every frame on that grid".
        """
        want = _video_fingerprint(video_path, model_id, interval)
        return all(self.meta.get(k) == v for k, v in want.items())


def _video_fingerprint(video_path: str, model_id: str,
                       interval: Optional[float] = None) -> dict:
    st = os.stat(video_path)
    fp = {
        "format": INDEX_FORMAT_VERSION,
        "video": os.path.basename(video_path),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "model": model_id,
    }
    if interval is not None:
        fp["interval"] = float(interval)
    return fp


def open_memo(video_path: str, model_id: str, cache_path: Optional[str] = None,
              cache_dir: str = "./cache") -> tuple[ClipFrameIndex, str]:
    """The embedding memo for a video: whatever was cached, or an empty one.

    Never raises on a bad cache — a corrupt or foreign file just means starting
    over, which costs time, not correctness.
    """
    path = cache_path or cache_path_for(video_path, cache_dir)
    if os.path.exists(path):
        try:
            memo = ClipFrameIndex.load(path)
            if memo.matches(video_path, model_id):
                print(f"🧠 CLIP memo: {len(memo)} frames already embedded")
                return memo, path
            print("🧠 CLIP memo: cache is for different video/model, starting fresh")
        except Exception as e:
            print(f"⚠️  CLIP memo: unreadable cache ({e}); starting fresh")
    meta = _video_fingerprint(video_path, model_id)
    return ClipFrameIndex(np.zeros(0, dtype=np.float32),
                          np.zeros((0, 512), dtype=np.float32), meta), path


def build_index(video_path: str, interval: float = 1.0, batch: int = 16,
                device: str = "GPU", embedder: Optional[ClipEmbedder] = None,
                progress=None) -> ClipFrameIndex:
    """Sample `video_path` every `interval` seconds and embed each frame once.

    `progress(done, total)` is called as batches complete, for a GUI bar.
    """
    import cv2

    pf = embedder
    if pf is None:
        pf = ClipEmbedder(device=device)
        pf.load()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * interval)))
    total_samples = (total_frames // step) if total_frames else 0

    timestamps: list[float] = []
    chunks: list[np.ndarray] = []
    buf_frames: list = []
    buf_ts: list[float] = []
    t0 = time.perf_counter()

    def flush():
        if buf_frames:
            chunks.append(pf.embed_frames_bgr(buf_frames))
            timestamps.extend(buf_ts)
            buf_frames.clear()
            buf_ts.clear()
            if progress:
                progress(len(timestamps), total_samples)

    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                buf_frames.append(frame)
                buf_ts.append(i / fps)
                if len(buf_frames) >= batch:
                    flush()
            i += 1
        flush()
    finally:
        cap.release()

    embeddings = (np.vstack(chunks) if chunks
                  else np.zeros((0, 512), dtype=np.float32))
    meta = _video_fingerprint(video_path, pf.model_id, interval)
    meta["logit_scale"] = pf.logit_scale
    elapsed = time.perf_counter() - t0
    per = (elapsed / len(timestamps) * 1000) if timestamps else 0.0
    print(f"🧠 CLIP index: {len(timestamps)} frames in {elapsed:.1f}s ({per:.1f} ms/frame)")
    return ClipFrameIndex(np.array(timestamps, dtype=np.float32), embeddings, meta)


def load_or_build(video_path: str, cache_path: str, interval: float = 1.0,
                  device: str = "GPU", **kw) -> ClipFrameIndex:
    """Reuse a cached index when it still fits the video, else build and cache."""
    model_id = kw.pop("model_id", None) or ClipFramePrefilter().model_id
    if os.path.exists(cache_path):
        try:
            idx = ClipFrameIndex.load(cache_path)
            if idx.matches(video_path, model_id, interval):
                print(f"🧠 CLIP index: reusing {len(idx)} cached frames")
                return idx
            print("🧠 CLIP index: cache stale (video/settings changed), rebuilding")
        except Exception as e:
            print(f"⚠️  CLIP index: unreadable cache ({e}); rebuilding")
    idx = build_index(video_path, interval=interval, device=device, **kw)
    try:
        idx.save(cache_path)
    except Exception as e:
        print(f"⚠️  CLIP index: could not cache ({e})")
    return idx


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build a CLIP frame index and query it.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", action="append", default=[],
                    help="repeatable; each is scored against the same index")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--cache", default=None, help="path to .npz index cache")
    args = ap.parse_args()

    cache = args.cache or os.path.splitext(args.video)[0] + ".clipindex.npz"
    idx = load_or_build(args.video, cache, interval=args.interval, device=args.device)

    pf = ClipEmbedder(device=args.device)
    pf.load()
    scale = idx.meta.get("logit_scale", 100.0)
    negatives = pf.embed_texts(list(ClipFramePrefilter.DEFAULT_NEGATIVES))

    for q in args.query:
        t0 = time.perf_counter()
        vec = pf.embed_texts([f"a photo of {q}"])[0]
        scores = idx.query_vector(vec, negatives=negatives, logit_scale=scale)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n🔎 {q!r} — scored {len(idx)} frames in {ms:.1f} ms")
        for ts, sc in idx.top_k(scores, args.topk):
            print(f"   {int(ts)//60:>3d}:{int(ts)%60:02d}  ({ts:7.1f}s)   score={sc:.3f}")


if __name__ == "__main__":
    main()
