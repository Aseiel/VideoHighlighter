"""
Face Identity Bank — persistent person identity via face recognition.

A self-contained module for video_ai_editor. It answers one question:
"which known person is this face?" and remembers the answer across frames,
scenes, and even across different videos.

How identity is remembered (the principle):
    An identity is a cluster of points in face-embedding space.
    - Remember  = store the face's embedding vector in that identity's gallery.
    - Recognize = embed a new face, find the nearest stored vector by cosine
                  similarity. Above `sim_threshold` -> same person, else NEW id.

This module does NOT track people between frames — that's the tracker's job
(ByteTrack/BoT-SORT track_id). The bank gives the *anchor* (who this is when a
face is visible); track_id carries that identity through frames where the face
is hidden (back turned, too small, profile).

Backbone: InsightFace `buffalo_l` (SCRFD detector + ArcFace recogniser, 512-d
normalised embeddings). On Intel hardware you can pass the OpenVINO provider.

Typical use (in an analysis pass):
    bank = FaceIdentityBank(db_path="./cache/face_db.json")
    faces = bank.detect_faces(frame_bgr)          # all faces in the frame
    for person_box in person_boxes:               # YOLO/tracker person boxes
        face = bank.best_face_for_box(faces, person_box)
        if face is not None:
            identity_id = bank.assign(face["embedding"],
                                      thumbnail=crop(frame_bgr, face["bbox"]))
            # attach identity_id to this person box in your cache
    bank.save()

Naming (from the GUI, e.g. right-click -> "This is Tomek"):
    bank.name_identity(identity_id, "Tomek")
    bank.save()

NOTE on threading: InsightFace's model is not guaranteed thread-safe. Use one
bank instance per worker thread, or serialise calls. `assign()` itself is
guarded by a lock so the registry stays consistent.

Install:
    pip install insightface onnxruntime          # CPU
    pip install insightface onnxruntime-gpu       # NVIDIA
    pip install insightface onnxruntime-openvino  # Intel (OpenVINO provider)
"""

from __future__ import annotations

import os
import json
import base64
import threading
import uuid
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "buffalo_l"        # SCRFD detector + ArcFace recogniser
DEFAULT_SIM_THRESHOLD = 0.45       # cosine sim; tune on YOUR footage (0.35–0.55)
DEFAULT_MAX_GALLERY = 8            # embeddings kept per identity (different angles)
DEFAULT_DET_SIZE = (640, 640)
EMBED_DIM = 512


# ──────────────────────────────────────────────────────────────────
# FaceIdentityBank
# ──────────────────────────────────────────────────────────────────

class FaceIdentityBank:
    """
    Stores known identities (gallery of face embeddings) and matches new faces
    against them. Persists to a single JSON file so identities survive across
    videos and app restarts.
    """

    def __init__(
        self,
        db_path: str | None = None,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        max_gallery: int = DEFAULT_MAX_GALLERY,
        model_name: str = DEFAULT_MODEL,
        providers: list[str] | None = None,
        det_size: tuple[int, int] = DEFAULT_DET_SIZE,
        ctx_id: int = 0,
    ):
        """
        Args:
            db_path:       JSON file to load/save the identity gallery.
            sim_threshold: cosine similarity above which a face is "the same person".
            max_gallery:   max embeddings stored per identity.
            model_name:    InsightFace model pack ("buffalo_l", "buffalo_s", ...).
            providers:     onnxruntime providers. None = InsightFace default.
                           Intel: ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            det_size:      detector input size.
            ctx_id:        0 = first GPU, -1 = CPU.
        """
        self.db_path = db_path
        self.sim_threshold = float(sim_threshold)
        self.max_gallery = int(max_gallery)
        self.model_name = model_name
        self.providers = providers
        self.det_size = det_size
        self.ctx_id = ctx_id

        # Registry: list of identity dicts
        #   {"id": str, "name": str|None, "embeddings": np.ndarray(N,512),
        #    "thumb": str|None (base64 jpeg), "count": int}
        self.identities: list[dict] = []
        self._id_index: dict[str, dict] = {}

        self._app = None                  # InsightFace, lazy-loaded
        self._lock = threading.Lock()

        if db_path and os.path.exists(db_path):
            self.load(db_path)

    # ── model lifecycle ───────────────────────────────────────────

    def _ensure_app(self):
            """Lazy-load the InsightFace model on first use."""
            if self._app is not None:
                return self._app

            try:
                from insightface.app import FaceAnalysis
            except ImportError as e:
                raise ImportError(
                    "insightface is required for face recognition.\n"
                    "  pip install insightface onnxruntime           (CPU)\n"
                    "  pip install insightface onnxruntime-gpu        (NVIDIA)\n"
                    "  pip install insightface onnxruntime-openvino   (Intel)"
                ) from e

            providers = self.providers
            if not providers:
                # auto-pick: prefer Intel GPU via OpenVINO, then CUDA, then CPU
                try:
                    import onnxruntime as ort
                    avail = set(ort.get_available_providers())
                except Exception:
                    avail = set()
                if "OpenVINOExecutionProvider" in avail:
                    providers = [("OpenVINOExecutionProvider", {"device_type": "GPU"}),
                                "CPUExecutionProvider"]
                elif "CUDAExecutionProvider" in avail:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]

            app = FaceAnalysis(name=self.model_name, providers=providers)
            app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            self._app = app
            print(f"✅ FaceIdentityBank: loaded '{self.model_name}' "
                f"(providers={providers}, ctx_id={self.ctx_id})")
            return self._app

    # ── detection ─────────────────────────────────────────────────

    def detect_faces(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Detect all faces in a frame and return their embeddings.

        Run this ONCE per frame, then match faces to person boxes with
        best_face_for_box(). Do not run it per person — it's a whole-frame pass.

        Returns list of dicts:
            {"bbox": (x1,y1,x2,y2),       # pixel coords
             "embedding": np.ndarray(512), # L2-normalised
             "det_score": float,
             "kps": list|None}            # 5 facial keypoints
        """
        app = self._ensure_app()
        out: list[dict] = []

        for f in app.get(frame_bgr):
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                raw = np.asarray(f.embedding, dtype=np.float32)
                emb = raw / (np.linalg.norm(raw) + 1e-9)
            out.append({
                "bbox": tuple(int(v) for v in f.bbox),
                "embedding": np.asarray(emb, dtype=np.float32),
                "det_score": float(getattr(f, "det_score", 0.0)),
                "kps": (np.asarray(f.kps).tolist()
                        if getattr(f, "kps", None) is not None else None),
            })
        return out

    # ── matching / assignment ─────────────────────────────────────

    def match(self, embedding: np.ndarray) -> tuple[Optional[str], float]:
        """
        Find the best-matching known identity for an embedding.

        Returns (identity_id, similarity) if similarity >= sim_threshold,
        else (None, best_similarity_seen).
        """
        emb = self._norm(embedding)
        best_id: Optional[str] = None
        best_sim = -1.0

        for ident in self.identities:
            gal = ident["embeddings"]
            if gal.size == 0:
                continue
            sim = float(np.max(gal @ emb))   # max over the identity's gallery
            if sim > best_sim:
                best_sim = sim
                best_id = ident["id"]

        if best_id is not None and best_sim >= self.sim_threshold:
            return best_id, best_sim
        return None, best_sim

    def assign(
        self,
        embedding: np.ndarray,
        thumbnail: np.ndarray | None = None,
        det_score: float | None = None,
    ) -> str:
        """
        Match an embedding to a known identity, or create a new one.

        Returns the identity_id (stable across calls and across videos).
        Pass `thumbnail` (a BGR face crop) on first sight so the GUI can show
        a face to name.
        """
        with self._lock:
            emb = self._norm(embedding)
            mid, sim = self.match(emb)

            if mid is None:
                mid = self._new_id()
                ident = {
                    "id": mid,
                    "name": None,
                    "embeddings": emb[None, :].copy(),
                    "thumb": self._encode_thumb(thumbnail),
                    "count": 1,
                }
                self.identities.append(ident)
                self._id_index[mid] = ident
            else:
                ident = self._id_index[mid]
                ident["embeddings"] = self._grow_gallery(ident["embeddings"], emb)
                ident["count"] = ident.get("count", 0) + 1
                # backfill a thumbnail if we never stored one
                if thumbnail is not None and not ident.get("thumb"):
                    ident["thumb"] = self._encode_thumb(thumbnail)

            return mid
        
    def reinforce(self, identity_id, embedding, novelty_max: float = 0.7) -> bool:
        """
        Teach a KNOWN identity a new view of itself.
        Adds the embedding only if it's a genuinely new angle (not already
        well-represented), so the gallery stays diverse instead of filling
        up with near-duplicate frontal shots.
        """
        ident = self._id_index.get(identity_id)
        if ident is None:
            return False
        emb = self._norm(embedding)
        gal = ident["embeddings"]
        if gal.size and float(np.max(gal @ emb)) > novelty_max:
            return False           # already have a similar view — don't bloat
        with self._lock:
            ident["embeddings"] = self._grow_gallery(ident["embeddings"], emb)
            ident["count"] = ident.get("count", 0) + 1
        return True

    # ── face ↔ person-box association ─────────────────────────────

    @staticmethod
    def face_in_box(face_bbox, person_bbox) -> bool:
        """True if the face centre lies inside the person box."""
        fx1, fy1, fx2, fy2 = face_bbox
        cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        px1, py1, px2, py2 = person_bbox
        return (px1 <= cx <= px2) and (py1 <= cy <= py2)

    def best_face_for_box(self, faces: list[dict], person_bbox) -> Optional[dict]:
        """
        Pick the face that belongs to a given person box.

        A face belongs to a person box if its centre is inside the box.
        If several qualify (overlapping people), the highest-confidence face wins.
        Returns the face dict from detect_faces(), or None if no face matches.
        """
        candidates = [f for f in faces if self.face_in_box(f["bbox"], person_bbox)]
        if not candidates:
            return None
        return max(candidates, key=lambda f: f["det_score"])

    def identify_person_box(
        self,
        faces: list[dict],
        person_bbox,
        frame_bgr: np.ndarray | None = None,
    ) -> Optional[str]:
        """
        Convenience: find the face inside a person box and assign an identity.

        Returns identity_id, or None if no usable face was found inside the box
        (caller should then fall back to track_id continuity).
        """
        face = self.best_face_for_box(faces, person_bbox)
        if face is None:
            return None

        thumb = None
        if frame_bgr is not None:
            thumb = self._crop(frame_bgr, face["bbox"])
        return self.assign(face["embedding"], thumbnail=thumb,
                            det_score=face["det_score"])

    # ── naming / lookup ───────────────────────────────────────────

    def name_identity(self, identity_id: str, name: str) -> bool:
        """Attach a human-readable name (e.g. 'Tomek') to an identity."""
        ident = self._id_index.get(identity_id)
        if ident is None:
            return False
        ident["name"] = name
        return True
    
    def set_avoid(self, identity_id, avoid: bool = True) -> bool:
        ident = self._id_index.get(identity_id)
        if ident is None:
            return False
        ident["avoid"] = bool(avoid)
        return True

    def is_avoided(self, identity_id) -> bool:
        ident = self._id_index.get(identity_id)
        return bool(ident and ident.get("avoid", False))

    def avoided_ids(self) -> list[str]:
        return [i["id"] for i in self.identities if i.get("avoid", False)]

    def merge_identities(self, keep_id: str, merge_id: str) -> bool:
        """
        Merge two identities into one (for fixing splits, e.g. when the same
        person was registered twice across very different lighting).
        Galleries are concatenated and capped.
        """
        keep = self._id_index.get(keep_id)
        gone = self._id_index.get(merge_id)
        if keep is None or gone is None or keep_id == merge_id:
            return False
        with self._lock:
            combined = np.vstack([keep["embeddings"], gone["embeddings"]])
            keep["embeddings"] = combined[-self.max_gallery:]
            keep["count"] = keep.get("count", 0) + gone.get("count", 0)
            if not keep.get("name") and gone.get("name"):
                keep["name"] = gone["name"]
            if not keep.get("thumb") and gone.get("thumb"):
                keep["thumb"] = gone["thumb"]
            self.identities = [i for i in self.identities if i["id"] != merge_id]
            del self._id_index[merge_id]
        return True

    def get_identity(self, identity_id: str) -> Optional[dict]:
        """Return the identity record (without the raw embedding array)."""
        ident = self._id_index.get(identity_id)
        if ident is None:
            return None
        return {
            "id": ident["id"],
            "name": ident.get("name"),
            "count": ident.get("count", 0),
            "thumb": ident.get("thumb"),
            "avoid": ident.get("avoid", False),
            "gallery_size": int(ident["embeddings"].shape[0]),
        }

    def all_identities(self) -> list[dict]:
        """List all known identities (metadata only, no embeddings)."""
        return [self.get_identity(i["id"]) for i in self.identities]

    def name_for(self, identity_id: str) -> str:
        """Display label for an identity (name if set, else short id)."""
        ident = self._id_index.get(identity_id)
        if ident is None:
            return "unknown"
        return ident.get("name") or f"Person {identity_id[:8]}"

    # ── persistence ───────────────────────────────────────────────

    def save(self, path: str | None = None) -> bool:
        """Write the gallery to JSON. Embeddings are stored as plain lists."""
        path = path or self.db_path
        if not path:
            print("⚠️ FaceIdentityBank.save: no db_path set")
            return False
        try:
            data = {
                "version": 1,
                "model_name": self.model_name,
                "sim_threshold": self.sim_threshold,
                "identities": [
                    {
                        "id": i["id"],
                        "name": i.get("name"),
                        "count": int(i.get("count", 0)),
                        "thumb": i.get("thumb"),
                        "avoid": bool(i.get("avoid", False)),
                        "embeddings": i["embeddings"].astype(np.float32).tolist(),
                    }
                    for i in self.identities
                ],
            }
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"💾 FaceIdentityBank: saved {len(self.identities)} identities → {path}")
            return True
        except Exception as e:
            print(f"⚠️ FaceIdentityBank.save failed: {e}")
            return False

    def load(self, path: str | None = None) -> bool:
        """Load the gallery from JSON."""
        path = path or self.db_path
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.identities = []
            self._id_index = {}
            for rec in data.get("identities", []):
                emb = np.asarray(rec.get("embeddings", []), dtype=np.float32)
                if emb.ndim == 1 and emb.size == EMBED_DIM:
                    emb = emb[None, :]
                if emb.ndim != 2:
                    emb = np.empty((0, EMBED_DIM), dtype=np.float32)
                ident = {
                    "id": rec["id"],
                    "name": rec.get("name"),
                    "count": int(rec.get("count", 0)),
                    "thumb": rec.get("thumb"),
                    "avoid": bool(rec.get("avoid", False)),
                    "embeddings": emb,
                }
                self.identities.append(ident)
                self._id_index[ident["id"]] = ident

            # Respect a saved threshold unless the caller overrode it explicitly
            if "sim_threshold" in data:
                self.sim_threshold = float(data["sim_threshold"])

            print(f"✅ FaceIdentityBank: loaded {len(self.identities)} identities ← {path}")
            return True
        except Exception as e:
            print(f"⚠️ FaceIdentityBank.load failed: {e}")
            return False

    # ── internals ─────────────────────────────────────────────────

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).ravel()
        return v / (np.linalg.norm(v) + 1e-9)

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex

    def _grow_gallery(self, gallery: np.ndarray, emb: np.ndarray) -> np.ndarray:
        """
        Append a new embedding and cap the gallery size.

        Simple policy: keep the most recent `max_gallery`. (A diversity-keeping
        policy could be swapped in here later without touching callers.)
        """
        stacked = np.vstack([gallery, emb[None, :]])
        if stacked.shape[0] > self.max_gallery:
            stacked = stacked[-self.max_gallery:]
        return stacked.astype(np.float32)

    @staticmethod
    def _crop(frame_bgr: np.ndarray, bbox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2].copy()

    @staticmethod
    def _encode_thumb(thumb_bgr: np.ndarray | None, max_dim: int = 96) -> Optional[str]:
        """Encode a small face crop as a base64 JPEG for the GUI."""
        if thumb_bgr is None or thumb_bgr.size == 0:
            return None
        try:
            import cv2
            h, w = thumb_bgr.shape[:2]
            if max(h, w) > max_dim:
                s = max_dim / max(h, w)
                thumb_bgr = cv2.resize(thumb_bgr, (int(w * s), int(h * s)))
            ok, buf = cv2.imencode(".jpg", thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return None
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception:
            return None

    @staticmethod
    def decode_thumb(thumb_b64: str | None) -> Optional[np.ndarray]:
        """Decode a stored thumbnail back to a BGR image (for the GUI)."""
        if not thumb_b64:
            return None
        try:
            import cv2
            raw = base64.b64decode(thumb_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.identities)

    def __repr__(self) -> str:
        named = sum(1 for i in self.identities if i.get("name"))
        return (f"<FaceIdentityBank identities={len(self.identities)} "
                f"named={named} threshold={self.sim_threshold}>")


# ──────────────────────────────────────────────────────────────────
# Smoke test:  python face_identity.py <image_or_video> [face_db.json]
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python face_identity.py <image_or_video> [face_db.json]")
        sys.exit(1)

    import cv2

    src = sys.argv[1]
    db = sys.argv[2] if len(sys.argv) > 2 else None
    bank = FaceIdentityBank(db_path=db)

    def process(frame, tag=""):
        faces = bank.detect_faces(frame)
        for f in faces:
            ident_id = bank.assign(f["embedding"], thumbnail=bank._crop(frame, f["bbox"]))
            print(f"  {tag} face det={f['det_score']:.2f} -> {bank.name_for(ident_id)}")

    ext = os.path.splitext(src)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        img = cv2.imread(src)
        if img is None:
            print(f"Could not read image: {src}")
            sys.exit(1)
        process(img, "img")
    else:
        cap = cv2.VideoCapture(src)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # sample ~20 frames across the clip
        for i in range(20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((i / 20) * total))
            ok, frame = cap.read()
            if not ok:
                continue
            process(frame, f"f{i}")
        cap.release()

    print(f"\n{bank}")
    for ident in bank.all_identities():
        print(f"  {ident['id'][:8]}  name={ident['name']}  "
              f"seen={ident['count']}  gallery={ident['gallery_size']}")

    if db:
        bank.save()