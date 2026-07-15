"""
Efficient cosine-similarity index over face embeddings.

Recognition must compare each detected face against *all* enrolled employee
embeddings, but must not re-scan them with a Python loop every frame. This
module provides a small index that:

* uses **FAISS** (``IndexFlatIP``) when it is installed — inner product on
  L2-normalised vectors is exactly cosine similarity, evaluated in optimised
  native code;
* otherwise falls back to a single vectorised NumPy matmul (``matrix @ vec``),
  which is still one BLAS call over the whole matrix — never a per-row loop.

Both paths return the same ``(similarities, indices)`` for the top-k nearest
stored vectors, so :class:`FaceRecognitionService` is agnostic to which is used.
The active backend is reported via :attr:`backend` for the UI / status.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # optional acceleration
    import faiss  # type: ignore

    _HAVE_FAISS = True
except Exception:  # pragma: no cover - faiss not installed
    faiss = None
    _HAVE_FAISS = False


class FaceIndex:
    def __init__(self, use_faiss: bool = True) -> None:
        self._use_faiss = bool(use_faiss and _HAVE_FAISS)
        self._index = None
        self._matrix: Optional[np.ndarray] = None  # always kept for fallback/centroids
        self._dim = 0

    @property
    def backend(self) -> str:
        # Report the engine that WILL serve queries, even before any vector is
        # added (an empty index has no FAISS object yet but still uses FAISS).
        return "faiss" if self._use_faiss else "numpy"

    @property
    def faiss_available(self) -> bool:
        return _HAVE_FAISS

    @property
    def size(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    @property
    def matrix(self) -> Optional[np.ndarray]:
        return self._matrix

    def build(self, matrix: Optional[np.ndarray]) -> None:
        """(Re)build the index from an ``[N, dim]`` array of unit vectors."""
        if matrix is None or len(matrix) == 0:
            self._index = None
            self._matrix = None
            self._dim = 0
            return
        matrix = np.ascontiguousarray(matrix.astype(np.float32))
        self._matrix = matrix
        self._dim = int(matrix.shape[1])
        if self._use_faiss:
            index = faiss.IndexFlatIP(self._dim)
            index.add(matrix)
            self._index = index
        else:
            self._index = None

    def search(self, vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(sims, idxs)`` for the ``k`` most similar stored vectors.

        ``vec`` must be a unit-norm 1-D float32 array of the index dimension.
        Results are sorted by descending similarity.
        """
        if self._matrix is None:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        k = int(min(max(1, k), self._matrix.shape[0]))
        q = np.ascontiguousarray(vec.astype(np.float32)).reshape(1, -1)
        if self._use_faiss and self._index is not None:
            sims, idxs = self._index.search(q, k)
            return sims[0], idxs[0]
        # vectorised fallback: one matmul over the whole matrix
        sims_all = self._matrix @ q[0]
        idxs = np.argsort(sims_all)[::-1][:k]
        return sims_all[idxs].astype(np.float32), idxs.astype(np.int64)

    def all_sims(self, vec: np.ndarray) -> np.ndarray:
        """Similarity of ``vec`` against every stored vector (aligned to rows)."""
        if self._matrix is None:
            return np.empty(0, dtype=np.float32)
        return (self._matrix @ vec.astype(np.float32)).astype(np.float32)
