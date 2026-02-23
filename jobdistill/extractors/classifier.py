"""Skill-likeness binary classifier: embedding + logistic regression."""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SkillClassifier:
    """Lightweight skill vs. not-skill classifier.

    Uses sentence-transformer embeddings + sklearn LogisticRegression.
    Supports train / predict_proba / save / load.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        threshold: float = 0.60,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.threshold = threshold
        self._embedder: Optional[object] = None
        self._clf: Optional[object] = None
        self._meta: Dict = {}

    def _ensure_embedder(self) -> None:
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        logger.info("Loading embedding model %s", self.embedding_model_name)
        self._embedder = SentenceTransformer(self.embedding_model_name)

    def _embed(self, phrases: List[str]) -> np.ndarray:
        self._ensure_embedder()
        return self._embedder.encode(phrases, show_progress_bar=False, batch_size=256)  # type: ignore[union-attr]

    @property
    def is_trained(self) -> bool:
        return self._clf is not None

    def train(
        self,
        phrases: List[str],
        labels: List[int],
        eval_phrases: Optional[List[str]] = None,
        eval_labels: Optional[List[int]] = None,
    ) -> Dict:
        """Train logistic regression on phrase embeddings.

        Returns dict of training metrics.
        """
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
        from sklearn.metrics import (  # type: ignore[import-untyped]
            accuracy_score,
            classification_report,
            f1_score,
        )

        logger.info("Embedding %d training phrases...", len(phrases))
        X_train = self._embed(phrases)
        y_train = np.array(labels)

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        self._clf = clf

        train_preds = clf.predict(X_train)
        metrics: Dict = {
            "train_accuracy": float(accuracy_score(y_train, train_preds)),
            "train_f1": float(f1_score(y_train, train_preds)),
            "train_samples": len(phrases),
            "positive_count": int(y_train.sum()),
            "negative_count": int(len(y_train) - y_train.sum()),
        }

        if eval_phrases and eval_labels:
            X_eval = self._embed(eval_phrases)
            y_eval = np.array(eval_labels)
            eval_preds = clf.predict(X_eval)
            metrics["eval_accuracy"] = float(accuracy_score(y_eval, eval_preds))
            metrics["eval_f1"] = float(f1_score(y_eval, eval_preds))
            metrics["eval_samples"] = len(eval_phrases)
            logger.info(
                "Eval report:\n%s",
                classification_report(y_eval, eval_preds, target_names=["not_skill", "skill"]),
            )

        self._meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": self.embedding_model_name,
            "metrics": metrics,
        }
        logger.info("Training complete. Metrics: %s", metrics)
        return metrics

    def predict_proba(self, phrases: List[str]) -> List[float]:
        """Return P(skill) for each phrase."""
        if not self.is_trained:
            raise RuntimeError("Classifier not trained; call train() or load() first.")
        if not phrases:
            return []
        X = self._embed(phrases)
        probas = self._clf.predict_proba(X)[:, 1]  # type: ignore[union-attr]
        return probas.tolist()

    def predict(self, phrases: List[str], threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """Return (phrase, probability) pairs for phrases above threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probas = self.predict_proba(phrases)
        return [(p, prob) for p, prob in zip(phrases, probas) if prob >= thresh]

    def save(self, model_dir: str) -> None:
        """Persist model artifacts to disk."""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "classifier.pkl", "wb") as f:
            pickle.dump(self._clf, f)

        with open(path / "metadata.json", "w") as f:
            json.dump(
                {**self._meta, "embedding_model": self.embedding_model_name, "threshold": self.threshold},
                f,
                indent=2,
            )
        logger.info("Model saved to %s", model_dir)

    def load(self, model_dir: str) -> None:
        """Load model artifacts from disk."""
        path = Path(model_dir)
        if not (path / "classifier.pkl").exists():
            raise FileNotFoundError(f"No classifier found at {model_dir}")

        with open(path / "classifier.pkl", "rb") as f:
            self._clf = pickle.load(f)  # noqa: S301

        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
            if "embedding_model" in self._meta:
                self.embedding_model_name = self._meta["embedding_model"]
            if "threshold" in self._meta:
                self.threshold = self._meta["threshold"]

        logger.info("Model loaded from %s", model_dir)
