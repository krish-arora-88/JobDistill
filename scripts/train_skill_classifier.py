#!/usr/bin/env python3
"""Train the skill-likeness classifier and save model artifacts.

Usage:
    python scripts/train_skill_classifier.py \
        --data data/training_data.jsonl \
        --model_dir models/skill_classifier \
        --eval_split 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobdistill.extractors.classifier import SkillClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(path: str) -> tuple[list[str], list[int]]:
    phrases: list[str] = []
    labels: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            phrases.append(rec["text"])
            labels.append(int(rec["label"]))
    return phrases, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train skill classifier")
    parser.add_argument("--data", type=str, required=True, help="JSONL training data")
    parser.add_argument("--model_dir", type=str, default="models/skill_classifier")
    parser.add_argument("--eval_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    random.seed(args.seed)

    phrases, labels = load_dataset(args.data)
    logger.info("Loaded %d samples (%d positive, %d negative)",
                len(phrases), sum(labels), len(labels) - sum(labels))

    indices = list(range(len(phrases)))
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - args.eval_split))

    train_idx = indices[:split_idx]
    eval_idx = indices[split_idx:]

    train_phrases = [phrases[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    eval_phrases = [phrases[i] for i in eval_idx]
    eval_labels = [labels[i] for i in eval_idx]

    clf = SkillClassifier(
        embedding_model=args.embedding_model,
        threshold=args.threshold,
    )

    metrics = clf.train(
        train_phrases, train_labels,
        eval_phrases=eval_phrases,
        eval_labels=eval_labels,
    )

    clf.save(args.model_dir)
    logger.info("Training complete. Model saved to %s", args.model_dir)
    logger.info("Final metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
