"""Skill extraction backends (regex, ML-based)."""

from jobdistill.extractors.base import SkillExtractor, ExtractionResult
from jobdistill.extractors.regex_extractor import RegexSkillExtractor
from jobdistill.extractors.ml_extractor import MLSkillExtractor

__all__ = [
    "SkillExtractor",
    "ExtractionResult",
    "RegexSkillExtractor",
    "MLSkillExtractor",
]
