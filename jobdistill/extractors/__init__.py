"""Skill extraction backends (gemini, regex)."""

from jobdistill.extractors.base import SkillExtractor, ExtractionResult
from jobdistill.extractors.regex_extractor import RegexSkillExtractor
from jobdistill.extractors.gemini_extractor import GeminiSkillExtractor

__all__ = [
    "SkillExtractor",
    "ExtractionResult",
    "RegexSkillExtractor",
    "GeminiSkillExtractor",
]
