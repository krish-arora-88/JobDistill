"""Abstract interface for skill extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractionResult:
    """Per-document extraction output."""

    skills: Dict[str, float]  # skill_phrase -> confidence (1.0 for regex)
    candidates_considered: int = 0
    example_mentions: Dict[str, List[str]] = field(default_factory=dict)
    debug_info: Optional[Dict[str, Any]] = None

    @property
    def skill_names(self) -> List[str]:
        return list(self.skills.keys())


class SkillExtractor(ABC):
    """Interface that both regex and ML extractors implement."""

    @abstractmethod
    def extract(self, text: str) -> ExtractionResult:
        """Extract skills from a single document's text.

        Returns an ExtractionResult with skill->confidence mapping.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the extractor."""
