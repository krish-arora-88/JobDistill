"""Gemini LLM-based skill extraction from job posting text."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from jobdistill.extractors.base import ExtractionResult, SkillExtractor

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 6000

SYSTEM_PROMPT = """\
You are a technical skill extractor. Given job posting text, extract ONLY \
concrete technical skills, tools, languages, frameworks, platforms, databases, \
and protocols that a candidate would put on a resume.

Rules:
- Return each skill with its canonical/normalized name (e.g. "JavaScript" not "JS", \
"PostgreSQL" not "postgres", "Amazon Web Services" should be "AWS").
- Categorize each skill as one of: Language, Framework, Library, Tool, Platform, \
Protocol, Database, Cloud, Methodology, Other.
- Do NOT include soft skills, job titles, degree names, or generic terms \
(e.g. "communication", "teamwork", "Bachelor's", "Software Engineer").
- Do NOT include company names, product names that aren't technical tools, \
or business domains.
- Deduplicate: if the same skill appears multiple times, return it only once.
- If the text contains no technical skills, return an empty list.
"""


class ExtractedSkill(BaseModel):
    name: str
    category: str


class SkillExtractionResponse(BaseModel):
    skills: List[ExtractedSkill]


class GeminiSkillExtractor(SkillExtractor):
    """Extracts skills using Gemini LLM with structured JSON output."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        self._model_name = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._max_retries = max_retries
        self._client = None

    def _get_client(self):
        """Lazy-init the Gemini client."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. Export it or pass api_key= to the extractor."
                )
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "gemini"

    def extract(self, text: str) -> ExtractionResult:
        """Extract skills from a single document's text via Gemini."""
        if not text or not text.strip():
            return ExtractionResult(skills={}, candidates_considered=0)

        truncated = text[:MAX_TEXT_CHARS]
        client = self._get_client()

        response = self._call_with_retry(client, truncated)

        skills: Dict[str, float] = {}
        categories: Dict[str, str] = {}
        for skill in response.skills:
            norm = skill.name.strip()
            if norm and norm not in skills:
                skills[norm] = 1.0
                categories[norm] = skill.category

        return ExtractionResult(
            skills=skills,
            candidates_considered=len(response.skills),
            debug_info={"categories": categories, "model": self._model_name},
        )

    def _call_with_retry(
        self, client, text: str
    ) -> SkillExtractionResponse:
        """Call Gemini API with exponential backoff on transient errors."""
        from google.genai import types

        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                response = client.models.generate_content(
                    model=self._model_name,
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=SkillExtractionResponse,
                    ),
                )
                return response.parsed
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "429" in err_str or "5" in err_str[:1]:
                    wait = 2 ** attempt
                    logger.warning(
                        "Gemini API error (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1,
                        self._max_retries,
                        e,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(
            f"Gemini API failed after {self._max_retries} retries: {last_err}"
        )
