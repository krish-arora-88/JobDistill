"""Tests for GeminiSkillExtractor — all Gemini API calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jobdistill.extractors.gemini_extractor import (
    SYSTEM_PROMPT,
    MAX_TEXT_CHARS,
    ExtractedSkill,
    GeminiSkillExtractor,
    SkillExtractionResponse,
)


def _mock_response(skills: list[dict]) -> MagicMock:
    """Build a mock Gemini response with .parsed returning a SkillExtractionResponse."""
    parsed = SkillExtractionResponse(
        skills=[ExtractedSkill(**s) for s in skills]
    )
    resp = MagicMock()
    resp.parsed = parsed
    return resp


class TestValidExtraction:
    def test_extracts_skills(self):
        ext = GeminiSkillExtractor(api_key="fake-key")
        mock_resp = _mock_response([
            {"name": "Python", "category": "Language"},
            {"name": "Docker", "category": "Tool"},
            {"name": "AWS", "category": "Cloud"},
        ])
        with patch.object(ext, "_get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            result = ext.extract("We need Python, Docker, and AWS experience.")

        assert "Python" in result.skills
        assert "Docker" in result.skills
        assert "AWS" in result.skills
        assert all(v == 1.0 for v in result.skills.values())

    def test_categories_in_debug_info(self):
        ext = GeminiSkillExtractor(api_key="fake-key")
        mock_resp = _mock_response([
            {"name": "React", "category": "Framework"},
            {"name": "PostgreSQL", "category": "Database"},
        ])
        with patch.object(ext, "_get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            result = ext.extract("React and PostgreSQL required.")

        assert result.debug_info is not None
        cats = result.debug_info["categories"]
        assert cats["React"] == "Framework"
        assert cats["PostgreSQL"] == "Database"

    def test_model_in_debug_info(self):
        ext = GeminiSkillExtractor(model="gemini-2.5-flash-lite", api_key="fake-key")
        mock_resp = _mock_response([{"name": "Go", "category": "Language"}])
        with patch.object(ext, "_get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            result = ext.extract("Go experience required.")

        assert result.debug_info["model"] == "gemini-2.5-flash-lite"


class TestEmptyText:
    def test_empty_string(self):
        ext = GeminiSkillExtractor(api_key="fake-key")
        result = ext.extract("")
        assert result.skills == {}
        assert result.candidates_considered == 0

    def test_whitespace_only(self):
        ext = GeminiSkillExtractor(api_key="fake-key")
        result = ext.extract("   \n\t  ")
        assert result.skills == {}


class TestTextTruncation:
    def test_long_text_truncated(self):
        ext = GeminiSkillExtractor(api_key="fake-key")
        long_text = "x" * (MAX_TEXT_CHARS + 5000)
        mock_resp = _mock_response([{"name": "Python", "category": "Language"}])

        with patch.object(ext, "_get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            ext.extract(long_text)
            call_args = mock_client.return_value.models.generate_content.call_args
            passed_text = call_args.kwargs.get("contents") or call_args[1].get("contents")
            assert len(passed_text) == MAX_TEXT_CHARS


class TestAPIErrorHandling:
    def test_retries_on_429(self):
        ext = GeminiSkillExtractor(api_key="fake-key", max_retries=3)
        mock_resp = _mock_response([{"name": "Python", "category": "Language"}])

        with patch.object(ext, "_get_client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = [
                Exception("429 Too Many Requests"),
                mock_resp,
            ]
            with patch("jobdistill.extractors.gemini_extractor.time.sleep"):
                result = ext.extract("Python required.")

        assert "Python" in result.skills

    def test_raises_non_retryable(self):
        ext = GeminiSkillExtractor(api_key="fake-key", max_retries=3)
        with patch.object(ext, "_get_client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = ValueError("bad input")
            with pytest.raises(ValueError, match="bad input"):
                ext.extract("test text")

    def test_raises_after_max_retries(self):
        ext = GeminiSkillExtractor(api_key="fake-key", max_retries=2)
        with patch.object(ext, "_get_client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = Exception("500 Server Error")
            with patch("jobdistill.extractors.gemini_extractor.time.sleep"):
                with pytest.raises(RuntimeError, match="failed after 2 retries"):
                    ext.extract("test text")


class TestMissingAPIKey:
    def test_no_key_raises(self):
        ext = GeminiSkillExtractor(api_key=None)
        ext._api_key = None
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
                ext.extract("test text")


class TestDeduplication:
    def test_duplicates_removed(self):
        ext = GeminiSkillExtractor(api_key="fake-key")
        mock_resp = _mock_response([
            {"name": "Python", "category": "Language"},
            {"name": "Python", "category": "Language"},
            {"name": "Docker", "category": "Tool"},
        ])
        with patch.object(ext, "_get_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_resp
            result = ext.extract("Python Python Docker")

        assert len(result.skills) == 2
        assert result.candidates_considered == 3


class TestPromptContent:
    def test_system_prompt_mentions_categories(self):
        assert "Language" in SYSTEM_PROMPT
        assert "Framework" in SYSTEM_PROMPT
        assert "Database" in SYSTEM_PROMPT
        assert "Cloud" in SYSTEM_PROMPT

    def test_system_prompt_excludes_soft_skills(self):
        assert "soft skills" in SYSTEM_PROMPT.lower() or "Do NOT include soft skills" in SYSTEM_PROMPT


class TestPydanticSchema:
    def test_extracted_skill_model(self):
        skill = ExtractedSkill(name="Python", category="Language")
        assert skill.name == "Python"
        assert skill.category == "Language"

    def test_response_model(self):
        resp = SkillExtractionResponse(skills=[
            ExtractedSkill(name="Python", category="Language"),
            ExtractedSkill(name="Docker", category="Tool"),
        ])
        assert len(resp.skills) == 2


class TestExtractorName:
    def test_name_is_gemini(self):
        ext = GeminiSkillExtractor(api_key="fake-key")
        assert ext.name == "gemini"
