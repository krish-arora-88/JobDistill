"""Smoke tests for the ML extractor pipeline using synthetic text."""

import pytest

from jobdistill.extractors.ml_extractor import MLSkillExtractor, _find_example_mentions


SYNTHETIC_JOB_TEXT = """
Senior Software Engineer — Full Stack

We are looking for a Senior Software Engineer to join our team.
You will work with Python, JavaScript, and TypeScript to build
scalable web applications using React and Node.js.

Requirements:
- 5+ years experience with Python, Django, or Flask
- Strong knowledge of SQL, PostgreSQL, and Redis
- Experience with Docker and Kubernetes for container orchestration
- Familiarity with AWS services (EC2, S3, Lambda)
- Understanding of CI/CD pipelines using Jenkins or GitHub Actions
- Experience with machine learning frameworks like TensorFlow or PyTorch
- Knowledge of RESTful API design and GraphQL
- Proficiency in Git version control

Nice to have:
- Experience with Kafka and event-driven architectures
- Knowledge of Terraform for infrastructure as code
- Familiarity with Agile/Scrum methodologies
"""


@pytest.fixture
def ml_extractor():
    return MLSkillExtractor(
        model_dir=None,
        top_k=20,
        min_confidence=0.5,
    )


class TestMLExtractorSmoke:
    def test_extract_returns_result(self, ml_extractor):
        result = ml_extractor.extract(SYNTHETIC_JOB_TEXT)
        assert result is not None
        assert isinstance(result.skills, dict)
        assert result.candidates_considered > 0

    def test_extract_finds_skills(self, ml_extractor):
        result = ml_extractor.extract(SYNTHETIC_JOB_TEXT)
        assert len(result.skills) > 0, "Should find at least some skill phrases"

    def test_extract_empty_text(self, ml_extractor):
        result = ml_extractor.extract("")
        assert len(result.skills) == 0
        assert result.candidates_considered == 0

    def test_extract_short_text(self, ml_extractor):
        result = ml_extractor.extract("Python and Java")
        assert isinstance(result.skills, dict)

    def test_confidence_values_in_range(self, ml_extractor):
        result = ml_extractor.extract(SYNTHETIC_JOB_TEXT)
        for skill, conf in result.skills.items():
            assert 0.0 <= conf <= 1.0, f"Confidence for {skill} out of range: {conf}"


class TestFindExampleMentions:
    def test_finds_mentions(self):
        text = "We use Python for backend development. Python is great."
        mentions = _find_example_mentions(text, "Python", max_examples=2)
        assert len(mentions) > 0
        assert all("Python" in m for m in mentions)

    def test_max_examples(self):
        text = "Python Python Python Python Python"
        mentions = _find_example_mentions(text, "Python", max_examples=2)
        assert len(mentions) <= 2

    def test_no_match(self):
        text = "We use Java for everything."
        mentions = _find_example_mentions(text, "Haskell")
        assert mentions == []
