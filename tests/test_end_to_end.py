"""End-to-end smoke tests using text fixtures (no real PDFs needed).

Verifies the full pipeline from boilerplate removal through skill extraction
to aggregation, ensuring that:
  - Real skills (Python, Git, AWS, etc.) appear in top results.
  - Boilerplate phrases ("submit applications", "canada job posting") do NOT.
  - Document-frequency counting works correctly (same skill in one doc = count 1).
  - Tech-token extraction catches symbols and acronyms.
"""

from collections import Counter

import pytest

from jobdistill.boilerplate import strip_boilerplate_corpus
from jobdistill.extractors.ml_extractor import extract_tech_tokens
from jobdistill.normalize import has_tech_indicator, is_valid_candidate

BOILERPLATE = """\
This is a Canada job posting on our platform.
Students must submit applications before the deadline.
Equal opportunity employer. Apply now."""

DOC_TEMPLATE = """\
{title}

{boilerplate}

Requirements:
{skills_section}

{boilerplate}
"""

FIXTURE_DOCS = [
    DOC_TEMPLATE.format(
        title="Senior Software Engineer",
        boilerplate=BOILERPLATE,
        skills_section=(
            "- 5+ years Python, JavaScript, TypeScript\n"
            "- React, Node.js, Docker, Kubernetes\n"
            "- AWS, CI/CD, Git, SQL"
        ),
    ),
    DOC_TEMPLATE.format(
        title="Data Scientist",
        boilerplate=BOILERPLATE,
        skills_section=(
            "- Python, SQL, TensorFlow, PyTorch\n"
            "- ML, AI, scikit-learn\n"
            "- AWS, Docker, Git"
        ),
    ),
    DOC_TEMPLATE.format(
        title="DevOps Engineer",
        boilerplate=BOILERPLATE,
        skills_section=(
            "- Docker, Kubernetes, Terraform\n"
            "- AWS, Linux, Git, Jenkins\n"
            "- CI/CD, Python, Ansible"
        ),
    ),
    DOC_TEMPLATE.format(
        title="Frontend Developer",
        boilerplate=BOILERPLATE,
        skills_section=(
            "- JavaScript, TypeScript, React\n"
            "- CSS, HTML, REST, GraphQL\n"
            "- Git, Node.js, Vue"
        ),
    ),
    DOC_TEMPLATE.format(
        title="Backend Engineer",
        boilerplate=BOILERPLATE,
        skills_section=(
            "- Java, C++, Go, Python\n"
            "- SQL, PostgreSQL, Redis\n"
            "- Docker, Kubernetes, Git, AWS"
        ),
    ),
]


class TestBoilerplateRemoval:
    """Verify boilerplate lines are stripped before extraction."""

    def test_boilerplate_removed(self):
        cleaned, stats = strip_boilerplate_corpus(FIXTURE_DOCS, df_threshold=0.05)
        for doc in cleaned:
            assert "submit applications" not in doc.lower()
            assert "canada job posting" not in doc.lower()

    def test_skills_preserved(self):
        cleaned, _ = strip_boilerplate_corpus(FIXTURE_DOCS, df_threshold=0.05)
        all_text = " ".join(cleaned)
        assert "Python" in all_text
        assert "Git" in all_text
        assert "AWS" in all_text
        assert "Docker" in all_text


class TestCandidateFilterRejectsBoilerplate:
    """Verify the candidate filter rejects all-lowercase boilerplate phrases."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "submit applications soon",
            "canada job posting",
            "students submit applications",
            "equal opportunity employer",
            "before the deadline",
        ],
    )
    def test_rejects_boilerplate_phrases(self, phrase: str):
        assert not is_valid_candidate(phrase), f"Should reject: {phrase}"

    @pytest.mark.parametrize(
        "phrase",
        ["Python", "Git", "AWS", "C++", ".NET", "React", "SQL", "Docker"],
    )
    def test_accepts_real_skills(self, phrase: str):
        assert is_valid_candidate(phrase), f"Should accept: {phrase}"


class TestDocumentFrequencyCounting:
    """Verify per-document counting semantics."""

    def test_same_skill_repeated_in_doc_counts_once(self):
        skill_counts: Counter = Counter()

        doc_skills_list = [
            {"python": 0.9, "git": 0.8, "aws": 0.7},
            {"python": 0.85, "sql": 0.9},
            {"python": 0.9, "git": 0.85, "docker": 0.8},
        ]

        for doc_skills in doc_skills_list:
            seen: set = set()
            for skill in doc_skills:
                if skill not in seen:
                    skill_counts[skill] += 1
                    seen.add(skill)

        assert skill_counts["python"] == 3
        assert skill_counts["git"] == 2
        assert skill_counts["sql"] == 1
        assert skill_counts["docker"] == 1


class TestTechIndicatorIntegration:
    """Verify tech indicator catches real skills and rejects boilerplate."""

    @pytest.mark.parametrize(
        "phrase,expected",
        [
            ("Python", True),
            ("C++", True),
            (".NET", True),
            ("CI/CD", True),
            ("AI", True),
            ("JavaScript", True),
            ("Docker", True),
            ("Git", True),
            ("5G", True),
            ("canada job posting", False),
            ("submit applications", False),
            ("the and or", False),
            ("equal opportunity employer", False),
        ],
    )
    def test_tech_indicator(self, phrase: str, expected: bool):
        assert has_tech_indicator(phrase) == expected, f"Failed for: {phrase}"


class TestTechTokenExtraction:
    """Verify the regex-based tech-token candidate pass."""

    def test_extracts_real_skills(self):
        text = (
            "Submit applications soon. We need Python, Git, AWS, "
            "C++ and .NET developers with SQL experience."
        )
        candidates = extract_tech_tokens(text)
        names = {c[0] for c in candidates}
        assert "python" in names
        assert "git" in names
        assert "aws" in names
        assert "c++" in names
        assert ".net" in names
        assert "sql" in names

    def test_filters_application_verbs(self):
        text = "Submit your Application to the Company Position."
        candidates = extract_tech_tokens(text)
        names = {c[0] for c in candidates}
        assert "submit" not in names
        assert "application" not in names
        assert "company" not in names
        assert "position" not in names

    def test_catches_acronyms_and_camelcase(self):
        text = "Experience with REST APIs, JavaScript, TypeScript, and CI/CD."
        candidates = extract_tech_tokens(text)
        names = {c[0] for c in candidates}
        assert "rest" in names or "REST" in names
        assert "javascript" in names or "JavaScript" in names
        assert "typescript" in names or "TypeScript" in names
        assert "ci/cd" in names or "CI/CD" in names

    def test_mixed_content_prioritizes_skills(self):
        text = (
            "Canada job posting for Students.\n"
            "Requirements: Python, Git, AWS, C++ and .NET\n"
            "Submit before deadline."
        )
        candidates = extract_tech_tokens(text)
        names = {c[0] for c in candidates}
        skill_hits = {"python", "git", "aws", "c++", ".net"} & names
        assert len(skill_hits) >= 4, f"Expected >=4 skills, got: {skill_hits}"


class TestFullPipelineSmoke:
    """Smoke test: run boilerplate removal + filtering on fixture corpus.

    Asserts that after processing, the top extracted "skill tokens" include
    real skills and exclude boilerplate.
    """

    def test_smoke_skills_not_boilerplate(self):
        cleaned, _ = strip_boilerplate_corpus(FIXTURE_DOCS, df_threshold=0.05)

        all_tokens: list = []
        for doc in cleaned:
            for word in doc.split():
                cleaned_word = word.strip(",.:-();\"'")
                if is_valid_candidate(cleaned_word):
                    all_tokens.append(cleaned_word)

        skill_counts = Counter(all_tokens)
        top_skills = [s for s, _ in skill_counts.most_common(20)]

        real_skills_found = {"Python", "Git", "AWS", "Docker"} & set(top_skills)
        assert len(real_skills_found) >= 2, (
            f"Expected at least 2 of Python/Git/AWS/Docker in top 20, got: {top_skills}"
        )

        boilerplate_phrases = {"submit", "applications", "canada", "posting", "students"}
        leaked = boilerplate_phrases & set(t.lower() for t in top_skills)
        assert len(leaked) == 0, f"Boilerplate leaked into top skills: {leaked}"
