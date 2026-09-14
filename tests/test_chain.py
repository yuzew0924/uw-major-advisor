from __future__ import annotations

from typing import Any

import rag.chain as chain
from rag.prerequisites import check_requirements


class FakeOpenAI:
    pass


class FakeRetriever:
    def __init__(self, client: Any) -> None:
        self.client = client

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        major: str | None = None,
    ) -> list[dict[str, Any]]:
        if "double major" in question:
            return [
                {
                    "source_label": "Official",
                    "major": "CSE",
                    "page_type": "double_major_policy",
                    "heading": "Double Majors",
                    "title": "Undergraduate Education",
                    "url": "https://www.cs.washington.edu/academics/undergraduate/degree-requirements/",
                    "text": "Some students choose to pursue a double major or double degree.",
                    "metadata": {
                        "chunk_id": "cse-requirements-s006-c000",
                        "major": "CSE",
                        "page_type": "double_major_policy",
                    },
                },
                {
                    "source_label": "Official",
                    "major": "UW",
                    "page_type": "double_major_policy",
                    "heading": "Is a double major or double degree right for you?",
                    "title": "Declaring a Major",
                    "url": "https://artsci.washington.edu/academics/declaring-a-major",
                    "text": "A double major is pursuing two or more majors that result in a single degree name.",
                    "metadata": {
                        "chunk_id": "uw-double_major_policy-s002-c000",
                        "major": "UW",
                        "page_type": "double_major_policy",
                    },
                },
            ]

        return []


def test_ambiguous_admission_question_requests_major() -> None:
    answer, chunks = chain.answer_question(
        "Can I apply without completing the prerequisites?"
    )

    assert answer == "Which major are you applying to?"
    assert chunks == []


def test_specific_double_major_pair_does_not_guess(monkeypatch: Any) -> None:
    monkeypatch.setattr(chain, "require_api_key", lambda: None)
    monkeypatch.setattr(chain, "OpenAI", lambda default_headers=None: FakeOpenAI())
    monkeypatch.setattr(chain, "Retriever", FakeRetriever)

    answer, _ = chain.answer_question(
        "Can I double major in Computer Science and Informatics?"
    )

    lowered = answer.lower()

    assert "cannot confirm" in lowered
    assert not lowered.startswith("yes")
    assert not lowered.startswith("no")


def test_informatics_prerequisite_checker_is_course_combination_based() -> None:
    result = check_requirements("INFO", {"CSE 122", "STAT 220"})

    assert result is not None
    assert result["satisfied"] == {
        "programming": ["CSE 122"],
        "statistics": ["STAT 220"],
    }
    assert result["missing"] == [
        "INFO 200",
        "one Social Sciences (SSc) course other than INFO 200",
    ]


def test_informatics_missing_prerequisites_answer_uses_checker() -> None:
    chunks = [
        {
            "source_label": "Official",
            "major": "INFO",
            "page_type": "major_admissions",
            "heading": "Informatics Prerequisites",
            "text": "Applicants must complete INFO 200, programming, statistics, and SSc prerequisites.",
        },
        {
            "source_label": "Official",
            "major": "INFO",
            "page_type": "major_admissions",
            "heading": "Informatics Prerequisites",
            "text": "Current UW students must complete all four prerequisites before applying.",
        },
    ]

    answer = chain.answer_informatics_missing_prerequisites(
        "I completed CSE 122 and MATH 124. What Informatics prerequisites am I still missing?",
        chunks,
    )

    assert answer is not None
    assert "programming prerequisite via CSE 122" in answer
    assert "INFO 200" in answer
    assert "one accepted statistics course" in answer
    assert "one Social Sciences (SSc) course other than INFO 200" in answer
