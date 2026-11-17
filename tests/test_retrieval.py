from __future__ import annotations

import numpy as np

from rag.retriever import (
    Retriever,
    filter_target_majors,
    infer_query_intent,
    infer_query_majors,
)


def analyze_query(query: str) -> dict[str, object]:
    return {
        "majors": set(infer_query_majors(query)),
        "intent": infer_query_intent(query),
    }


class FakeRetriever(Retriever):
    def __init__(self) -> None:
        self.auto_major_filter = "soft"

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray([[1.0]], dtype=np.float32)

    def retrieve_for_majors(
        self,
        query: str,
        query_vector: np.ndarray,
        majors: list[str],
        per_major_k: int,
        page_types: set[str] | None = None,
    ) -> list[dict[str, object]]:
        return [
            {
                "id": f"{major.lower()}-admissions",
                "major": major,
                "page_type": "major_admissions",
                "source_type": "official",
                "source_label": "Official",
                "final_score": 1.0,
                "score": 1.0,
                "text": f"{major} admissions prerequisites",
            }
            for major in majors
        ]


def test_course_code_does_not_trigger_major_filter() -> None:
    query = "Does CSE 122 satisfy the Informatics programming prerequisite?"

    analysis = analyze_query(query)

    assert analysis["majors"] == {"INFO"}
    assert analysis["intent"] == "course_eligibility"


def test_filter_target_majors_removes_course_code_major_noise() -> None:
    results = [
        {"metadata": {"major": "INFO"}, "id": "info"},
        {"metadata": {"major": "UW"}, "id": "uw"},
        {"metadata": {"major": "CSE"}, "id": "cse"},
        {"metadata": {"major": "ACMS"}, "id": "acms"},
    ]

    filtered = filter_target_majors(results, ["INFO"])

    assert [result["id"] for result in filtered] == ["info", "uw"]


def test_multi_major_admission_comparison() -> None:
    query = "Compare the admission prerequisites for Computer Science and Informatics."

    results = FakeRetriever().retrieve(query, top_k=10)

    majors = {
        result["metadata"]["major"]
        for result in results
    }

    assert "CSE" in majors
    assert "INFO" in majors
    assert all(
        result["metadata"]["page_type"] != "course_prerequisites"
        for result in results
    )


def test_cse_admission_prerequisites_prioritize_pathway_requirements(monkeypatch) -> None:
    retriever = Retriever(client=None)
    monkeypatch.setattr(retriever, "embed_query", lambda _: retriever.index.reconstruct(0)[None, :])

    for question, pathway in (
        ("What are the CSE prerequisites?", "current-uw"),
        ("What are the CSE transfer prerequisites?", "transfers"),
    ):
        results = retriever.retrieve(question)
        assert results[0]["heading"] == "Minimum Academic Requirements"
        assert all(result["url"].rstrip("/").endswith(f"/admissions/{pathway}") for result in results)

    assert infer_query_intent("What are the prerequisites for CSE 414?") == "course_prerequisites"
