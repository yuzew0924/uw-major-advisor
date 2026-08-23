from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import APIConnectionError, OpenAI, OpenAIError, RateLimitError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "chunks" / "index.faiss"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "chunks" / "metadata.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
KNOWN_MAJORS = {"INFO", "CSE", "ACMS", "HCDE", "STATS", "AMATH", "MATH"}
COURSE_CODE_PATTERN = re.compile(
    r"\b(?:CSE|INFO|MATH|STAT|AMATH|HCDE|ACMS)\s*\d{3}\b",
    re.IGNORECASE,
)
SUBJECT_COURSE_PHRASE_PATTERN = re.compile(
    r"\b(?:computer science|informatics|statistics|stat|math|mathematics|info|cse)\s+"
    r"(?:course|courses|class|classes)\b",
    re.IGNORECASE,
)
MAJOR_PATTERNS = (
    ("CSE", r"\b(computer science|cs major|cse major)\b"),
    ("INFO", r"\b(informatics|info major)\b"),
    ("HCDE", r"\b(human centered design|hcde major)\b"),
    ("STATS", r"\b(statistics|stats major)\b"),
    ("ACMS", r"\b(acms|applied computational mathematical sciences)\b"),
    ("AMATH", r"\b(applied mathematics|applied math|amath major)\b"),
    ("MATH", r"\b(mathematics|math major)\b"),
)
INTENT_TO_PAGE_TYPES = {
    "course_eligibility": {"major_admissions", "course_prerequisites"},
    "major_admissions": {"major_admissions"},
    "course_prerequisites": {"course_prerequisites"},
    "course_registration": {"course_registration", "course_prerequisites"},
    "double_major_policy": {"double_major_policy", "general_policy"},
    "degree_requirements": {"degree_requirements"},
}
MAJOR_BONUS = 0.08
PER_MAJOR_K = 5


class RetrievalError(RuntimeError):
    pass


def remove_course_codes(query: str) -> str:
    return COURSE_CODE_PATTERN.sub(" ", query)


def remove_subject_course_phrases(query: str) -> str:
    return SUBJECT_COURSE_PHRASE_PATTERN.sub(" ", query)


def extract_course_codes(query: str) -> list[str]:
    return [match.group(0).upper().replace(" ", "") for match in COURSE_CODE_PATTERN.finditer(query)]


def append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def infer_query_majors(query: str) -> list[str]:
    query_without_courses = remove_subject_course_phrases(remove_course_codes(query)).lower()
    majors: list[str] = []

    for major, pattern in MAJOR_PATTERNS:
        if re.search(pattern, query_without_courses):
            append_unique(majors, major)

    for match in re.finditer(r"\b(INFO|CSE|ACMS|HCDE|STATS|AMATH|MATH)\b", query_without_courses.upper()):
        append_unique(majors, match.group(1))

    return majors


def infer_query_major(query: str) -> str | None:
    majors = infer_query_majors(query)
    return majors[0] if len(majors) == 1 else None


def infer_query_intent(query: str) -> str:
    q = query.lower()
    course_codes = extract_course_codes(query)
    majors = infer_query_majors(query)

    if re.search(r"\b(double major|double degree)\b", q):
        return "double_major_policy"

    is_course_eligibility_language = bool(
        re.search(
            r"\b(satisfy|satisfies|count|counts|accepted|eligible|fulfill|fulfills|completed|missing)\b",
            q,
        )
    )
    if course_codes and is_course_eligibility_language:
        return "course_eligibility"

    is_admissions_language = bool(re.search(r"\b(apply|admission|admitted|application)\b", q))
    if course_codes and is_admissions_language and majors:
        return "major_admissions"

    is_major_requirement_question = bool(
        re.search(r"\b(satisfy|satisfies|count|counts|use)\b", q)
        and majors
    )

    if "prerequisite" in q and (is_admissions_language or is_major_requirement_question):
        return "major_admissions"

    if COURSE_CODE_PATTERN.search(query) and "prerequisite" in q:
        return "course_prerequisites"

    if re.search(r"\b(register|registration|enroll|enrollment|remain enrolled)\b", q):
        return "course_registration"

    if re.search(r"\b(degree requirement|degree requirements|graduate|graduation)\b", q):
        return "degree_requirements"

    return "general"


def page_types_for_intent(intent: str) -> set[str] | None:
    return INTENT_TO_PAGE_TYPES.get(intent)


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []

    for result in results:
        chunk_id = result.get("id") or result.get("metadata", {}).get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique.append(result)

    return unique


def check_major_coverage(
    results: list[dict[str, Any]],
    required_majors: list[str],
) -> dict[str, Any]:
    retrieved_majors = {
        result.get("metadata", result).get("major")
        for result in results
    }
    missing = [
        major
        for major in required_majors
        if major not in retrieved_majors
    ]
    return {"complete": len(missing) == 0, "missing": missing}


def check_topic_coverage(
    results: list[dict[str, Any]],
    required_majors: list[str],
    required_page_type: str,
) -> list[str]:
    missing = []

    for major in required_majors:
        found = any(
            result.get("metadata", result).get("major") == major
            and result.get("metadata", result).get("page_type") == required_page_type
            for result in results
        )
        if not found:
            missing.append(major)

    return missing


def filter_target_majors(
    results: list[dict[str, Any]],
    inferred_majors: list[str],
) -> list[dict[str, Any]]:
    if not inferred_majors:
        return results

    allowed_majors = set(inferred_majors) | {"UW"}
    return [
        result
        for result in results
        if result.get("metadata", result).get("major") in allowed_majors
    ]


def query_terms(query: str) -> list[str]:
    terms = []
    for term in query.lower().split():
        cleaned = term.strip(",:;()[]{}?")
        if len(cleaned) < 3 or cleaned.upper() in KNOWN_MAJORS:
            continue
        terms.append(cleaned)
    return terms


def lexical_boost(record: dict[str, Any], terms: list[str]) -> float:
    heading = record.get("heading", "").lower()
    page_type = record.get("page_type", "").lower()
    text = record.get("text", "").lower()

    boost = 0.0
    for term in terms:
        if term in heading or term in page_type:
            boost += 0.05
        elif term in text:
            boost += 0.015

    return min(boost, 0.10)


def source_label(source_type: str) -> str:
    if source_type == "student_experience":
        return "Student Experience"
    return "Official"


class Retriever:
    def __init__(
        self,
        client: OpenAI,
        index_path: Path = DEFAULT_INDEX_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        embedding_model: str | None = None,
        auto_major_filter: bool | str = "soft",
    ) -> None:
        if not index_path.exists():
            raise RetrievalError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise RetrievalError(f"Metadata file not found: {metadata_path}")

        self.client = client
        self.index = faiss.read_index(str(index_path))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.records = self.metadata["records"]
        self.embedding_model = embedding_model or self.metadata.get(
            "embedding_model",
            DEFAULT_EMBEDDING_MODEL,
        )
        self.auto_major_filter = auto_major_filter

    def embed_query(self, query: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[query],
            )
        except RateLimitError as error:
            print(f"OpenAI rate limit/quota error: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        except APIConnectionError as error:
            print(f"OpenAI connection error: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        except OpenAIError as error:
            print(f"OpenAI API error: {error}", file=sys.stderr)
            raise SystemExit(1) from error

        query_vector = np.asarray([response.data[0].embedding], dtype=np.float32)
        if self.metadata.get("index_metric") == "ip":
            faiss.normalize_L2(query_vector)
        return query_vector

    def format_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        metadata_fields = (
            "id",
            "major",
            "url",
            "title",
            "page_type",
            "heading",
            "source_type",
            "chunk_index",
            "section_index",
            "token_start",
            "token_end",
            "token_count",
        )

        for rank, result in enumerate(results, start=1):
            result["rank"] = rank
            result["metadata"] = {
                "chunk_id": result.get("id"),
                **{
                    field: result.get(field)
                    for field in metadata_fields
                    if field in result
                },
            }

        return results

    def search_candidates(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int,
        major_filter: str | None = None,
        target_majors: list[str] | None = None,
        page_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        target_majors = target_majors or []
        candidate_count = (
            self.index.ntotal
            if major_filter or page_types
            else max(top_k * 20, 100)
        )
        scores, indices = self.index.search(
            query_vector,
            min(candidate_count, self.index.ntotal),
        )
        terms = query_terms(query)
        candidates = []

        for score, index_position in zip(scores[0], indices[0], strict=False):
            if index_position < 0:
                continue

            record = self.records[int(index_position)]
            if major_filter and record.get("major") != major_filter:
                continue
            if page_types and record.get("page_type") not in page_types:
                continue

            boost = lexical_boost(record, terms)
            major_bonus = MAJOR_BONUS if record.get("major") in target_majors else 0.0
            candidates.append(
                {
                    "score": float(score),
                    "final_score": float(score) + boost + major_bonus,
                    "semantic_score": float(score),
                    "lexical_boost": boost,
                    "major_bonus": major_bonus,
                    "source_label": source_label(record.get("source_type", "official")),
                    **record,
                }
            )

        candidates.sort(key=lambda result: result["final_score"], reverse=True)
        return candidates[:top_k]

    def retrieve_for_majors(
        self,
        query: str,
        query_vector: np.ndarray,
        majors: list[str],
        per_major_k: int,
        page_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        combined = []

        for major in majors:
            major_results = self.search_candidates(
                query=query,
                query_vector=query_vector,
                top_k=max(per_major_k, 20),
                major_filter=major,
                page_types=page_types,
            )
            combined.extend(major_results[:per_major_k])

        combined = deduplicate_results(combined)
        combined.sort(key=lambda result: result["final_score"], reverse=True)
        return combined

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        major: str | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = self.embed_query(query)
        intent = infer_query_intent(query)
        page_types = page_types_for_intent(intent)

        if major:
            results = self.search_candidates(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                major_filter=major,
                page_types=page_types,
            )
            return self.format_results(results)

        target_majors = infer_query_majors(query)
        if not self.auto_major_filter:
            results = self.search_candidates(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                page_types=page_types,
            )
            results = filter_target_majors(results, target_majors)
            return self.format_results(results)

        if intent == "double_major_policy":
            results = self.search_candidates(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                target_majors=target_majors,
                page_types=page_types,
            )
            results = filter_target_majors(results, target_majors)
            return self.format_results(results)

        if len(target_majors) > 1:
            per_major_k = max(PER_MAJOR_K, top_k // len(target_majors))
            results = self.retrieve_for_majors(
                query=query,
                query_vector=query_vector,
                majors=target_majors,
                per_major_k=per_major_k,
                page_types=page_types,
            )
            results = filter_target_majors(results, target_majors)
            return self.format_results(results)

        results = self.search_candidates(
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            target_majors=target_majors,
            page_types=page_types,
        )
        results = filter_target_majors(results, target_majors)
        return self.format_results(results)
