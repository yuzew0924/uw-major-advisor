from __future__ import annotations

import json
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


class RetrievalError(RuntimeError):
    pass


def infer_query_major(query: str) -> str | None:
    query_terms = {term.strip(",:;()[]{}").upper() for term in query.split()}
    matches = sorted(KNOWN_MAJORS & query_terms)
    return matches[0] if len(matches) == 1 else None


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
        auto_major_filter: bool = True,
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

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        major: str | None = None,
    ) -> list[dict[str, Any]]:
        major_filter = major
        if not major_filter and self.auto_major_filter:
            major_filter = infer_query_major(query)

        query_vector = self.embed_query(query)
        candidate_count = self.index.ntotal if major_filter else max(top_k * 10, 50)
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

            boost = lexical_boost(record, terms)
            candidates.append(
                {
                    "score": float(score) + boost,
                    "semantic_score": float(score),
                    "lexical_boost": boost,
                    "source_label": source_label(record.get("source_type", "official")),
                    **record,
                }
            )

        candidates.sort(key=lambda result: result["score"], reverse=True)
        return candidates[:top_k]
