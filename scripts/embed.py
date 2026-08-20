from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI, OpenAIError, RateLimitError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import Retriever

DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "chunks" / "index.faiss"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "chunks" / "metadata.json"
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "chunks" / "embeddings.npy"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100
KNOWN_MAJORS = {"INFO", "CSE", "ACMS", "HCDE", "STATS", "AMATH", "MATH"}


def load_chunks(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def embedding_input(chunk: dict[str, Any]) -> str:
    metadata = chunk["metadata"]
    heading = metadata.get("heading", "")
    major = metadata.get("major", "")
    page_type = metadata.get("page_type", "")
    return f"Major: {major}\nPage type: {page_type}\nHeading: {heading}\n\n{chunk['text']}"


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def embed_texts(
    client: OpenAI,
    texts: list[str],
    model: str,
    batch_size: int,
) -> np.ndarray:
    vectors: list[list[float]] = []

    for batch_number, batch in enumerate(batched(texts, batch_size), start=1):
        try:
            response = client.embeddings.create(model=model, input=batch)
        except RateLimitError as error:
            print(f"OpenAI rate limit/quota error: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        except APIConnectionError as error:
            print(f"OpenAI connection error: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        except OpenAIError as error:
            print(f"OpenAI API error: {error}", file=sys.stderr)
            raise SystemExit(1) from error

        vectors.extend(item.embedding for item in response.data)
        print(f"Embedded batch {batch_number}: {len(batch)} texts")

    return np.asarray(vectors, dtype=np.float32)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    normalized = vectors.copy()
    faiss.normalize_L2(normalized)
    return normalized


def build_index(vectors: np.ndarray, metric: str) -> faiss.Index:
    dimension = vectors.shape[1]

    if metric == "ip":
        index = faiss.IndexFlatIP(dimension)
        index.add(normalize_vectors(vectors))
        return index

    if metric == "l2":
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        return index

    raise ValueError(f"Unsupported metric: {metric}")


def metadata_record(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": chunk["id"],
        "text": chunk["text"],
        "token_count": chunk["token_count"],
        "token_start": chunk["token_start"],
        "token_end": chunk["token_end"],
        "chunk_index": chunk["chunk_index"],
        "section_index": chunk["section_index"],
        **chunk["metadata"],
    }


def write_outputs(
    chunks: list[dict[str, Any]],
    vectors: np.ndarray,
    index: faiss.Index,
    args: argparse.Namespace,
) -> None:
    args.index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(args.index_path))
    np.save(args.embeddings_path, vectors)

    payload = {
        "embedding_model": args.model,
        "index_metric": args.metric,
        "normalized_for_index": args.metric == "ip",
        "vector_count": len(chunks),
        "dimension": int(vectors.shape[1]),
        "records": [metadata_record(chunk) for chunk in chunks],
    }
    args.metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_index_and_metadata(
    index_path: Path,
    metadata_path: Path,
) -> tuple[faiss.Index, dict[str, Any]]:
    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return index, metadata


def query_terms(query: str) -> list[str]:
    terms = []
    for term in query.lower().split():
        cleaned = term.strip(",:;()[]{}")
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


def search(
    client: OpenAI,
    query: str,
    index: faiss.Index,
    metadata: dict[str, Any],
    model: str,
    top_k: int,
    major_filter: str | None = None,
) -> list[dict[str, Any]]:
    try:
        response = client.embeddings.create(model=model, input=[query])
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

    if metadata.get("index_metric") == "ip":
        faiss.normalize_L2(query_vector)

    candidate_count = index.ntotal if major_filter else max(top_k * 10, 50)
    scores, indices = index.search(query_vector, min(candidate_count, index.ntotal))
    records = metadata["records"]
    terms = query_terms(query)

    candidates = []
    for score, index_position in zip(scores[0], indices[0], strict=False):
        if index_position < 0:
            continue
        record = records[int(index_position)]
        if major_filter and record.get("major") != major_filter:
            continue
        boost = lexical_boost(record, terms)
        candidates.append(
            {
                "score": float(score) + boost,
                "semantic_score": float(score),
                "lexical_boost": boost,
                **record,
            }
        )

    candidates.sort(key=lambda result: result["score"], reverse=True)
    return candidates[:top_k]


def print_results(results: list[dict[str, Any]]) -> None:
    for rank, result in enumerate(results, start=1):
        preview = " ".join(result["text"].split())[:350]
        print(
            f"\n#{rank} score={result['score']:.4f} "
            f"semantic={result.get('semantic_score', result['score']):.4f} "
            f"{result['major']} | {result['heading']}"
        )
        print(result["url"])
        print(preview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunks and build a FAISS index.")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--metric", choices=("ip", "l2"), default="ip")
    parser.add_argument("--query", help="Optional query to run after building/loading the index.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--major",
        choices=sorted(KNOWN_MAJORS),
        help="Optional metadata filter for search results.",
    )
    parser.add_argument(
        "--no-auto-major-filter",
        action="store_true",
        help="Disable automatic filtering when a query contains one known major code.",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Load an existing FAISS index and metadata, then run --query.",
    )
    return parser.parse_args()


def require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is missing. Add it to .env before running embeddings.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    require_api_key()

    args = parse_args()
    client = OpenAI(default_headers={"Accept-Encoding": "identity"})

    if args.search_only:
        index, metadata = load_index_and_metadata(args.index_path, args.metadata_path)
    else:
        chunks = load_chunks(args.chunks_path)
        texts = [embedding_input(chunk) for chunk in chunks]
        vectors = embed_texts(client, texts, args.model, args.batch_size)
        index = build_index(vectors, args.metric)
        write_outputs(chunks, vectors, index, args)
        metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
        print(f"Wrote FAISS index to {args.index_path}")
        print(f"Wrote metadata to {args.metadata_path}")
        print(f"Wrote vectors to {args.embeddings_path}")

    if args.query:
        retriever = Retriever(
            client=client,
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            embedding_model=args.model,
            auto_major_filter=not args.no_auto_major_filter,
        )
        results = retriever.retrieve(args.query, top_k=args.top_k, major=args.major)
        print_results(results)


if __name__ == "__main__":
    main()
