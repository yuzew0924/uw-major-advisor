from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "pages.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 50
IGNORED_SECTION_HEADINGS = {
    "more information",
    "resources",
    "resources:",
}
PAGE_TYPE_MAP = {
    "admissions": "major_admissions",
    "admissions_prerequisites": "major_admissions",
    "current_uw_admissions": "major_admissions",
    "transfer_admissions": "major_admissions",
    "transfer_course_policy": "major_admissions",
    "requirements": "degree_requirements",
    "non_major_options": "course_registration",
    "double_major_policy": "double_major_policy",
    "double_degree_policy": "general_policy",
}


def encoding_for_model(model: str = EMBEDDING_MODEL) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def should_skip_section(heading: str, text: str) -> bool:
    normalized_heading = clean_text(heading).lower()
    if normalized_heading in IGNORED_SECTION_HEADINGS:
        return True

    # These link-list sections are useful on a web page but noisy for retrieval.
    if len(text.split()) <= 12 and "request more information" in text.lower():
        return True

    return False


def normalized_page_type(page_type: str, heading: str) -> str:
    normalized = PAGE_TYPE_MAP.get(page_type, page_type)
    heading_lower = clean_text(heading).lower()

    if "courses required for admission" in heading_lower:
        return "major_admissions"

    if normalized == "major_admissions":
        return normalized

    if "prerequisite" in heading_lower and "admission" not in heading_lower:
        return "course_prerequisites"

    if re.search(r"\b(cse \d{3}|info \d{3}|stat \d{3}|math \d{3})\b", heading_lower):
        return "course_prerequisites"

    if re.search(r"\b(register|registration|enroll|non-major)\b", heading_lower):
        return "course_registration"

    return normalized


def split_tokens(
    text: str,
    encoding: tiktoken.Encoding,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, int, int]]:
    token_ids = encoding.encode(text)
    if not token_ids:
        return []

    chunks = []
    start = 0
    step = chunk_tokens - overlap_tokens

    if step <= 0:
        raise ValueError("chunk_tokens must be greater than overlap_tokens")

    while start < len(token_ids):
        end = min(start + chunk_tokens, len(token_ids))
        chunk_text = clean_text(encoding.decode(token_ids[start:end]))

        if chunk_text:
            chunks.append((chunk_text, start, end))

        if end == len(token_ids):
            break
        start += step

    return chunks


def build_chunk_id(
    major: str,
    page_type: str,
    section_index: int,
    chunk_index: int,
) -> str:
    return f"{major.lower()}-{page_type}-s{section_index:03d}-c{chunk_index:03d}"


def build_chunks(
    pages: list[dict[str, Any]],
    chunk_tokens: int,
    overlap_tokens: int,
    model: str,
) -> list[dict[str, Any]]:
    encoding = encoding_for_model(model)
    chunks = []

    for page in pages:
        for section_index, section in enumerate(page.get("sections", [])):
            heading = section.get("heading", "")
            section_text = clean_text(section.get("text", ""))
            if not section_text or should_skip_section(heading, section_text):
                continue

            section_chunks = split_tokens(
                section_text,
                encoding,
                chunk_tokens,
                overlap_tokens,
            )

            for chunk_index, (chunk_text, token_start, token_end) in enumerate(section_chunks):
                token_count = token_end - token_start
                page_type = normalized_page_type(page.get("page_type", ""), heading)
                chunk = {
                    "id": build_chunk_id(
                        page["major"],
                        page.get("page_type", "page"),
                        section_index,
                        chunk_index,
                    ),
                    "text": chunk_text,
                    "token_count": token_count,
                    "token_start": token_start,
                    "token_end": token_end,
                    "chunk_index": chunk_index,
                    "section_index": section_index,
                    "metadata": {
                        "major": page["major"],
                        "url": page["url"],
                        "title": page.get("title", ""),
                        "page_type": page_type,
                        "heading": heading,
                        "source_type": page.get("source_type", "official"),
                    },
                }
                chunks.append(chunk)

    return chunks


def write_chunks(chunks: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk scraped UW major pages.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages = json.loads(args.input.read_text(encoding="utf-8"))
    chunks = build_chunks(
        pages,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        model=args.model,
    )
    write_chunks(chunks, args.output)

    total_tokens = sum(chunk["token_count"] for chunk in chunks)
    max_tokens = max((chunk["token_count"] for chunk in chunks), default=0)
    print(f"Wrote {len(chunks)} chunks to {args.output}")
    print(f"Total chunk tokens: {total_tokens}; max chunk tokens: {max_tokens}")


if __name__ == "__main__":
    main()
