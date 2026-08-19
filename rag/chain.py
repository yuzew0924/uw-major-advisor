from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI, OpenAIError, RateLimitError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import KNOWN_MAJORS, Retriever


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 5

SYSTEM_PROMPT = """You are UW Major Advisor, a concise assistant for University of Washington major admissions and requirements.

Rules:
- Answer only from the provided retrieved context. Do not use outside knowledge.
- If the context is insufficient, say what is missing and suggest checking the official page.
- Cite sources inline using the labels from context, such as [Official S1].
- The MVP context is official-source only; do not invent student experience evidence.
- Never guarantee admission. Use cautious language such as "Historically, competitive applicants..." only if supported by context.
- Keep the answer brief. Students are busy.
"""


def require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is missing. Add it to .env.", file=sys.stderr)
        raise SystemExit(1)


def format_context(chunks: list[dict[str, Any]]) -> str:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        source_id = f"S{index}"
        context_blocks.append(
            "\n".join(
                [
                    f"[{chunk['source_label']} {source_id}]",
                    f"Major: {chunk.get('major', '')}",
                    f"Page type: {chunk.get('page_type', '')}",
                    f"Heading: {chunk.get('heading', '')}",
                    f"URL: {chunk.get('url', '')}",
                    "Text:",
                    chunk.get("text", ""),
                ]
            )
        )
    return "\n\n---\n\n".join(context_blocks)


def build_messages(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    context = format_context(chunks)
    user_prompt = f"""Retrieved context:

{context}

User question:
{question}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(
    client: OpenAI,
    question: str,
    chunks: list[dict[str, Any]],
    model: str,
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(question, chunks),
            temperature=0.2,
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

    content = response.choices[0].message.content
    return content.strip() if content else ""


def print_sources(chunks: list[dict[str, Any]]) -> None:
    print("\nSources:")
    seen = set()
    for index, chunk in enumerate(chunks, start=1):
        key = (chunk.get("source_label"), chunk.get("url"), chunk.get("heading"))
        if key in seen:
            continue
        seen.add(key)
        print(
            f"[{chunk['source_label']} S{index}] "
            f"{chunk.get('major', '')} | {chunk.get('heading', '')} | {chunk.get('url', '')}"
        )


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    model: str = DEFAULT_MODEL,
    major: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    load_dotenv(PROJECT_ROOT / ".env")
    require_api_key()

    client = OpenAI()
    retriever = Retriever(client=client)
    chunks = retriever.retrieve(question, top_k=top_k, major=major)
    if not chunks:
        return "I could not find relevant retrieved context for this question.", []

    answer = generate_answer(client, question, chunks, model=model)
    return answer, chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the UW Major Advisor RAG pipeline.")
    parser.add_argument("question", nargs="?", help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--major", choices=sorted(KNOWN_MAJORS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = args.question or input("Question: ").strip()
    if not question:
        print("Error: question is required.", file=sys.stderr)
        raise SystemExit(1)

    answer, chunks = answer_question(
        question,
        top_k=args.top_k,
        model=args.model,
        major=args.major,
    )
    print(answer)
    if chunks:
        print_sources(chunks)


if __name__ == "__main__":
    main()
