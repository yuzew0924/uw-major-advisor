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

from rag.prerequisites import check_requirements
from rag.retriever import (
    KNOWN_MAJORS,
    Retriever,
    check_topic_coverage,
    extract_course_codes,
    infer_query_intent,
    infer_query_majors,
)


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 5
MAJOR_ALIASES = {
    "CSE": ("computer science", "cse", "allen school"),
    "INFO": ("informatics", "info"),
    "STATS": ("statistics", "stat"),
    "ACMS": ("acms", "applied computational mathematical sciences"),
    "HCDE": ("hcde", "human centered design"),
    "AMATH": ("amath", "applied mathematics", "applied math"),
    "MATH": ("mathematics", "math"),
}

SYSTEM_PROMPT = """You are UW Major Advisor, a concise assistant for University of Washington major admissions and requirements.

Rules:
- Answer only from the provided retrieved context. Do not use outside knowledge.
- If the context is insufficient, say what is missing and suggest checking the official page.
- Cite sources inline using the labels from context, such as [Official S1].
- Cite sources for statements about insufficient evidence by referencing the retrieved sources you reviewed.
- The MVP context is official-source only; do not invent student experience evidence.
- Never guarantee admission. Use cautious language such as "Historically, competitive applicants..." only if supported by context.
- Preserve applicant-pathway distinctions. Do not generalize a transfer-student exception to current UW students, or a current-UW rule to transfer students.
- If the question does not specify an applicant pathway and the sources distinguish pathways, state the relevant pathway distinction.
- For Informatics admissions, current UW students must be treated separately from transfer applicants when the retrieved context distinguishes them; do not say INFO 200 can be deferred unless the user is a transfer applicant.
- If the user lists completed courses and asks what prerequisites are still missing, identify every required prerequisite category from the retrieved context, mark only explicitly completed matching categories as satisfied, and list every remaining unresolved category.
- If a course-number prerequisite question retrieves related course-policy context but no explicit prerequisite for that course, say the retrieved sources do not state the prerequisite and cite the closest retrieved source.
- For a question about a specific combination of majors, give a definitive yes or no only when the sources explicitly address that combination or provide a general rule that unambiguously applies to it.
- A general double-major policy alone does not prove that a specific combination is permitted or prohibited.
- If the evidence is insufficient, clearly state that the specific combination cannot be confirmed from the retrieved official sources.
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


def source_ref(chunks: list[dict[str, Any]], index: int) -> str:
    if index >= len(chunks):
        return ""
    return f"[{chunks[index]['source_label']} S{index + 1}]"


def answer_informatics_missing_prerequisites(
    question: str,
    chunks: list[dict[str, Any]],
) -> str | None:
    q = question.lower()
    majors = infer_query_majors(question)
    if majors != ["INFO"] or "missing" not in q or "prerequisite" not in q:
        return None

    completed_courses = set(extract_course_codes(question))
    if not completed_courses:
        return None

    requirement_result = check_requirements("INFO", completed_courses)
    if not requirement_result:
        return None

    completed_list = ", ".join(requirement_result["completed"])
    satisfied_parts = []
    for requirement, courses in requirement_result["satisfied"].items():
        if requirement == "programming":
            satisfied_parts.append(f"the programming prerequisite via {', '.join(courses)}")
        elif requirement == "statistics":
            satisfied_parts.append(f"the statistics prerequisite via {', '.join(courses)}")
        else:
            satisfied_parts.append(", ".join(courses))

    satisfied_text = (
        ", ".join(satisfied_parts)
        if satisfied_parts
        else "none of the listed prerequisite categories"
    )
    missing_text = "; ".join(requirement_result["missing"])
    primary_source = source_ref(chunks, 0) or "[Official S1]"
    secondary_source = source_ref(chunks, 1)
    refs = primary_source + secondary_source

    return (
        f"Based on the courses you listed ({completed_list}), you have satisfied "
        f"{satisfied_text}. You still need: {missing_text} {refs}.\n\n"
        "Pathway note: for current UW students, the retrieved Informatics source says all four "
        "prerequisite courses must be completed before applying. Transfer applicants are treated "
        "differently for INFO 200; the source says they are not required to complete INFO 200 "
        f"before applying and complete it after becoming Informatics students {refs}."
    )


def retrieval_coverage_error(question: str, chunks: list[dict[str, Any]]) -> str | None:
    majors = infer_query_majors(question)
    intent = infer_query_intent(question)

    if intent == "double_major_policy" and len(majors) >= 2:
        missing_specific_policy = True
        for chunk in chunks:
            chunk_text = " ".join(
                [
                    str(chunk.get("heading", "")),
                    str(chunk.get("title", "")),
                    str(chunk.get("text", "")),
                ]
            ).lower()
            if all(
                any(alias in chunk_text for alias in MAJOR_ALIASES.get(major, (major.lower(),)))
                for major in majors
            ):
                missing_specific_policy = False
                break

        if missing_specific_policy:
            reviewed_sources = "[Official S1]"
            if len(chunks) > 1:
                reviewed_sources += "[Official S2]"
            major_names = " and ".join(majors)
            return (
                "The retrieved official sources include general double-major/double-degree policy "
                f"and related program policy, but they do not explicitly confirm the {major_names} "
                f"combination {reviewed_sources}. I cannot confirm that specific combination from "
                "the retrieved official sources, so you should confirm with the relevant advising offices."
            )

    if intent != "major_admissions" or len(majors) < 2:
        return None

    missing = check_topic_coverage(chunks, majors, "major_admissions")
    if not missing:
        return None

    missing_list = ", ".join(missing)
    return (
        "I could not find complete admissions-prerequisite context for all requested majors. "
        f"Missing major_admissions context for: {missing_list}."
    )


def needs_major_clarification(question: str) -> bool:
    intent = infer_query_intent(question)
    majors = infer_query_majors(question)
    return intent in {"major_admissions", "course_eligibility"} and not majors


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

    if not major and needs_major_clarification(question):
        return "Which major are you applying to?", []

    require_api_key()

    client = OpenAI(default_headers={"Accept-Encoding": "identity"})
    retriever = Retriever(client=client)
    chunks = retriever.retrieve(question, top_k=top_k, major=major)
    if not chunks:
        return "I could not find relevant retrieved context for this question.", []
    coverage_error = retrieval_coverage_error(question, chunks)
    if coverage_error:
        return coverage_error, chunks
    deterministic_answer = answer_informatics_missing_prerequisites(question, chunks)
    if deterministic_answer:
        return deterministic_answer, chunks

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
