from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.chain import answer_question


QUESTIONS = [
    "What GPA do I need for INFO admission?",
    "What prerequisite courses are required before applying to INFO?",
    "When are the INFO application deadlines?",
    "What are the CSE undergraduate admission pathways?",
    "What CSE degree requirements should I know about?",
    "What are the ACMS admission prerequisites?",
    "What are the HCDE application requirements?",
    "What does the context say about HCDE degree workload and total credits?",
    "What courses are required for the Statistics major?",
    "How competitive is AMATH admission and what GPA is mentioned?",
]


def main() -> None:
    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\n=== Question {index}: {question} ===")
        answer, chunks = answer_question(question)
        print(answer)
        print("Top sources:")
        for source_index, chunk in enumerate(chunks[:3], start=1):
            print(
                f"- S{source_index}: [{chunk['source_label']}] "
                f"{chunk.get('major', '')} | {chunk.get('heading', '')} | {chunk.get('url', '')}"
            )


if __name__ == "__main__":
    main()
