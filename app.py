from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag.chain import (
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    generate_answer,
    needs_major_clarification,
    retrieval_coverage_error,
)
from rag.retriever import KNOWN_MAJORS, Retriever, RetrievalError


PROJECT_ROOT = Path(__file__).resolve().parent


def init_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI:
    return OpenAI(default_headers={"Accept-Encoding": "identity"})


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    return Retriever(client=get_openai_client())


def source_title(source: dict[str, Any], index: int) -> str:
    return (
        f"[{source['source_label']} S{index}] "
        f"{source.get('major', '')} | {source.get('heading', '')}"
    )


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander("View sources"):
        seen = set()
        for index, source in enumerate(sources, start=1):
            key = (source.get("source_label"), source.get("url"), source.get("heading"))
            if key in seen:
                continue
            seen.add(key)

            st.markdown(f"**{source_title(source, index)}**")
            st.markdown(f"[Open source page]({source.get('url', '')})")
            preview = " ".join(source.get("text", "").split())[:700]
            if preview:
                st.caption(preview)
            st.divider()


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))


def selected_major() -> str | None:
    option = st.session_state.major_filter
    return None if option == "All majors" else option


def answer_user_message(prompt: str) -> tuple[str, list[dict[str, Any]]]:
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is missing. Add it to `.env` before asking questions.", []

    if not selected_major() and needs_major_clarification(prompt):
        return "Which major are you applying to?", []

    client = get_openai_client()
    retriever = get_retriever()
    chunks = retriever.retrieve(
        prompt,
        top_k=st.session_state.top_k,
        major=selected_major(),
    )
    if not chunks:
        return "I could not find relevant retrieved context for this question.", []
    coverage_error = retrieval_coverage_error(prompt, chunks)
    if coverage_error:
        return coverage_error, chunks

    answer = generate_answer(
        client=client,
        question=prompt,
        chunks=chunks,
        model=st.session_state.model_name,
    )
    return answer, chunks


def configure_sidebar() -> None:
    with st.sidebar:
        st.header("Filters")
        st.selectbox(
            "Major",
            ["All majors", *sorted(KNOWN_MAJORS)],
            key="major_filter",
        )
        st.slider("Sources to retrieve", 3, 8, DEFAULT_TOP_K, key="top_k")
        st.text_input("Model", value=DEFAULT_MODEL, key="model_name")

        st.divider()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.caption("Answers use the local FAISS index and cite retrieved UW sources.")


def main() -> None:
    init_environment()
    st.set_page_config(page_title="UW Major Advisor", page_icon="UW")

    st.title("UW Major Advisor")
    st.caption("Ask about UW major admissions and requirements. Answers cite retrieved sources.")

    configure_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        get_retriever()
    except RetrievalError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error(f"Could not initialize retriever: {error}")
        st.stop()

    render_chat_history()

    prompt = st.chat_input("Ask about admissions, prerequisites, GPA, timelines, or requirements")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and drafting answer..."):
            try:
                answer, sources = answer_user_message(prompt)
            except SystemExit:
                answer = "The OpenAI request failed. Check your API key, billing, and network access."
                sources = []
            except Exception as error:
                answer = f"Something went wrong: {error}"
                sources = []

        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    main()
