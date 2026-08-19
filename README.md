# UW Major Advisor

UW Major Advisor is a small RAG-based advising assistant for University of Washington majors. It scrapes official UW admissions and requirements pages, builds a local FAISS vector index, and answers student questions with citations to retrieved sources.

The MVP currently covers:

- INFO
- CSE
- ACMS
- HCDE
- STATS
- AMATH
- MATH

## How It Works

```text
Official UW pages
  -> scrape admissions / requirements pages
  -> remove HTML noise
  -> split by h2 / h3 sections
  -> chunk sections to about 400 tokens
  -> embed chunks with text-embedding-3-small
  -> store vectors in FAISS
  -> retrieve top-k chunks for a question
  -> answer with gpt-4o-mini using only retrieved context
  -> show citations and source URLs
```

## Project Structure

```text
.
├── app.py                  # Streamlit chat UI
├── rag/
│   ├── chain.py            # RAG prompt + GPT call
│   └── retriever.py        # FAISS retrieval wrapper
├── scripts/
│   ├── scrape.py           # Scrape official UW pages
│   ├── chunk.py            # Token-based chunking
│   ├── embed.py            # Embeddings + FAISS index build
│   └── test_rag.py         # 10-question CLI regression test
├── data/
│   ├── raw/pages.json      # Generated scraped pages, ignored by git
│   └── chunks/             # Generated chunks, embeddings, FAISS index, ignored by git
├── requirements.txt
└── .env.example
```

## Setup

Use Python 3.11+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```bash
cp .env.example .env
```

Then add your API key:

```text
OPENAI_API_KEY=your_key_here
```

`.env`, `venv/`, and generated `data/` files are intentionally ignored by git.

## Build the RAG Index

Run the pipeline in order:

```bash
python scripts/scrape.py
python scripts/chunk.py
python scripts/embed.py --query "INFO admission GPA"
```

Generated files:

```text
data/raw/pages.json
data/chunks/chunks.json
data/chunks/embeddings.npy
data/chunks/index.faiss
data/chunks/metadata.json
```

The FAISS index uses normalized embeddings with `IndexFlatIP`, which approximates cosine similarity.

## Command-Line Usage

Ask one question:

```bash
python rag/chain.py "What prerequisite courses are required before applying to INFO?"
```

Run the regression questions:

```bash
python scripts/test_rag.py
```

The current test set covers GPA requirements, prerequisites, application timelines, degree requirements, workload, and competitiveness.

## Streamlit UI

Start the chat UI:

```bash
streamlit run app.py
```

The UI supports:

- Chat history with `st.session_state`
- Major filtering in the sidebar
- Source expanders under each assistant answer
- Official-source citations such as `[Official S1]`

## RAG Rules

The assistant is prompted to:

- Answer only from retrieved context
- Say when context is insufficient
- Cite sources inline
- Distinguish source types, currently `[Official]`
- Avoid guaranteeing admission
- Keep answers concise

## Current Status

Completed:

- Phase 0: project initialization
- Phase 1: scraper
- Phase 2: chunking, embeddings, FAISS
- Phase 3: RAG retriever and LLM chain
- Phase 4: Streamlit chat UI

The MVP can answer questions from official UW pages with retrievable citations.
