"""Remove out-of-scope records while retaining existing embedding vectors."""

import json
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.admissions_scope import keep_section


def main():
    data = ROOT / "data"
    raw_path = data / "raw/pages.json"
    pages = json.loads(raw_path.read_text())
    allowed = set()
    filtered_pages = []
    for page in pages:
        sections = [
            section for section in page["sections"]
            if keep_section(page["page_type"], section["heading"])
        ]
        for section in sections:
            allowed.add((page["url"], section["heading"]))
        if sections:
            filtered_pages.append({**page, "sections": sections})

    metadata_path = data / "chunks/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    index_path = data / "chunks/index.faiss"
    old_index = faiss.read_index(str(index_path))
    records = metadata["records"]
    if len(records) != old_index.ntotal:
        raise ValueError("Index and metadata counts do not match")
    positions = [i for i, record in enumerate(records) if (record["url"], record["heading"]) in allowed]
    if not positions:
        raise ValueError("Cleanup would leave an empty corpus")
    vectors = np.stack([old_index.reconstruct(i) for i in positions])
    new_index = faiss.IndexFlatIP(old_index.d) if metadata["index_metric"] == "ip" else faiss.IndexFlatL2(old_index.d)
    new_index.add(vectors)
    metadata["records"] = [records[i] for i in positions]
    metadata["vector_count"] = len(positions)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    faiss.write_index(new_index, str(index_path))
    raw_path.write_text(json.dumps(filtered_pages, ensure_ascii=False, indent=2) + "\n")
    registry_path = data / "source_registry.json"
    registry = json.loads(registry_path.read_text())
    retained_urls = {page["url"] for page in filtered_pages}
    for source in registry:
        source["current_scrape_status"] = (
            "present" if source["url"] in retained_urls else "excluded_from_admissions_scope"
        )
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    chunks_path = data / "chunks/chunks.json"
    if chunks_path.exists():
        chunks = json.loads(chunks_path.read_text())
        ids = {record["id"] for record in metadata["records"]}
        chunks_path.write_text(json.dumps([chunk for chunk in chunks if chunk["id"] in ids], ensure_ascii=False, indent=2) + "\n")
    vectors_path = data / "chunks/embeddings.npy"
    if vectors_path.exists():
        old_vectors = np.load(vectors_path)
        if len(old_vectors) == len(records):
            np.save(vectors_path, old_vectors[positions])
        else:
            raise ValueError("Stored embeddings count does not match original metadata")
    print(f"Pages {len(pages)} -> {len(filtered_pages)}; chunks {len(records)} -> {len(positions)}")


if __name__ == "__main__":
    main()
