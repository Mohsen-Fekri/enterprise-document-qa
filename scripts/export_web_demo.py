"""Export the chunked sample corpus for the browser demo in docs/ (served by GitHub Pages)."""

from __future__ import annotations

import json
from pathlib import Path

from enterprise_qa.pipeline import get_pipeline

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "corpus.json"


def main() -> None:
    pipeline = get_pipeline(512)
    chunks = [
        {"title": c.title, "filename": c.filename, "classification": c.classification, "page": c.page, "text": c.text}
        for c in pipeline.corpus.chunks
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(chunks, ensure_ascii=False))
    print(f"wrote {len(chunks)} chunks to {OUTPUT}")


if __name__ == "__main__":
    main()
