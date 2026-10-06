"""One command builds both indexes: python -m docsage.ingest.build_index"""
import json
import time

import numpy as np

from docsage.config import settings
from docsage.ingest.chunking import CHUNKERS
from docsage.ingest.pdf_loader import load_pdf
from docsage.retrieval.embedder import embed_passages


def build(strategies: list[str] | None = None) -> None:
    pdfs = sorted(settings.raw_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {settings.raw_dir}; run scripts/fetch_reports.py")
    pages = []
    for pdf in pdfs:
        n = len(pages)
        pages.extend(load_pdf(pdf))
        print(f"parsed {pdf.name}: {len(pages) - n} pages")

    for name in strategies or list(CHUNKERS):
        t0 = time.time()
        chunks = CHUNKERS[name](pages, settings.chunk_tokens, settings.chunk_overlap)
        print(f"[{name}] {len(chunks)} chunks; embedding...")
        emb = embed_passages([c.text for c in chunks])
        out = settings.index_dir / name
        out.mkdir(parents=True, exist_ok=True)
        with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(out / "embeddings.npy", emb)
        print(f"[{name}] saved to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    build()
