"""Build the vector database from a folder of PDFs.

Usage: python ingest.py [--pdf-dir data/pdfs]
"""

import argparse
import sys
from pathlib import Path

from rag import PDF_DIR, build_index, load_chunks, load_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a folder of PDFs into the vector DB.")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    args = parser.parse_args()

    print(f"Reading PDFs from {args.pdf_dir}")
    chunks = load_chunks(args.pdf_dir)
    if not chunks:
        sys.exit(f"No extractable text found in PDFs under {args.pdf_dir}")

    print(f"Embedding {len(chunks)} chunks")
    build_index(chunks, load_embedder())
    print("Done. Start the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
