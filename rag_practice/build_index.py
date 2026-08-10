"""Ingest the PDF, chunk it, embed it, and persist it into a Chroma collection.

Run this once (or whenever the source PDF changes):
    uv run python rag_practice/build_index.py
"""

# PersistentClient opens (or creates) a Chroma database that saves to disk,
# so the embeddings survive between script runs instead of living only in memory.
from chromadb import PersistentClient

# This wraps a sentence-transformers model so Chroma can call it automatically
# to turn text into embedding vectors whenever we add or query documents.
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# pypdf reads PDF files and lets us pull the raw text out of each page.
from pypdf import PdfReader

# Shared constants (file paths, model name, chunk sizing) live in config.py
# so build_index.py and ask.py always agree on them.
from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PDF_PATH,
)


def load_pages(pdf_path) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) for every non-empty page."""
    # Open the PDF and prepare to read it page by page.
    reader = PdfReader(str(pdf_path))
    pages = []
    # enumerate(..., start=1) numbers pages from 1 instead of 0, matching how
    # a human would refer to "page 1" of the document.
    for i, page in enumerate(reader.pages, start=1):
        # extract_text() can return None for image-only pages, so fall back
        # to "" before calling .strip() to avoid a crash.
        text = (page.extract_text() or "").strip()
        # Skip pages that had no extractable text (e.g. blank or scanned pages).
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows, breaking on whitespace."""
    chunks = []
    start = 0
    text_len = len(text)
    # Slide a window across the text until we've consumed all of it.
    while start < text_len:
        # The naive end of this window is chunk_size characters past start,
        # capped so we never read past the end of the text.
        end = min(start + chunk_size, text_len)
        # If this isn't the final window, try to end on a space instead of
        # mid-word, so chunks read as whole words rather than being cut off.
        if end < text_len:
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Once this window reaches the end of the text, there's nothing left to chunk.
        if end >= text_len:
            break
        # Move the window forward, but step back by `overlap` characters first
        # so consecutive chunks share some context (helps retrieval quality
        # for answers that span a chunk boundary).
        start = end - overlap
    return chunks


def build_index() -> None:
    # Step 1: extract text from every page of the PDF.
    pages = load_pages(PDF_PATH)
    print(f"Loaded {len(pages)} non-empty pages from {PDF_PATH.name}")

    # Step 2: break each page's text into smaller overlapping chunks, and
    # build three parallel lists that Chroma's add() call expects:
    # the chunk text itself, metadata describing where it came from, and a
    # unique id for each chunk.
    documents, metadatas, ids = [], [], []
    for page_number, page_text in pages:
        for chunk_index, chunk in enumerate(chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)):
            documents.append(chunk)
            metadatas.append({"source": PDF_PATH.name, "page": page_number})
            ids.append(f"page{page_number}-chunk{chunk_index}")
    print(f"Split into {len(documents)} chunks")

    # Step 3: connect to (or create) the on-disk Chroma database, and set up
    # the embedding function that will convert each chunk of text into a
    # vector using the all-MiniLM-L6-v2 sentence-transformers model.
    client = PersistentClient(path=str(CHROMA_DIR))
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    # Recreate the collection so re-running this script reflects the latest PDF content.
    existing_collections = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_function
    )

    # Step 4: add the chunks to the collection in batches. Chroma embeds each
    # document automatically (using embedding_function) as it's added.
    # Batching avoids sending one huge request when there are many chunks.
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    # collection.count() confirms how many vectors actually made it into storage.
    print(f"Persisted {collection.count()} chunks to {CHROMA_DIR}")


# Only run build_index() when this file is executed directly
# (e.g. `uv run python rag_practice/build_index.py`), not when imported.
if __name__ == "__main__":
    build_index()
