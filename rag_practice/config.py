# Path gives us cross-platform file paths (works the same on Windows/Mac/Linux).
from pathlib import Path

# Folder this config.py file lives in — used as the anchor for all other
# paths below, so the scripts work no matter what directory you run them from.
BASE_DIR = Path(__file__).parent

# The source document we're building the RAG pipeline over.
PDF_PATH = BASE_DIR / "lung_cancer_reserch.pdf"

# Folder where Chroma persists its on-disk vector database.
CHROMA_DIR = BASE_DIR / "chroma_db"

# Name of the Chroma collection (like a table name) that stores our chunks.
COLLECTION_NAME = "lung_cancer_research"

# Sentence-transformers model used to turn text into embedding vectors.
# Small and fast, good for local/tutorial use.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini model used to generate the final answer from retrieved context.
GENERATION_MODEL_NAME = "gemini-2.5-flash"

# Max characters per chunk when splitting page text (see chunk_text in build_index.py).
CHUNK_SIZE = 800

# How many characters consecutive chunks overlap by, so context isn't lost
# when a relevant sentence happens to fall on a chunk boundary.
CHUNK_OVERLAP = 150

# How many chunks to retrieve from Chroma per question in ask.py.
TOP_K = 4
