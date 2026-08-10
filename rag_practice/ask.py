"""Interactive RAG Q&A over the ingested PDF.

Retrieves the most relevant chunks from Chroma and asks Gemini to answer
using only that context. Run rag_practice/build_index.py first.

    uv run python rag_practice/ask.py
"""

# os.getenv lets us read the Gemini API key from the environment (loaded from .env).
import os

# PersistentClient opens the same on-disk Chroma database that build_index.py created.
from chromadb import PersistentClient

# Must match the embedding function used in build_index.py, so questions are
# embedded with the exact same model as the stored document chunks.
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# load_dotenv reads key=value pairs from the project's .env file into the
# environment, so os.getenv("GOOGLE_GEMINI_API_KEY") below can find it.
from dotenv import load_dotenv

# LangChain's wrapper for calling Google Gemini chat models.
from langchain_google_genai import ChatGoogleGenerativeAI

# Shared constants from config.py, kept in sync with build_index.py.
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL_NAME,
    TOP_K,
)

# Populate environment variables (like GOOGLE_GEMINI_API_KEY) from .env.
load_dotenv()

# Instructions given to the model on every call so it stays grounded in the
# retrieved context instead of making things up.
_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a research document. "
    "Answer only using the provided context. If the context does not contain the "
    "answer, say you don't know instead of guessing. Cite page numbers when relevant."
)


def get_collection():
    # Open the same persisted database directory build_index.py wrote to.
    client = PersistentClient(path=str(CHROMA_DIR))
    # Recreate the embedding function so Chroma can embed the question text
    # the same way it embedded the stored document chunks.
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    # Fetch the existing collection by name (created during build_index.py).
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)


def retrieve(collection, question: str, top_k: int = TOP_K):
    # Chroma embeds `question` with the embedding function and finds the
    # top_k stored chunks whose vectors are closest (most semantically similar).
    results = collection.query(query_texts=[question], n_results=top_k)
    # query() returns results as lists-of-lists (one list per input query);
    # since we only sent one question, we take index 0 for its own results.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    # Pair each chunk's text with its metadata (source, page number).
    return list(zip(documents, metadatas))


def build_prompt(question: str, chunks: list[tuple[str, dict]]) -> str:
    # Join the retrieved chunks into one context block, prefixing each with
    # its page number so the model can cite sources in its answer.
    context = "\n\n".join(f"[page {meta['page']}] {doc}" for doc, meta in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer_question(llm: ChatGoogleGenerativeAI, collection, question: str) -> str:
    # 1. Find the most relevant chunks for this question.
    chunks = retrieve(collection, question)
    # 2. Combine them with the question into a single prompt.
    prompt = build_prompt(question, chunks)
    # 3. Send the system instructions + prompt to Gemini and get its reply.
    response = llm.invoke([f"system: {_SYSTEM_PROMPT}", f"user: {prompt}"])
    return response.text


def main() -> None:
    # Read the Gemini API key from the environment (set via .env).
    google_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    # Create the chat model client used to generate answers.
    llm = ChatGoogleGenerativeAI(model=GENERATION_MODEL_NAME, api_key=google_api_key)
    # Connect to the persisted vector store built by build_index.py.
    collection = get_collection()

    print("RAG Q&A ready. Type a question, or 'exit' to quit.")
    # Simple REPL loop: keep asking for questions until the user quits.
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        # Ignore empty input (e.g. user just pressed Enter) instead of
        # sending a blank question to the model.
        if not question:
            continue
        print("\n" + answer_question(llm, collection, question))


# Only start the interactive loop when this file is run directly, not when imported.
if __name__ == "__main__":
    main()
