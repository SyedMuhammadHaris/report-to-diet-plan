import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
google_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
print(f"Google Gemini API Key: {google_api_key}")


def call_gemini_model():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key)
    # response = llm.invoke("Write a short poem about the beauty of nature.")
    response = llm.invoke([
        "system: You are a helpful assistant. give in answer one sentence.",
        "user: Can you provide a tagline for my healthy snack brand?",
    ])
    print(response.text)
