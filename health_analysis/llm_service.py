"""LLM call layer for the health analysis app.

This module only defines the interface the UI depends on. Implement
generate_diet_plan() with your own LLM call (see gemini_call.py at the
project root for a reference pattern using langchain_google_genai).
"""

import io
import os
import re
import traceback
from groq import Groq
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader

load_dotenv()

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PARSER_MODEL = "qwen/qwen3.6-27b"
_DIET_MODEL = "gemini-2.5-flash"
_google_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks some models include in content."""
    return _THINK_TAG_RE.sub("", text).strip()


def _extract_report_text(report_file) -> str:
    """Read the uploaded report as text, extracting text from PDFs properly."""
    filename = (getattr(report_file, "name", "") or "").lower()
    content_type = getattr(report_file, "type", "") or ""
    raw_bytes = report_file.read()

    if filename.endswith(".pdf") or content_type == "application/pdf":
        reader = PdfReader(io.BytesIO(raw_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return raw_bytes.decode("utf-8", errors="ignore")


def generate_diet_plan(report_file, profile: dict) -> str:
    """Analyze an uploaded health report and return a diet plan.

    Args:
        report_file: Streamlit UploadedFile from st.file_uploader
            (has .name, .type, .getvalue()/.read()).
        profile: dict with keys age, gender, weight_kg, height_cm,
            activity_level, dietary_restrictions, allergies, goal.

    Returns:
        Markdown-formatted diet plan text.
    """

    try:
        report = _extract_report_text(report_file)
        print("report")
        print(report)
        prompt = f"""
        You are an expert medical document parser.

Extract all structured information from the medical report.

Instructions:
1. Identify report type.
2. Extract patient demographics if available.
3. Extract laboratory tests with:
   - test_name
   - value
   - unit
   - reference_range
   - flag (High/Low/Normal if present)
4. Extract doctor's impression or diagnosis.
5. Extract recommendations.
6. If a field is missing, return null.
7. Return ONLY valid JSON.
{report}
"""
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        chat = groq_client.chat.completions.create(
            model=_PARSER_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            reasoning_format="hidden",
            # max_completion_tokens=8192,
        )
        extracted_info_json = _strip_thinking(chat.choices[0].message.content)
        print(extracted_info_json)

        prompt_diet = f"""
You are a nutritionist and dietitian. Based on the following extracted information from a health report and the user's profile, create a personalized diet plan. The diet plan should be in markdown format and include meal suggestions, portion sizes, and any relevant dietary advice. Ensure that the diet plan aligns with the user's health goals, dietary restrictions, and allergies.

Keep the plan meaningful and actionable but well-balanced in length: not a brief summary, and not an exhaustive multi-day breakdown. Cover a full day's meals (breakfast, lunch, dinner, one or two snacks) with portion sizes, plus a short section of key dietary advice.

Extracted Information: {extracted_info_json}
User Profile: {profile}
"""
        gemini_llm = ChatGoogleGenerativeAI(model=_DIET_MODEL, api_key=_google_api_key)
        diet_response = gemini_llm.invoke([
            "system: You are a helpful assistant.",
            f"user: {prompt_diet}",
        ])
        return _strip_thinking(diet_response.text)
    except Exception as e:
        print(f"Error generating diet plan: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Failed to generate diet plan: {e}") from e
