# Health Report → Diet Plan Generator



## About the Project

This is a collection of small AI-powered tools built with LangChain, Streamlit, and multiple LLM providers.

The main app is **Health Report → Diet Plan** ([health_analysis/](health_analysis/)), a Streamlit application that:

- Takes a user's profile (age, gender, weight, height, activity level, goal, dietary restrictions, allergies) from the sidebar.
- Accepts an uploaded health report (PDF, image, TXT, or CSV).
- Extracts structured medical information from the report (report type, patient demographics, lab test results, diagnosis/impression, recommendations).
- Generates a personalized, markdown-formatted diet plan based on the extracted report data and the user's profile.
- Lets the user download the generated diet plan as a PDF.

## LLM Call Flow

The app chains two separate LLM calls, each suited to a different job:

```
User (report + profile)
        │
        ▼
┌───────────────────────┐
│  LLM 1 — Parser        │   Groq · qwen/qwen3.6-27b
│  Extracts structured   │   in: raw report text (from PDF/TXT/CSV)
│  JSON from the report  │   out: report_type, patient demographics,
└──────────┬─────────────┘        lab tests, diagnosis, recommendations
           │  (extracted JSON)
           ▼
┌───────────────────────┐
│  LLM 2 — Dietitian      │   Google Gemini · gemini-2.5-flash
│  Turns extracted data  │   in: extracted JSON + user profile
│  + profile into a plan │   out: markdown diet plan (meals, portions, advice)
└──────────┬─────────────┘
           │
           ▼
   Diet plan shown in UI
   + downloadable as PDF
```

1. **User** uploads a health report and fills in their profile (age, weight, goal, allergies, etc.) in the Streamlit sidebar.
2. **LLM 1 (parser)** reads the raw report text and returns structured JSON — this keeps report parsing decoupled from diet advice, so it can be tuned/replaced independently.
3. **LLM 2 (dietitian)** receives that structured JSON plus the user's profile and produces the actual diet plan in markdown.
4. The **UI** renders the plan and offers a PDF download of it.

## Tools & Tech Stack

- **[Streamlit](https://streamlit.io/)** — the UI layer for the app (sidebar inputs, file upload, results display, PDF download).
- **[LangChain](https://www.langchain.com/)** — orchestration layer for LLM calls.
- **[langchain-google-genai](https://pypi.org/project/langchain-google-genai/)** + **Google Gemini (`gemini-2.5-flash`)** — generates the final diet plan from the extracted report data and user profile.
- **[Groq](https://groq.com/)** (`qwen/qwen3.6-27b`) — parses the uploaded health report and extracts structured medical information as JSON.
- **[pypdf](https://pypi.org/project/pypdf/)** — extracts text from uploaded PDF health reports.
- **[fpdf2](https://pypi.org/project/fpdf2/)** — generates the downloadable diet plan PDF.
- **[python-dotenv](https://pypi.org/project/python-dotenv/)** — loads API keys (`GOOGLE_GEMINI_API_KEY`, `GROQ_API_KEY`) from a local `.env` file.

## Project Structure

```
.
├── health_analysis/       # Health Report → Diet Plan Streamlit app
│   ├── app.py             # UI layer
│   └── llm_service.py     # LLM call layer (report parsing + diet plan generation)
├── main.py                # Simple Gemini call example
├── gemini_call.py         # Gemini call helper used by main.py
└── pyproject.toml         # Project dependencies
```

## Setup

1. Install dependencies:
   ```
   uv sync
   ```
2. Create a `.env` file in the project root with:
   ```
   GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
   ```
3. Run the health analysis app:
   ```
   uv run streamlit run health_analysis/app.py
   ```


## Demo

▶️ <a href="https://www.loom.com/share/bffc287038f6464ea36833f5a8c77cff" target="_blank" rel="noopener noreferrer">Watch the demo video</a>