# AI Agent (Streamlit + Groq)

Loop: Understand Task -> Decide Action -> Use Tool -> Process Result -> Respond

- LLM: Groq `openai/gpt-oss-120b` (native tool/function calling)
- Tools: calculator, current time, weather (Open-Meteo), Wikipedia, notes database (SQLite)
- Memory: short-term (chat history) + long-term (SQLite notes)
- Safety: tool allowlist, destructive-action guard, max-step limit, safe math parser,
  tool errors returned to the model, timeouts and output truncation

## Run
1. `python -m venv .venv` and activate it
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your `GROQ_API_KEY`
4. `streamlit run app.py`

## Structure
- `agent/tools.py` tool functions, JSON schemas, safe executor
- `agent/core.py`  the agent loop (tool-calling, step limit, retries)
- `app.py`         Streamlit UI with a visible step-by-step trace

## Limitations
Agents can pick wrong tools, loop, or act on bad data. Keep destructive tools gated,
verify important results, and never give an agent unrestricted code/shell access.
