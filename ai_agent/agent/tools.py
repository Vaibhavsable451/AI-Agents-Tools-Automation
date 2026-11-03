"""Tools the agent can call + their JSON schemas + a safe executor.

Safety design:
- Only tools in the registry (and enabled in the UI) can run.
- Destructive tools (delete_note) are blocked unless the user allows them.
- Calculator uses an AST whitelist, never eval().
- Network calls have timeouts; outputs are truncated.
"""
import ast
import json
import operator as op
import os
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

DB_PATH = os.getenv("AGENT_DB", "agent_data.db")
MAX_OUTPUT_CHARS = 3000


# ---------------- helpers ----------------
def _http_json(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-agent-demo/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS notes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, "
        "created TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    return conn


# ---------------- tools ----------------
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    result = _eval(ast.parse(expression, mode="eval").body)
    return str(result)


def get_current_time(timezone: str = "UTC") -> str:
    now = datetime.now(ZoneInfo(timezone))
    return now.strftime(f"%A, %d %B %Y, %H:%M:%S ({timezone})")


def get_weather(city: str) -> str:
    geo = _http_json(
        "https://geocoding-api.open-meteo.com/v1/search?"
        + urllib.parse.urlencode({"name": city, "count": 1})
    )
    if not geo.get("results"):
        return f"City not found: {city}"
    g = geo["results"][0]
    w = _http_json(
        "https://api.open-meteo.com/v1/forecast?"
        + urllib.parse.urlencode({
            "latitude": g["latitude"], "longitude": g["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        })
    )["current"]
    return (f"{g['name']}, {g.get('country', '')}: {w['temperature_2m']}°C, "
            f"humidity {w['relative_humidity_2m']}%, wind {w['wind_speed_10m']} km/h, "
            f"precipitation {w['precipitation']} mm")


def wikipedia_search(query: str) -> str:
    data = _http_json(
        "https://en.wikipedia.org/w/api.php?"
        + urllib.parse.urlencode({
            "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": 1,
            "prop": "extracts", "exintro": 1, "explaintext": 1, "format": "json",
        })
    )
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return "No Wikipedia result found."
    page = next(iter(pages.values()))
    return f"{page['title']}: {page.get('extract', '')[:1500]}"


def save_note(text: str) -> str:
    with _db() as c:
        cur = c.execute("INSERT INTO notes(text) VALUES (?)", (text,))
        return f"Saved note #{cur.lastrowid}"


def list_notes() -> str:
    with _db() as c:
        rows = c.execute("SELECT id, text, created FROM notes ORDER BY id").fetchall()
    return "\n".join(f"#{i} [{t}] {x}" for i, x, t in rows) or "No notes saved yet."


def search_notes(keyword: str) -> str:
    with _db() as c:
        rows = c.execute(
            "SELECT id, text FROM notes WHERE text LIKE ?", (f"%{keyword}%",)
        ).fetchall()
    return "\n".join(f"#{i} {x}" for i, x in rows) or "No matching notes."


def delete_note(note_id: int) -> str:
    with _db() as c:
        cur = c.execute("DELETE FROM notes WHERE id = ?", (int(note_id),))
        return f"Deleted note #{note_id}" if cur.rowcount else f"Note #{note_id} not found"


# ---------------- registry + schemas ----------------
def _spec(name, desc, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {}, "required": required or []},
    }}


TOOL_SPECS = {
    "calculator": _spec("calculator", "Evaluate a math expression, e.g. '(12.5*8)/3'. Use for ALL arithmetic.",
                        {"expression": {"type": "string"}}, ["expression"]),
    "get_current_time": _spec("get_current_time", "Get current date and time in an IANA timezone like 'Asia/Kolkata'.",
                              {"timezone": {"type": "string", "description": "IANA timezone, default UTC"}}),
    "get_weather": _spec("get_weather", "Get current weather for a city.",
                         {"city": {"type": "string"}}, ["city"]),
    "wikipedia_search": _spec("wikipedia_search", "Look up factual background info on Wikipedia.",
                              {"query": {"type": "string"}}, ["query"]),
    "save_note": _spec("save_note", "Save a note/fact to the user's persistent notes database.",
                       {"text": {"type": "string"}}, ["text"]),
    "list_notes": _spec("list_notes", "List all saved notes."),
    "search_notes": _spec("search_notes", "Search saved notes by keyword.",
                          {"keyword": {"type": "string"}}, ["keyword"]),
    "delete_note": _spec("delete_note", "Delete a saved note by id. DESTRUCTIVE.",
                         {"note_id": {"type": "integer"}}, ["note_id"]),
}

REGISTRY = {  # name -> (function, is_destructive)
    "calculator": (calculator, False),
    "get_current_time": (get_current_time, False),
    "get_weather": (get_weather, False),
    "wikipedia_search": (wikipedia_search, False),
    "save_note": (save_note, False),
    "list_notes": (list_notes, False),
    "search_notes": (search_notes, False),
    "delete_note": (delete_note, True),
}


def execute(name: str, args: dict, enabled: set[str], allow_destructive: bool) -> str:
    """Run a tool safely. Always returns a string (errors included) so the LLM can recover."""
    if name not in REGISTRY or name not in enabled:
        return f"ERROR: tool '{name}' is not available."
    fn, destructive = REGISTRY[name]
    if destructive and not allow_destructive:
        return "BLOCKED: this action is destructive and the user has not enabled it."
    try:
        out = str(fn(**args))
    except TypeError as e:
        return f"ERROR: bad arguments for {name}: {e}"
    except Exception as e:
        return f"ERROR running {name}: {type(e).__name__}: {e}"
    return out[:MAX_OUTPUT_CHARS]
