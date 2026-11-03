"""
AI Agent (Streamlit + Groq openai/gpt-oss-120b)
Understand Task -> Decide Action -> Use Tool -> Process Result -> Respond
"""
import os

import streamlit as st
from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, AuthenticationError, Groq, RateLimitError

from agent.core import MODEL, run_agent
from agent.tools import TOOL_SPECS

load_dotenv()
st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="centered")

st.session_state.setdefault("messages", [])  # {"role","content","trace"?}
st.session_state.setdefault("tokens", 0)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Agent settings")
    if not os.getenv("GROQ_API_KEY"):
        st.text_input("Groq API key", type="password", key="manual_key")
    st.caption(f"Model: `{MODEL}`")

    enabled = set(st.multiselect("Enabled tools", list(TOOL_SPECS), default=list(TOOL_SPECS)))
    allow_destructive = st.toggle("Allow destructive actions (delete_note)", value=False,
                                  help="Safety guard: off by default.")
    max_steps = st.slider("Max steps per task", 1, 10, 6)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    st.metric("Tokens used (session)", st.session_state.tokens)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages, st.session_state.tokens = [], 0
        st.rerun()
    st.caption("Try: *What's 18% of 2450, and what's the weather in Pune?*  \n"
               "or: *Save a note that my exam is on Friday*")


def render_trace(trace):
    for ev in trace:
        if ev["type"] == "tool_call":
            st.markdown(f"🔧 **Call** `{ev['name']}` `{ev['args']}`")
        else:
            st.markdown(f"📥 **Result:** {ev['result']}")


# ---------- Main ----------
st.title("🤖 AI Agent")
st.caption("Understand → Decide → Use Tool → Process Result → Respond")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("trace"):
            with st.expander(f"🧠 Agent steps ({len(m['trace']) // 2} tool calls)"):
                render_trace(m["trace"])
        st.markdown(m["content"])

if task := st.chat_input("Give the agent a task..."):
    key = os.getenv("GROQ_API_KEY") or st.session_state.get("manual_key")
    if not key:
        st.error("Add GROQ_API_KEY to .env or enter it in the sidebar.")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": task})
    with st.chat_message("user"):
        st.markdown(task)

    with st.chat_message("assistant"):
        trace, final = [], None
        try:
            with st.status("Agent working...", expanded=True) as status:
                for ev in run_agent(Groq(api_key=key), history, task, enabled,
                                    allow_destructive, max_steps, temperature):
                    if ev["type"] == "final":
                        final = ev
                    else:
                        trace.append(ev)
                        render_trace([ev])
                status.update(label=f"Done in {final['steps']} step(s)", state="complete", expanded=False)
            st.markdown(final["content"])
            st.session_state.tokens += final["tokens_in"] + final["tokens_out"]
            st.session_state.messages.append(
                {"role": "assistant", "content": final["content"], "trace": trace})
            st.rerun()
        except AuthenticationError:
            st.error("🔑 Invalid API key.")
        except RateLimitError:
            st.error("⏳ Rate limit reached. Try again shortly.")
        except APIConnectionError:
            st.error("🌐 Network problem reaching Groq.")
        except APIStatusError as e:
            st.error(f"API error {e.status_code}: {e.message}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
