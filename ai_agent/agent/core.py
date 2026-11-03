"""The agent loop:  Understand Task -> Decide Action -> Use Tool -> Process Result -> Respond.

The LLM decides (via tool calling) whether to answer or call a tool. We run the tool,
feed the result back, and repeat until the model answers or max_steps is reached.
"""
import json
import time

from groq import APIConnectionError, RateLimitError

from .tools import TOOL_SPECS, execute

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a helpful AI agent with tools.
1. Understand the user's task and break it into steps if needed.
2. Use tools when they give a more reliable answer (math, weather, time, facts, notes). Do not guess.
3. You may call several tools in sequence; use earlier results to decide the next call.
4. When you have enough information, reply clearly and concisely.
If a tool returns an error or is blocked, explain that honestly instead of making things up."""


def _chat(client, messages, tools, temperature, retries=3):
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools or None,
                tool_choice="auto" if tools else None,
                temperature=temperature, max_tokens=2048,
            )
        except (RateLimitError, APIConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def run_agent(client, history, user_input, enabled, allow_destructive=False,
              max_steps=6, temperature=0.2):
    """Generator yielding events: tool_call, tool_result, final.

    `history` is a list of previous {"role","content"} turns (short-term memory).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-10:]
    messages.append({"role": "user", "content": user_input})
    tools = [TOOL_SPECS[n] for n in TOOL_SPECS if n in enabled]
    usage_in = usage_out = 0

    for step in range(1, max_steps + 1):
        resp = _chat(client, messages, tools, temperature)
        usage_in += resp.usage.prompt_tokens
        usage_out += resp.usage.completion_tokens
        msg = resp.choices[0].message

        if not msg.tool_calls:  # Respond
            yield {"type": "final", "content": msg.content or "",
                   "steps": step, "tokens_in": usage_in, "tokens_out": usage_out}
            return

        # Decide Action: record the assistant's tool request
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })
        # Use Tool + Process Result
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            yield {"type": "tool_call", "name": tc.function.name, "args": args}
            result = execute(tc.function.name, args, enabled, allow_destructive)
            yield {"type": "tool_result", "name": tc.function.name, "result": result}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Safety stop: too many steps
    yield {"type": "final", "steps": max_steps, "tokens_in": usage_in, "tokens_out": usage_out,
           "content": f"⚠️ I stopped after {max_steps} steps to stay safe. "
                      "Try a more specific request or raise the step limit."}
