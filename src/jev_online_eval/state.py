"""Build the judge State from a LangSmith root run of the agent."""
import json
from typing import Any

MAX_RESULT_CHARS = 2500
MAX_ANSWER_CHARS = 3000


def _content(m: dict) -> str:
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
    if isinstance(c, list):
        c = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c or ""


def _type(m: dict) -> str:
    if isinstance(m, dict):
        return m.get("type") or m.get("role") or (m.get("id", [""])[-1] if isinstance(m.get("id"), list) else "")
    return getattr(m, "type", "")


def _msg_dict(m: Any) -> dict:
    # LangSmith serialises LC messages as {"id": [...,"HumanMessage"], "kwargs": {...}} or flat dicts.
    if isinstance(m, dict) and "kwargs" in m:
        d = dict(m["kwargs"])
        d["type"] = m["id"][-1].replace("Message", "").lower()
        return d
    if isinstance(m, dict):
        return m
    return {"type": m.type, "content": m.content, "tool_calls": getattr(m, "tool_calls", None),
            "name": getattr(m, "name", None)}


def state_from_messages(messages: list) -> dict:
    msgs = [_msg_dict(m) for m in messages]
    request = next((_content(m) for m in msgs if _type(m) in ("human", "user")), "")
    calls: dict[str, dict] = {}
    order: list[str] = []
    for m in msgs:
        t = _type(m)
        if t in ("ai", "assistant"):
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id") or f"call{len(order)}"
                calls[cid] = {"tool": tc.get("name"), "args": tc.get("args"), "result": ""}
                order.append(cid)
        elif t == "tool":
            cid = m.get("tool_call_id")
            res = _content(m)[:MAX_RESULT_CHARS]
            if cid in calls:
                calls[cid]["result"] = res
            else:
                cid = f"orphan{len(order)}"
                calls[cid] = {"tool": m.get("name"), "args": None, "result": res}
                order.append(cid)
    final = ""
    for m in reversed(msgs):
        if _type(m) in ("ai", "assistant") and _content(m).strip():
            final = _content(m)[:MAX_ANSWER_CHARS]
            break
    return {"request": request, "tool_results": [calls[c] for c in order], "final_answer": final}


def state_from_run(run) -> dict:
    out = run.outputs or {}
    msgs = out.get("messages") if isinstance(out, dict) else None
    if not msgs:
        inp = run.inputs or {}
        msgs = inp.get("messages", [])
    return state_from_messages(msgs)


def state_tokens(state: dict) -> int:
    import tiktoken
    return len(tiktoken.get_encoding("o200k_base").encode(json.dumps(state)))
