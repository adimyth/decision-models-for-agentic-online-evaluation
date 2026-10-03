"""A small but real web-research agent built with Deep Agents."""
import os
import httpx
import tiktoken
import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gpt-5.6-luna")
MAX_PAGE_TOKENS = 4000
_enc = tiktoken.get_encoding("o200k_base")

SYSTEM_PROMPT = """You are a research assistant. Answer the user's factual question.

Rules:
- Use web_search to find candidate sources, then fetch_page to read the most relevant one before answering.
- Give a short, direct answer (one or two sentences) followed by the URL(s) you relied on.
- If after searching you cannot find reliable evidence, say so plainly instead of guessing.
- Do not use the file system or todo tools; just search, read and answer."""


@tool
def web_search(query: str) -> str:
    """Search the web. Returns up to 6 results with title, URL and snippet."""
    try:
        results = DDGS().text(query, max_results=6)
    except Exception as e:  # noqa: BLE001
        return f"search error: {e}"
    if not results:
        return "no results"
    return "\n\n".join(f"{r.get('title','')}\n{r.get('href','')}\n{r.get('body','')}" for r in results)


@tool
def fetch_page(url: str) -> str:
    """Fetch a web page and return its main text (truncated)."""
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0 (research-agent)"})
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"fetch error: {e}"
    text = trafilatura.extract(resp.text, include_links=False, include_comments=False) or ""
    if not text.strip():
        return "fetch error: no extractable text"
    toks = _enc.encode(text)
    if len(toks) > MAX_PAGE_TOKENS:
        text = _enc.decode(toks[:MAX_PAGE_TOKENS]) + "\n[truncated]"
    return text


def build_agent():
    model = ChatOpenAI(model=AGENT_MODEL, timeout=60, max_retries=3, use_responses_api=True)
    return create_deep_agent(model=model, tools=[web_search, fetch_page], system_prompt=SYSTEM_PROMPT)
