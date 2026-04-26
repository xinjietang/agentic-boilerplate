"""Tools for the daily news curator agent.

These tools use in-process mock data so the example runs without any
external API key.  To adapt them for production, replace the stub
implementations with real HTTP calls to a news API such as NewsAPI.org,
GNews, or an RSS feed reader.
"""

from __future__ import annotations

import json
from datetime import date

from agentic_boilerplate.tools.base import ToolResult, ToolSpec
from agentic_boilerplate.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Mock headline database (keyed by topic)
# ---------------------------------------------------------------------------

_TODAY = date.today().isoformat()

_MOCK_HEADLINES: dict[str, list[dict]] = {
    "technology": [
        {
            "title": "AI models reach new reasoning benchmarks",
            "source": "TechCrunch",
            "url": "https://techcrunch.com/example/ai-benchmarks",
            "summary": "Researchers report that the latest language models now exceed human-level performance on several standardised reasoning tests.",
            "published": _TODAY,
        },
        {
            "title": "Open-source LLM ecosystem grows rapidly",
            "source": "The Verge",
            "url": "https://theverge.com/example/open-source-llm",
            "summary": "The number of open-source LLM releases on Hugging Face has doubled year-over-year, driven by community contributions.",
            "published": _TODAY,
        },
        {
            "title": "New battery tech promises 10-minute EV charging",
            "source": "Ars Technica",
            "url": "https://arstechnica.com/example/ev-battery",
            "summary": "A startup claims its solid-state battery cells can charge to 80% in under 10 minutes at room temperature.",
            "published": _TODAY,
        },
    ],
    "finance": [
        {
            "title": "Central banks signal cautious rate path",
            "source": "Financial Times",
            "url": "https://ft.com/example/central-banks",
            "summary": "Fed and ECB officials indicated they will hold rates steady while monitoring inflation data over the next quarter.",
            "published": _TODAY,
        },
        {
            "title": "S&P 500 closes at record high",
            "source": "Bloomberg",
            "url": "https://bloomberg.com/example/sp500",
            "summary": "Strong earnings from the technology sector pushed the S&P 500 to an all-time closing high.",
            "published": _TODAY,
        },
    ],
    "health": [
        {
            "title": "Study links sleep quality to cognitive decline risk",
            "source": "Nature Medicine",
            "url": "https://nature.com/example/sleep-study",
            "summary": "A 20-year longitudinal study found that adults averaging fewer than 6 hours of sleep show higher rates of early-onset cognitive decline.",
            "published": _TODAY,
        },
        {
            "title": "GLP-1 drugs show promise beyond weight loss",
            "source": "NEJM",
            "url": "https://nejm.org/example/glp1",
            "summary": "New clinical trial data suggests GLP-1 receptor agonists may reduce cardiovascular events independent of weight loss.",
            "published": _TODAY,
        },
    ],
    "science": [
        {
            "title": "James Webb telescope images earliest galaxy ever observed",
            "source": "NASA",
            "url": "https://nasa.gov/example/jwst-galaxy",
            "summary": "The JWST has captured light from a galaxy formed just 300 million years after the Big Bang, setting a new distance record.",
            "published": _TODAY,
        },
        {
            "title": "CRISPR gene editing enters clinical trials for sickle cell",
            "source": "Science",
            "url": "https://science.org/example/crispr-sickle-cell",
            "summary": "Phase 3 trials of a CRISPR-based therapy for sickle cell disease report 95% of patients achieving transfusion independence.",
            "published": _TODAY,
        },
    ],
}

_ALL_TOPICS = sorted(_MOCK_HEADLINES.keys())

# ---------------------------------------------------------------------------
# Tool: fetch_headlines
# ---------------------------------------------------------------------------

FETCH_HEADLINES_SPEC = ToolSpec(
    name="fetch_headlines",
    description=(
        "Fetch the latest news headlines for a given topic. "
        f"Available topics: {', '.join(_ALL_TOPICS)}. "
        "Returns a JSON array of headline objects with title, source, url, summary, and published fields."
    ),
    parameters={
        "topic": {
            "type": "string",
            "description": f"News topic to fetch. One of: {', '.join(_ALL_TOPICS)}.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of headlines to return (default: 5).",
        },
    },
    requires_approval=False,
    tags=["news", "fetch"],
)


def fetch_headlines(topic: str, limit: int = 5) -> ToolResult:
    """Return mock headlines for the requested topic."""
    key = topic.lower().strip()
    articles = _MOCK_HEADLINES.get(key)
    if articles is None:
        available = ", ".join(_ALL_TOPICS)
        return ToolResult.fail(
            f"Unknown topic '{topic}'. Available topics: {available}."
        )
    sliced = articles[:limit]
    return ToolResult.ok(json.dumps(sliced, indent=2))


# ---------------------------------------------------------------------------
# Tool: filter_by_keyword
# ---------------------------------------------------------------------------

FILTER_BY_KEYWORD_SPEC = ToolSpec(
    name="filter_by_keyword",
    description=(
        "Filter a JSON array of headline objects by a keyword. "
        "Pass the raw JSON string from fetch_headlines as the 'headlines_json' argument."
    ),
    parameters={
        "headlines_json": {
            "type": "string",
            "description": "JSON array of headline objects as returned by fetch_headlines.",
        },
        "keyword": {
            "type": "string",
            "description": "Case-insensitive keyword to search for in title and summary.",
        },
    },
    requires_approval=False,
    tags=["news", "filter"],
)


def filter_by_keyword(headlines_json: str, keyword: str) -> ToolResult:
    """Filter a headlines JSON array to entries matching *keyword*."""
    try:
        items: list[dict] = json.loads(headlines_json)
    except json.JSONDecodeError as exc:
        return ToolResult.fail(f"Invalid JSON: {exc}")

    kw = keyword.lower()
    filtered = [
        item
        for item in items
        if kw in item.get("title", "").lower()
        or kw in item.get("summary", "").lower()
    ]
    if not filtered:
        return ToolResult.ok(f"No headlines matched keyword '{keyword}'.")
    return ToolResult.ok(json.dumps(filtered, indent=2))


# ---------------------------------------------------------------------------
# Tool: format_digest
# ---------------------------------------------------------------------------

FORMAT_DIGEST_SPEC = ToolSpec(
    name="format_digest",
    description=(
        "Format a JSON array of headline objects into a human-readable daily digest. "
        "Pass the raw JSON string from fetch_headlines or filter_by_keyword."
    ),
    parameters={
        "headlines_json": {
            "type": "string",
            "description": "JSON array of headline objects.",
        },
        "title": {
            "type": "string",
            "description": "Optional title for the digest section (e.g. 'Tech News').",
        },
    },
    requires_approval=False,
    tags=["news", "format"],
)


def format_digest(headlines_json: str, title: str = "Daily Digest") -> ToolResult:
    """Render *headlines_json* as a plain-text digest."""
    try:
        items: list[dict] = json.loads(headlines_json)
    except json.JSONDecodeError as exc:
        return ToolResult.fail(f"Invalid JSON: {exc}")

    if not items:
        return ToolResult.ok(f"No headlines to display for '{title}'.")

    lines = [f"=== {title} — {_TODAY} ===", ""]
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item.get('title', '(no title)')}")
        lines.append(f"   Source : {item.get('source', '?')}")
        lines.append(f"   Summary: {item.get('summary', '')}")
        lines.append(f"   URL    : {item.get('url', '')}")
        lines.append("")
    return ToolResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_news_tools(registry: ToolRegistry) -> None:
    """Register all news curator tools into *registry*."""
    registry.register(FETCH_HEADLINES_SPEC, fetch_headlines)
    registry.register(FILTER_BY_KEYWORD_SPEC, filter_by_keyword)
    registry.register(FORMAT_DIGEST_SPEC, format_digest)
