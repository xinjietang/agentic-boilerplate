"""Tools for the research assistant agent.

All storage is in-process.  To adapt for production, replace
the mock search with a real search API (Brave Search, Tavily,
SerpAPI, etc.) and the note store with a database or vector store.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from agentic_boilerplate.tools.base import ToolResult, ToolSpec
from agentic_boilerplate.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Mock search database
# ---------------------------------------------------------------------------

_MOCK_SEARCH_DB: list[dict] = [
    {
        "title": "Introduction to Retrieval-Augmented Generation (RAG)",
        "url": "https://arxiv.org/abs/2005.11401",
        "snippet": (
            "RAG combines dense retrieval with seq2seq generation.  A retriever "
            "finds relevant documents; a generator conditions its output on them, "
            "dramatically reducing hallucination rates."
        ),
        "tags": ["rag", "retrieval", "llm", "generation", "nlp"],
    },
    {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/abs/1706.03762",
        "snippet": (
            "The Transformer architecture replaces recurrence entirely with "
            "multi-head self-attention, enabling parallelised training and "
            "state-of-the-art results on machine translation."
        ),
        "tags": ["transformer", "attention", "nlp", "deep learning"],
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "url": "https://arxiv.org/abs/2201.11903",
        "snippet": (
            "Chain-of-thought (CoT) prompting enables LLMs to decompose complex "
            "problems into intermediate reasoning steps, significantly improving "
            "arithmetic and commonsense QA accuracy."
        ),
        "tags": ["cot", "prompting", "llm", "reasoning"],
    },
    {
        "title": "LangChain: Building Applications with LLMs",
        "url": "https://docs.langchain.com",
        "snippet": (
            "LangChain is an open-source framework that simplifies building "
            "agentic and retrieval-augmented applications using language models "
            "with composable chains, agents, and tools."
        ),
        "tags": ["langchain", "agents", "framework", "llm"],
    },
    {
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "url": "https://arxiv.org/abs/2212.08073",
        "snippet": (
            "Anthropic's Constitutional AI trains models to be helpful and "
            "harmless using a set of principles (a 'constitution') to guide "
            "RLHF, reducing reliance on human labellers for harmful content."
        ),
        "tags": ["safety", "rlhf", "anthropic", "alignment"],
    },
    {
        "title": "Python asyncio: Asynchronous I/O",
        "url": "https://docs.python.org/3/library/asyncio.html",
        "snippet": (
            "asyncio provides infrastructure for writing single-threaded "
            "concurrent code using coroutines, event loops, and futures, "
            "ideal for I/O-bound applications."
        ),
        "tags": ["python", "asyncio", "async", "concurrency"],
    },
    {
        "title": "Model Context Protocol (MCP) Specification",
        "url": "https://modelcontextprotocol.io",
        "snippet": (
            "MCP is an open protocol that standardises how AI applications "
            "connect to external tools, data sources, and services, enabling "
            "plug-and-play tool integration across frameworks."
        ),
        "tags": ["mcp", "protocol", "tools", "agents"],
    },
]

# ---------------------------------------------------------------------------
# In-memory note store
# ---------------------------------------------------------------------------

@dataclass
class Note:
    id: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


_NOTES: dict[str, Note] = {}
_NOTE_COUNTER = 1


def _next_note_id() -> str:
    global _NOTE_COUNTER
    nid = f"note_{_NOTE_COUNTER:04d}"
    _NOTE_COUNTER += 1
    return nid


# ---------------------------------------------------------------------------
# Tool: search_topic
# ---------------------------------------------------------------------------

SEARCH_TOPIC_SPEC = ToolSpec(
    name="search_topic",
    description=(
        "Search for information about a topic and return relevant results. "
        "Returns a JSON array of results with title, url, and snippet fields."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Search query or topic to research.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return (default: 3).",
        },
    },
    requires_approval=False,
    tags=["research", "search"],
)


def search_topic(query: str, limit: int = 3) -> ToolResult:
    """Return mock search results relevant to *query*."""
    keywords = re.findall(r"\w+", query.lower())
    scored: list[tuple[int, dict]] = []
    for item in _MOCK_SEARCH_DB:
        score = sum(
            1
            for kw in keywords
            if kw in item["title"].lower()
            or kw in item["snippet"].lower()
            or any(kw in tag for tag in item["tags"])
        )
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [item for _, item in scored[:limit]]

    if not results:
        return ToolResult.ok(
            json.dumps(
                [
                    {
                        "title": "No results found",
                        "url": "",
                        "snippet": f"No mock results matched '{query}'. In production, this would call a real search API.",
                    }
                ]
            )
        )
    return ToolResult.ok(json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# Tool: take_note
# ---------------------------------------------------------------------------

TAKE_NOTE_SPEC = ToolSpec(
    name="take_note",
    description="Save a research note with a title, content, and optional tags.",
    parameters={
        "title": {
            "type": "string",
            "description": "Short title for the note.",
        },
        "content": {
            "type": "string",
            "description": "Body of the note — can include key findings, quotes, or analysis.",
        },
        "tags": {
            "type": "string",
            "description": "Comma-separated tags for the note (optional).",
        },
    },
    requires_approval=False,
    tags=["research", "notes"],
)


def take_note(title: str, content: str, tags: str = "") -> ToolResult:
    """Persist a research note in the in-memory store."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    note = Note(id=_next_note_id(), title=title, content=content, tags=tag_list)
    _NOTES[note.id] = note
    return ToolResult.ok(f"Note '{title}' saved with id {note.id}.")


# ---------------------------------------------------------------------------
# Tool: list_notes
# ---------------------------------------------------------------------------

LIST_NOTES_SPEC = ToolSpec(
    name="list_notes",
    description="List all saved research notes, optionally filtered by tag.",
    parameters={
        "tag": {
            "type": "string",
            "description": "Optional tag to filter notes by.",
        }
    },
    requires_approval=False,
    tags=["research", "notes"],
)


def list_notes(tag: str = "") -> ToolResult:
    """Return all notes (optionally filtered by *tag*) as formatted text."""
    notes = list(_NOTES.values())
    if tag:
        notes = [n for n in notes if tag.lower() in [t.lower() for t in n.tags]]

    if not notes:
        msg = f"No notes found" + (f" with tag '{tag}'." if tag else ".")
        return ToolResult.ok(msg)

    lines = [f"Research notes ({len(notes)} total):", ""]
    for note in notes:
        tag_str = f"  [tags: {', '.join(note.tags)}]" if note.tags else ""
        lines.append(f"• [{note.id}] {note.title}{tag_str}")
        lines.append(f"  {note.content[:200]}{'...' if len(note.content) > 200 else ''}")
        lines.append("")
    return ToolResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Tool: format_report
# ---------------------------------------------------------------------------

FORMAT_REPORT_SPEC = ToolSpec(
    name="format_report",
    description=(
        "Compile all saved research notes into a structured research report. "
        "Optionally filter by tag and provide a custom report title."
    ),
    parameters={
        "report_title": {
            "type": "string",
            "description": "Title for the research report.",
        },
        "tag": {
            "type": "string",
            "description": "Optional tag to filter which notes are included.",
        },
    },
    requires_approval=False,
    tags=["research", "report"],
)


def format_report(report_title: str = "Research Report", tag: str = "") -> ToolResult:
    """Render all (or tag-filtered) notes as a markdown-style research report."""
    notes = list(_NOTES.values())
    if tag:
        notes = [n for n in notes if tag.lower() in [t.lower() for t in n.tags]]

    if not notes:
        return ToolResult.ok(
            f"No notes to compile" + (f" with tag '{tag}'." if tag else ". Use take_note first.")
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {report_title}",
        f"Generated: {timestamp}",
        f"Notes included: {len(notes)}",
        "",
        "---",
        "",
    ]
    for idx, note in enumerate(notes, start=1):
        lines.append(f"## {idx}. {note.title}")
        if note.tags:
            lines.append(f"*Tags: {', '.join(note.tags)}*")
        lines.append("")
        lines.append(note.content)
        lines.append("")
        lines.append("---")
        lines.append("")

    return ToolResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_research_tools(registry: ToolRegistry) -> None:
    """Register all research assistant tools into *registry*."""
    registry.register(SEARCH_TOPIC_SPEC, search_topic)
    registry.register(TAKE_NOTE_SPEC, take_note)
    registry.register(LIST_NOTES_SPEC, list_notes)
    registry.register(FORMAT_REPORT_SPEC, format_report)
