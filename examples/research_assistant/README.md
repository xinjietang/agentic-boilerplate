# Research Assistant

An agentic example built on [agentic-boilerplate](../../README.md).  
The agent searches for information, takes structured notes, and compiles
research reports — all in a conversational workflow.

---

## What it demonstrates

| Boilerplate feature | How it's used |
|---|---|
| `ToolSpec` / `ToolResult` | Four research tools with descriptive JSON-schema parameters |
| `ToolRegistry` | Centralised tool registration via `register_research_tools()` |
| `AgentLoop` subclass | `ResearchAgentLoop._call_llm()` dispatches search / note / report intents |
| `ContextManager` | Conversation history lets the agent remember previously found sources |
| `EventBus` | All tool calls and results surface as structured events in the CLI |
| Config YAML | Custom system prompt in `config.yaml` |

---

## Tools

| Tool | Description |
|---|---|
| `search_topic` | Search a mock knowledge base for articles matching a query |
| `take_note` | Save a research note with title, content, and optional tags |
| `list_notes` | List all saved notes, optionally filtered by tag |
| `format_report` | Compile all (or tag-filtered) notes into a markdown research report |

> **Stub vs production:** `search_topic` uses an in-process mock database.
> Replace it with a real search API such as
> [Tavily](https://tavily.com), [Brave Search](https://brave.com/search/api/),
> or [SerpAPI](https://serpapi.com).  For note persistence, swap the in-memory
> dict with SQLite, PostgreSQL, or a vector store (Chroma, Pinecone, etc.).

---

## Quick start

```bash
pip install -e ".[dev]"

cd examples/research_assistant
python agent.py

# Headless single-shot examples
python agent.py --headless "search for RAG"
python agent.py --headless "what is chain-of-thought prompting?"
python agent.py --headless "take note: RAG insight — combines retrieval with generation to reduce hallucinations"
python agent.py --headless "list my notes"
python agent.py --headless "generate report"
```

---

## Sample multi-turn research session

```
you> search for transformer architecture
⚙ TOOL CALL  search_topic(query='transformer architecture', limit=3)
⚙ TOOL RESULT  search_topic → [
  {
    "title": "Attention Is All You Need",
    "url": "https://arxiv.org/abs/1706.03762",
    "snippet": "The Transformer architecture replaces recurrence entirely with multi-head self-attention…"
  }
]

you> take note: Transformer paper — replaces RNNs with self-attention; enables parallelised training
⚙ TOOL CALL  take_note(title='Transformer paper', content='replaces RNNs with self-attention…')
⚙ TOOL RESULT  Note 'Transformer paper' saved with id note_0001.

you> search for RAG retrieval
⚙ TOOL CALL  search_topic(query='RAG retrieval', limit=3)
...

you> generate report
⚙ TOOL CALL  format_report(report_title='Research Report')
⚙ TOOL RESULT  format_report →
# Research Report
Generated: 2025-01-15 14:32

---

## 1. Transformer paper
replaces RNNs with self-attention; enables parallelised training

---
```

---

## Adapting to production

1. **Swap the LLM** — replace `_call_llm()` in `agent.py` with a real OpenAI / Anthropic call.
   With a real LLM the agent can decide autonomously when to search, when to take notes,
   and how to structure a multi-step research workflow.

2. **Use a real search API:**
   ```python
   import httpx
   
   def search_topic(query: str, limit: int = 3) -> ToolResult:
       resp = httpx.get(
           "https://api.tavily.com/search",
           json={"query": query, "max_results": limit},
           headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
       )
       results = resp.json().get("results", [])
       return ToolResult.ok(json.dumps(results, indent=2))
   ```

3. **Persist notes** — replace `_NOTES` dict with a SQLite table or vector DB for
   semantic note retrieval.

4. **Add web scraping** — add a `fetch_url` tool that retrieves and parses article
   content, then calls `take_note` automatically with key excerpts.

5. **Scheduled research digests** — run the agent headlessly on a topic nightly and
   email the compiled report.
