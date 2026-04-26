# Daily News Curator

An agentic example built on [agentic-boilerplate](../../README.md).  
The agent fetches, filters, and formats a personalised news digest each morning.

---

## What it demonstrates

| Boilerplate feature | How it's used |
|---|---|
| `ToolSpec` / `ToolResult` | Three news tools with JSON-schema parameter descriptions |
| `ToolRegistry` | All tools registered via a `register_news_tools()` helper |
| `AgentLoop` subclass | `NewsCuratorLoop._call_llm()` routes natural-language requests to tools |
| `ApprovalPolicy` | All news tools are `requires_approval=False` — safe, read-only operations |
| `CLIRenderer` | Event-driven terminal output (tool calls, results, assistant messages) |
| Config YAML | Custom system prompt in `config.yaml` |

---

## Tools

| Tool | Description |
|---|---|
| `fetch_headlines` | Returns the latest headlines for a topic (technology, finance, health, science) |
| `filter_by_keyword` | Filters a headlines JSON array by a case-insensitive keyword |
| `format_digest` | Renders a headlines JSON array as a human-readable morning digest |

> **Stub vs production:** The tools use in-process mock data so no API key is needed.
> To use real data, replace `fetch_headlines` with an HTTP call to
> [NewsAPI](https://newsapi.org), [GNews](https://gnews.io), or an RSS feed parser.

---

## Quick start

```bash
# From the repo root
pip install -e ".[dev]"

# Interactive REPL
cd examples/daily_news_curator
python agent.py

# Headless single-shot examples
python agent.py --headless "fetch me technology news"
python agent.py --headless "get health headlines"
python agent.py --headless "show me a science digest"
python agent.py --headless "get finance headlines"
```

---

## Sample session

```
you> fetch me technology news
⚙ TOOL CALL  fetch_headlines(topic='technology', limit=5)
⚙ TOOL RESULT  fetch_headlines → [{"title": "AI models reach new reasoning...

◀ AGENT  Fetching technology headlines for you…

Tool results:
[fetch_headlines] → [
  {
    "title": "AI models reach new reasoning benchmarks",
    "source": "TechCrunch",
    ...
  },
  ...
]

you> show me a finance digest
⚙ TOOL CALL  fetch_headlines(topic='finance', limit=5)
...
```

---

## Adapting to production

1. **Swap the LLM** — replace `_call_llm()` in `agent.py` with a real OpenAI / Anthropic call.
2. **Use a real news API** — replace the mock data in `tools/news_tools.py` with an HTTP client:
   ```python
   import httpx
   
   async def fetch_headlines(topic: str, limit: int = 5) -> ToolResult:
       resp = httpx.get(
           "https://newsapi.org/v2/everything",
           params={"q": topic, "pageSize": limit, "apiKey": os.environ["NEWS_API_KEY"]},
       )
       articles = resp.json().get("articles", [])
       return ToolResult.ok(json.dumps(articles[:limit], indent=2))
   ```
3. **Schedule it** — run `python agent.py --headless "tech digest"` from a cron job or
   GitHub Actions workflow each morning to get your briefing in Slack / email.
