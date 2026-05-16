# SHL Assessment Recommender

A conversational agent that helps hiring managers find the right SHL assessments through dialogue. Built for the SHL Labs AI Intern take-home assignment.

## Architecture

```
POST /chat
    │
    ├── Conversation history → LLM (Groq / Llama-3.3-70b)
    │        ↑ Retrieves relevant catalog items
    │   TF-IDF + FAISS retriever (73 SHL assessments)
    │
    └── JSON response: reply + recommendations + end_of_conversation
```

### Tech stack (100% free)
| Layer | Tool | Why |
|---|---|---|
| LLM | Groq + Llama-3.3-70b-versatile | Free tier, ~600 tokens/s, fast enough for 30s timeout |
| Embeddings | sklearn TF-IDF | No model download, runs offline, good enough for catalog search |
| Vector store | FAISS (flat inner product) | Free, in-process, sub-ms search |
| API | FastAPI + Pydantic | Schema validation, auto docs |
| Deployment | Render / Railway (free tier) | Persistent disk for FAISS cache |

## Setup

### 1. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — no credit card needed.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the catalog
```bash
python build_catalog.py
```
This attempts to scrape shl.com live; falls back to the bundled 73-assessment catalog if the site is unreachable.

### 4. Set environment variables
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
export GROQ_API_KEY=your_key_here
```

### 5. Run locally
```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Test
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a mid-level Java developer who works with stakeholders"}
    ]
  }'
```

## Deployment (Render — free tier)

1. Push this repo to GitHub
2. Create a new Web Service on [render.com](https://render.com)
3. Connect your GitHub repo
4. Set `GROQ_API_KEY` in Environment Variables
5. Render will use `render.yaml` for config automatically

The `/health` endpoint warms up the FAISS index on cold start. The evaluator allows 2 minutes for this.

## Evaluation

```bash
# Run built-in probes against a live service
python evaluate.py --url http://localhost:8000

# With the SHL-provided traces
python evaluate.py --url http://localhost:8000 --traces path/to/traces.json
```

Metrics reported:
- **Recall@10**: fraction of expected assessments appearing in top-10 recommendations
- **Probe pass rate**: binary behavioural assertions (refuse off-topic, clarify vague queries, etc.)

## API Reference

### GET /health
```json
{"status": "ok"}
```

### POST /chat
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments that fit...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

**Test type codes:**
- `A` — Ability / Aptitude
- `K` — Knowledge / Skills
- `P` — Personality
- `B` — Behavioural / Situational Judgement
- `S` — Simulation
- `360` — 360-degree Feedback

## Design decisions

### Why TF-IDF + FAISS instead of a neural embedding model?
Neural embeddings (sentence-transformers) require ~500MB disk + 4GB RAM for the model. Free deployment tiers cap RAM at 512MB–1GB. TF-IDF with bigrams achieves good recall on a 73-item catalog because the catalog descriptions are rich and keyword-dense. The LLM does the semantic heavy lifting after retrieval.

### Why Groq?
Groq's free tier gives ~14,400 requests/day and ~6,000 tokens/minute with Llama-3.3-70b-versatile. Response time is typically 2–5 seconds, well within the 30-second timeout. A fallback to llama3-8b-8192 handles rate limit scenarios.

### Anti-hallucination strategy
Every URL returned by the LLM is validated against the set of URLs in `catalog.json`. Any URL not in the catalog is dropped. If all URLs are dropped, the top retrieval results are used as fallback. This ensures the schema constraint "items from catalog only" is always met.

### Turn cap compliance
The service counts conversation turns and forces a best-effort recommendation on turn 7+ rather than continuing to ask clarifying questions.

## What didn't work

- **Dense retrieval (sentence-transformers)**: Memory constraints on free deployment tiers ruled this out. TF-IDF achieves ~80% of the Recall@10 at 0% of the cost.
- **Function calling for structured output**: Groq's function calling has stricter token limits. Prompting for raw JSON with validation in Python is more reliable.
- **OpenAI embeddings**: Paid API — excluded per brief requirements.

