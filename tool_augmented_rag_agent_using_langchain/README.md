# Tool Augmented RAG Agent using LangChain

A LangChain/LangGraph agent that combines tool calling with hybrid RAG retrieval and
long-term conversation memory, all backed by ChromaDB.

## What it does

- Runs a [LangGraph](https://langchain-ai.github.io/langgraphjs/) ReAct agent
  (`createReactAgent`) with an in-memory checkpointer for its tool-calling loop.
- Gives the agent four LangChain tools:
  - `get_flight_schedule` — round-trip flight schedule and cost in USD.
  - `get_hotel_booking_schedule` — hotel schedule and cost in USD for a stay.
  - `convert_currency` — convert an amount between currencies using a fixed rate table.
  - `query_internal_knowledge_base` — hybrid (semantic + keyword) search over documents
    indexed from the `data/` folder.
- Automatically indexes every file placed in `data/` into ChromaDB using HuggingFace
  `sentence-transformers/all-MiniLM-L6-v2` embeddings and sentence-aware semantic
  chunking. Re-runs are idempotent: unchanged files are skipped, changed files are
  re-embedded, removed files are pruned from the index.
- Retrieves documents with **hybrid retrieval**: Chroma vector similarity combined with
  a lexical keyword-overlap score, re-ranked and blended (`HYBRID_ALPHA` controls the
  weighting).
- Persists every **conversation turn** (the user's message and the agent's final
  answer — not the intermediate tool-calling steps) into a separate Chroma collection,
  and recalls semantically relevant past turns at the start of each run as long-term
  memory.
- Reasons with OpenRouter's `nvidia/nemotron-3-nano-30b-a3b:free` model.
- Prints the full conversation trace (recalled memory, system/user/tool/assistant
  messages, and the final response) to standard output. No HTTP server, no endpoints.

## Files

- `main.js` - CLI entry point, tools, ingestion pipeline, hybrid retrieval, memory, and the LangGraph agent.
- `data/` - drop any files here to have them indexed automatically; ships with one sample document.
- `package.json` - dependencies and scripts.
- `.env.example` - example environment variables.

## Setup

Install dependencies:

```bash
npm install --legacy-peer-deps
```

`--legacy-peer-deps` is required because `@langchain/community` lists many optional
integration peers (e.g. `@browserbasehq/stagehand`) that are pinned to zod v3, while
this project uses zod v4. None of those optional integrations are used here, so the
flag is safe.

Start a ChromaDB server (required — the JS Chroma client talks to a running server):

```bash
docker run -p 8000:8000 chromadb/chroma
# or
pip install chromadb && chroma run
```

Create a `.env` file in the same folder as `main.js`:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
HF_API_KEY=your_huggingface_api_key_here
```

Every other variable (model names, Chroma URL, chunk size, retrieval weights, etc.)
already has a default in `main.js` — see `.env.example` for the full list and only
override what you need.

## Usage

```bash
node main.js "What is our company's policy on conference hotel stays?"
```

On each run the script:

1. Ingests any new or changed files in `data/` into the ChromaDB documents collection.
2. Recalls relevant past conversation turns from the memory collection (if any).
3. Runs the agent, letting it call whichever tools are relevant to the prompt.
4. Prints the recalled memory (if any), the full step-by-step conversation history for
   this run, and the final response.
5. Saves this run's user message + final answer into the conversation-memory
   collection for future recall.

## Notes

- `data/.index-manifest.json` tracks a content hash per indexed file so re-running the
  script never re-embeds or duplicates unchanged documents.
- The RAG tool and the long-term memory store use separate Chroma collections
  (`CHROMA_COLLECTION_NAME` and `CHROMA_MEMORY_COLLECTION_NAME`) so document retrieval
  and conversation recall never mix.
