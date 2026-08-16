# Mastra-Based Tool-Enhanced RAG Agent

A command-line AI assistant built with the [Mastra](https://mastra.ai) TypeScript
agent framework: tool calling, hybrid RAG retrieval, streaming output, and
persistent conversation memory, all driven from `cli.ts`.

## What it does

- Defines a production-style Mastra `Agent` (`src/mastra/agent.ts`) backed by
  OpenRouter's `nvidia/nemotron-3-nano-30b-a3b:free` model (via
  `@openrouter/ai-sdk-provider`) and registers it inside a `Mastra` instance
  (`src/mastra/index.ts`).
- Gives the agent four Mastra tools (`src/mastra/tools.ts`):
  - `get_flight_schedule` — round-trip flight schedule and cost in USD.
  - `get_hotel_booking_schedule` — hotel schedule and cost in USD for a stay.
  - `convert_currency` — convert an amount between currencies using a fixed rate table.
  - `query_internal_knowledge_base` — hybrid (semantic + keyword) search over
    documents indexed from the `data/` folder.
- Automatically indexes every file placed in `data/` into ChromaDB on startup,
  using HuggingFace `sentence-transformers/all-MiniLM-L6-v2` embeddings and
  sentence-aware semantic chunking (`src/mastra/rag/`). Re-runs are idempotent
  via a content-hash manifest — unchanged files are skipped, changed files are
  re-embedded, removed files are pruned.
- Retrieves documents with **hybrid retrieval**: Chroma vector similarity
  blended with a lexical keyword-overlap score (`HYBRID_ALPHA`).
- Maintains **conversation memory** with Mastra's built-in `Memory`, backed by
  a local LibSQL (SQLite) file. History persists across separate `cli.ts`
  runs — the agent remembers earlier turns even after the process exits and
  restarts.
- `cli.ts` runs an interactive REPL: prompts for input, streams the agent's
  response token-by-token to the terminal, shows a spinner while waiting and
  transient notices when a tool is invoked, loops for repeated interaction,
  handles errors gracefully (bad API key, unreachable Chroma, rate limits,
  timeouts) without crashing, and exits cleanly on `exit`/`quit`, Ctrl+D, or
  Ctrl+C (which cancels an in-flight generation first, then exits on a second
  idle Ctrl+C).

## Files

- `cli.ts` — interactive CLI entry point (streaming loop, spinner, error handling, Ctrl+C).
- `src/mastra/index.ts` — the registered `Mastra` instance (also what `mastra dev`/`mastra build` serve).
- `src/mastra/agent.ts` — the `Agent` definition (model, instructions, tools, memory).
- `src/mastra/tools.ts` — the four Mastra tools.
- `src/mastra/rag/` — chunking, HuggingFace embeddings, and the Chroma-backed ingestion/hybrid-retrieval store.
- `src/mastra/config.ts` — environment loading with validation and defaults.
- `data/` — drop any file here to have it indexed automatically; ships with one sample document.
- `.env.example` — example environment variables.

## Setup

Install dependencies:

```bash
npm install
```

Start a ChromaDB server (required — the JS Chroma client talks to a running server):

```bash
docker run -p 8000:8000 chromadb/chroma
# or
pip install chromadb && chroma run
```

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
HF_API_KEY=your_huggingface_api_key_here
```

Every other variable (model name, Chroma URL, chunk size, memory settings,
etc.) already has a default in `src/mastra/config.ts` — see `.env.example`
for the full list and only override what you need.

## Usage

Start the interactive chat:

```bash
npm run cli
```

```
Mastra Tool-Enhanced RAG Agent
Model: nvidia/nemotron-3-nano-30b-a3b:free · Type "exit" or Ctrl+C to quit.

You: What is our company's policy on conference hotel stays?
↳ using query_internal_knowledge_base...
Assistant: Our internal handbook states that conference hotel stays are
capped at a mid-range rate and should not exceed 4 nights...

You: exit
Goodbye!
```

Run the Mastra dev server (REST API + playground) for the same registered agent:

```bash
npm run dev
```

Other scripts: `npm run build` (production build), `npm start` (serve the
build), `npm run typecheck` (type-check without emitting).

## Notes

- `data/.index-manifest.json` (gitignored) tracks a content hash per indexed
  file so re-running the CLI never re-embeds or duplicates unchanged documents.
- Conversation memory lives in a local LibSQL file (`MEMORY_DB_URL`, default
  `file:./mastra-memory.db`), gitignored. Delete it to start a fresh thread.
- `MEMORY_THREAD_ID` / `MEMORY_RESOURCE_ID` control which persistent thread
  the CLI resumes — change them (or delete the DB file) to start over.
- `nvidia/nemotron-3-nano-30b-a3b` is a reasoning model. The agent requests
  `reasoning: { exclude: true }` from OpenRouter (`REASONING_EFFORT`, default
  `low`) so its chain-of-thought is never returned in the response — without
  this, any consumer of the agent (this CLI, `mastra dev`'s playground/API,
  etc.) would see raw reasoning text instead of the final answer.
