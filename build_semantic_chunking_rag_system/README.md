# Build Semantic-Chunking RAG System

This task implements a complete retrieval-augmented generation service in Node.js without LangChain.

## What it does

- Accepts documents through `POST /upload`
- Splits content into semantic chunks using a configurable chunk length
- Generates embeddings with the HuggingFace `sentence-transformers/all-MiniLM-L6-v2` model
- Stores and searches vectors in ChromaDB
- Answers questions through Gemini Flash using `POST /prompt`
- Exposes `GET /health` for liveness checks

## Files

- `main.js` - Express server and full RAG pipeline
- `package.json` - dependencies and scripts
- `.env.example` - sample environment configuration

## Setup

Install dependencies:

```bash
npm install
```

Create a `.env` file in the same folder as `main.js`:

```bash
HF_API_KEY=your_huggingface_api_key_here
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL_NAME=gemini-3.6-flash
CHROMA_DB_HOST=localhost
CHROMA_DB_PORT=9000
RAG_DATA_DIR=rag_data
CHUNK_LENGTH=1000
SERVER_PORT=3000
```

## Run

```bash
node main.js
```

## Endpoints

- `POST /upload` - multipart/form-data with files under the field name `files`
- `POST /prompt` - application/json body with `{ "query": "..." }`
- `GET /health` - returns 200 when the service is live

## Notes

- Documents are saved under `RAG_DATA_DIR`.
- Chunking is sentence-aware and respects the configured `CHUNK_LENGTH`.
- Retrieved chunks are passed to Gemini together with the user question to generate the final answer.
