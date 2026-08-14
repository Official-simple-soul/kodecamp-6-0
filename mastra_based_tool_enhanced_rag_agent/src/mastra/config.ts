import path from "path";
import { fileURLToPath } from "url";

import dotenv from "dotenv";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, "..", "..");

dotenv.config({ path: path.join(PROJECT_ROOT, ".env") });

function clamp01(value: number, fallback: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : fallback;
}

const REQUIRED_ENV = {
  openRouterApiKey: process.env.OPENROUTER_API_KEY,
  hfApiKey: process.env.HF_API_KEY,
};

const missingRequired = Object.entries(REQUIRED_ENV)
  .filter(([, value]) => value == null || value.trim() === "")
  .map(([key]) => key);

if (missingRequired.length > 0) {
  console.error(`Missing required environment variables: ${missingRequired.join(", ")}`);
  console.error("Set them in a .env file in this project's root (see .env.example).");
  process.exit(1);
}

export const CONFIG = {
  projectRoot: PROJECT_ROOT,
  openRouterApiKey: REQUIRED_ENV.openRouterApiKey as string,
  hfApiKey: REQUIRED_ENV.hfApiKey as string,
  modelName: process.env.MODEL_NAME || "nvidia/nemotron-3-nano-30b-a3b:free",
  embedModelName: process.env.EMBED_MODEL_NAME || "sentence-transformers/all-MiniLM-L6-v2",
  chromaUrl: process.env.CHROMA_URL || "http://localhost:8000",
  documentsCollection: process.env.CHROMA_COLLECTION_NAME || "mastra_rag_documents",
  dataDir: path.resolve(PROJECT_ROOT, process.env.DATA_DIR || "data"),
  chunkLength: Math.max(200, Number(process.env.CHUNK_LENGTH) || 1000),
  topK: Math.max(1, Number(process.env.TOP_K) || 4),
  hybridAlpha: clamp01(Number(process.env.HYBRID_ALPHA), 0.6),
  maxOutputTokens: Number(process.env.MAX_OUTPUT_TOKENS) || 2000,
  memoryDbUrl: process.env.MEMORY_DB_URL || "file:./mastra-memory.db",
  memoryLastMessages: Math.max(1, Number(process.env.MEMORY_LAST_MESSAGES) || 20),
  memoryThreadId: process.env.MEMORY_THREAD_ID || "cli-session",
  memoryResourceId: process.env.MEMORY_RESOURCE_ID || "cli-user",
} as const;
