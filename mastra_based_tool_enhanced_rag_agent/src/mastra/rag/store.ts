import crypto from "crypto";
import fs from "fs";
import path from "path";

import { ChromaClient, type Collection, type Metadata } from "chromadb";

import { CONFIG } from "../config.js";
import { buildSemanticChunks, extractTextFromFile, lexicalOverlapScore, tokenize } from "./chunking.js";
import { embedText } from "./embeddings.js";

function chromaClientArgsFromUrl(url: string) {
  const parsed = new URL(url);
  return {
    host: parsed.hostname,
    port: parsed.port ? Number(parsed.port) : 8000,
    ssl: parsed.protocol === "https:",
  };
}

const chromaClient = new ChromaClient(chromaClientArgsFromUrl(CONFIG.chromaUrl));
const MANIFEST_PATH = path.join(CONFIG.dataDir, ".index-manifest.json");

interface ManifestEntry {
  contentHash: string;
  ids: string[];
  indexedAt: string;
}
type Manifest = Record<string, ManifestEntry>;

export interface RetrievedChunk {
  text: string;
  sourceFile: string;
  chunkIndex: number;
  vectorSimilarity: number;
  lexical: number;
  hybrid: number;
}

let collectionPromise: Promise<Collection> | null = null;

function getCollection(): Promise<Collection> {
  if (!collectionPromise) {
    collectionPromise = chromaClient.getOrCreateCollection({
      name: CONFIG.documentsCollection,
      embeddingFunction: null,
    });
  }
  return collectionPromise;
}

function createRecordId(...parts: string[]): string {
  return crypto.createHash("sha1").update(parts.join("::")).digest("hex");
}

function loadManifest(): Manifest {
  try {
    return JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8"));
  } catch {
    return {};
  }
}

function saveManifest(manifest: Manifest): void {
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
}

export interface IngestSummaryItem {
  file: string;
  status: "indexed" | "skipped" | "removed";
  chunks?: number;
  reason?: string;
}

/**
 * Indexes every file in `data/` into Chroma. Idempotent: unchanged files are
 * skipped, changed files are re-embedded, and files removed from disk are
 * pruned from the collection — tracked via a content-hash manifest.
 */
export async function ingestDataFolder(): Promise<IngestSummaryItem[]> {
  fs.mkdirSync(CONFIG.dataDir, { recursive: true });
  const collection = await getCollection();
  const manifest = loadManifest();

  const fileNames = fs
    .readdirSync(CONFIG.dataDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && !entry.name.startsWith("."))
    .map((entry) => entry.name);

  const seen = new Set<string>();
  const summary: IngestSummaryItem[] = [];

  for (const fileName of fileNames) {
    seen.add(fileName);
    const filePath = path.join(CONFIG.dataDir, fileName);
    const buffer = fs.readFileSync(filePath);
    const contentHash = crypto.createHash("sha256").update(buffer).digest("hex");
    const previous = manifest[fileName];

    if (previous && previous.contentHash === contentHash) {
      continue;
    }

    if (previous?.ids?.length) {
      await collection.delete({ ids: previous.ids }).catch(() => {});
    }

    const text = await extractTextFromFile(buffer, fileName);
    const chunks = buildSemanticChunks(text, CONFIG.chunkLength);

    if (chunks.length === 0) {
      manifest[fileName] = { contentHash, ids: [], indexedAt: new Date().toISOString() };
      summary.push({ file: fileName, status: "skipped", reason: "no extractable text" });
      continue;
    }

    const ids = chunks.map((chunk, index) => createRecordId(fileName, String(index), chunk));
    const embeddings = [];
    for (const chunk of chunks) {
      embeddings.push(await embedText(chunk));
    }

    await collection.upsert({
      ids,
      documents: chunks,
      embeddings,
      metadatas: chunks.map((_, index) => ({ source_file: fileName, chunk_index: index })),
    });

    manifest[fileName] = { contentHash, ids, indexedAt: new Date().toISOString() };
    summary.push({ file: fileName, status: "indexed", chunks: chunks.length });
  }

  for (const fileName of Object.keys(manifest)) {
    if (!seen.has(fileName)) {
      const stale = manifest[fileName];
      if (stale?.ids?.length) {
        await collection.delete({ ids: stale.ids }).catch(() => {});
      }
      delete manifest[fileName];
      summary.push({ file: fileName, status: "removed" });
    }
  }

  saveManifest(manifest);
  return summary;
}

/** Hybrid retrieval: blends Chroma vector similarity with lexical keyword overlap. */
export async function hybridRetrieveDocuments(query: string, k: number): Promise<RetrievedChunk[]> {
  const collection = await getCollection();
  const overfetch = Math.max(k * 3, k + 4);
  const queryEmbedding = await embedText(query);

  const result = await collection.query({
    queryEmbeddings: [queryEmbedding],
    nResults: overfetch,
    include: ["documents", "metadatas", "distances"],
  });
  const rows = result.rows()[0] ?? [];
  if (rows.length === 0) return [];

  const queryTokens = tokenize(query);
  const scored: RetrievedChunk[] = rows.map((row) => {
    const distance = row.distance ?? 1;
    const vectorSimilarity = 1 / (1 + distance);
    const text = row.document ?? "";
    const lexical = lexicalOverlapScore(queryTokens, text);
    const hybrid = CONFIG.hybridAlpha * vectorSimilarity + (1 - CONFIG.hybridAlpha) * lexical;
    const meta = row.metadata as Metadata & { source_file?: string; chunk_index?: number };
    return {
      text,
      sourceFile: String(meta?.source_file ?? "unknown"),
      chunkIndex: Number(meta?.chunk_index ?? -1),
      vectorSimilarity,
      lexical,
      hybrid,
    };
  });

  scored.sort((a, b) => b.hybrid - a.hybrid);
  return scored.slice(0, k);
}

export function formatRetrievedContext(items: RetrievedChunk[]): string {
  if (items.length === 0) {
    return "No relevant internal documents were found in the knowledge base.";
  }

  return items
    .map((item, index) =>
      [
        `Result ${index + 1} (hybrid=${item.hybrid.toFixed(4)}, vector=${item.vectorSimilarity.toFixed(4)}, keyword=${item.lexical.toFixed(4)})`,
        `Source: ${item.sourceFile} (chunk ${item.chunkIndex})`,
        `Content: ${item.text}`,
      ].join("\n"),
    )
    .join("\n\n---\n\n");
}
