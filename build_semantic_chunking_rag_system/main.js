import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";

import dotenv from "dotenv";
import express from "express";
import multer from "multer";
import { InferenceClient } from "@huggingface/inference";
import mammoth from "mammoth";
import pdfParse from "pdf-parse";
import { ChromaClient } from "chromadb";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, ".env.local") });
dotenv.config({ path: path.join(__dirname, ".env") });

const requiredConfig = {
  hfApiKey: process.env.HF_API_KEY,
  embedModelName: process.env.EMBED_MODEL_NAME,
  geminiApiKey: process.env.GEMINI_API_KEY,
  llmModelName: process.env.LLM_MODEL_NAME,
  chromaDbHost: process.env.CHROMA_DB_HOST,
  chromaDbPort: process.env.CHROMA_DB_PORT,
  ragDataDir: process.env.RAG_DATA_DIR,
  chunkLength: process.env.CHUNK_LENGTH,
  serverPort: process.env.SERVER_PORT,
};

const missingRequired = Object.entries(requiredConfig)
  .filter(([, value]) => value == null || String(value).trim() === "")
  .map(([key]) => key);

if (missingRequired.length > 0) {
  console.error(
    `Missing required environment variables: ${missingRequired.join(", ")}`
  );
  process.exit(1);
}

const CONFIG = {
  hfApiKey: requiredConfig.hfApiKey,
  embedModelName: requiredConfig.embedModelName,
  geminiApiKey: requiredConfig.geminiApiKey,
  llmModelName: normalizeGeminiModelName(requiredConfig.llmModelName),
  chromaDbHost: normalizeChromaHost(requiredConfig.chromaDbHost),
  chromaDbPort: Number(requiredConfig.chromaDbPort),
  ragDataDir: path.resolve(__dirname, requiredConfig.ragDataDir),
  chunkLength: Math.max(200, Number(requiredConfig.chunkLength) || 1000),
  serverPort: Number(requiredConfig.serverPort) || 3000,
  collectionName: process.env.CHROMA_COLLECTION_NAME || "semantic_chunking_rag",
  topK: Number(process.env.TOP_K) || 4,
  requestTimeoutMs: Number(process.env.REQUEST_TIMEOUT_MS) || 120000,
};

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const hfClient = new InferenceClient(CONFIG.hfApiKey);
const chromaClient = new ChromaClient({
  host: CONFIG.chromaDbHost,
  port: CONFIG.chromaDbPort,
  ssl: String(process.env.CHROMA_DB_SSL || "false").toLowerCase() === "true",
});

let collectionPromise = null;

function normalizeGeminiModelName(modelName) {
  return String(modelName).replace(/^models\//, "");
}

function normalizeChromaHost(host) {
  const value = String(host).trim();
  if (!value) {
    return value;
  }

  if (/^https?:\/\//i.test(value)) {
    return new URL(value).hostname;
  }

  return value.replace(/\/.*$/, "");
}

function ensureDirSync(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function sanitizeFilename(name) {
  return String(name || "document")
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "document";
}

function stripMarkdownNoise(text) {
  return String(text || "")
    .replace(/\u0000/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function sentenceSegments(text) {
  const normalized = String(text || "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .trim();

  if (!normalized) {
    return [];
  }

  if (typeof Intl !== "undefined" && Intl.Segmenter) {
    const segmenter = new Intl.Segmenter("en", { granularity: "sentence" });
    return Array.from(segmenter.segment(normalized), (segment) =>
      segment.segment.trim()
    ).filter(Boolean);
  }

  return normalized
    .split(/(?<=[.!?])\s+|\n+/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function splitLongSegment(segment, maxLength) {
  const words = segment.split(/\s+/).filter(Boolean);
  const chunks = [];
  let current = "";

  for (const word of words) {
    if (!current) {
      current = word;
      continue;
    }

    if ((current + " " + word).length > maxLength) {
      chunks.push(current);
      current = word;
    } else {
      current += ` ${word}`;
    }
  }

  if (current) {
    chunks.push(current);
  }

  return chunks;
}

function buildSemanticChunks(text, maxLength) {
  const segments = sentenceSegments(text);
  const chunks = [];
  let current = "";

  for (const segment of segments) {
    if (segment.length > maxLength) {
      if (current) {
        chunks.push(current);
        current = "";
      }

      const subChunks = splitLongSegment(segment, maxLength);
      chunks.push(...subChunks);
      continue;
    }

    const candidate = current ? `${current} ${segment}` : segment;
    if (candidate.length > maxLength && current) {
      chunks.push(current);
      current = segment;
    } else {
      current = candidate;
    }
  }

  if (current) {
    chunks.push(current);
  }

  return chunks.map(stripMarkdownNoise).filter(Boolean);
}

function isPdf(file) {
  return file.mimetype === "application/pdf" || file.originalname.toLowerCase().endsWith(".pdf");
}

function isDocx(file) {
  return (
    file.mimetype ===
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
    file.originalname.toLowerCase().endsWith(".docx")
  );
}

async function extractTextFromFile(file) {
  if (isPdf(file)) {
    const parsed = await pdfParse(file.buffer);
    return stripMarkdownNoise(parsed.text);
  }

  if (isDocx(file)) {
    const parsed = await mammoth.extractRawText({ buffer: file.buffer });
    return stripMarkdownNoise(parsed.value);
  }

  if (
    file.mimetype.startsWith("text/") ||
    file.originalname.match(/\.(txt|md|csv|json|yaml|yml|html|htm)$/i)
  ) {
    return stripMarkdownNoise(file.buffer.toString("utf8"));
  }

  return stripMarkdownNoise(file.buffer.toString("utf8"));
}

function meanPoolTokenEmbeddings(tokenEmbeddings) {
  if (!Array.isArray(tokenEmbeddings) || tokenEmbeddings.length === 0) {
    throw new Error("Empty HuggingFace embedding response.");
  }

  if (typeof tokenEmbeddings[0] === "number") {
    return tokenEmbeddings;
  }

  const vectors = tokenEmbeddings.filter(
    (row) => Array.isArray(row) && row.every((value) => typeof value === "number")
  );

  if (vectors.length === 0) {
    throw new Error("Unexpected embedding shape from HuggingFace API.");
  }

  const dimension = vectors[0].length;
  const totals = new Array(dimension).fill(0);

  for (const vector of vectors) {
    for (let index = 0; index < dimension; index += 1) {
      totals[index] += vector[index];
    }
  }

  return totals.map((value) => value / vectors.length);
}

function normalizeVector(vector) {
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (!magnitude) {
    return vector;
  }

  return vector.map((value) => value / magnitude);
}

async function callWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Request failed for ${url}: ${message}`);
  } finally {
    clearTimeout(timeout);
  }
}

async function embedText(text) {
  try {
    const result = await hfClient.featureExtraction({
      model: CONFIG.embedModelName,
      inputs: text,
    });

    return normalizeVector(meanPoolTokenEmbeddings(result));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`HuggingFace embeddings failed: ${message}`);
  }
}

async function getCollection() {
  if (!collectionPromise) {
    collectionPromise = chromaClient.getOrCreateCollection({
      name: CONFIG.collectionName,
      embeddingFunction: null,
    });
  }

  return collectionPromise;
}

async function saveUploadedFile(file) {
  const uploadsDir = path.join(CONFIG.ragDataDir, "uploads");
  ensureDirSync(uploadsDir);

  const storedName = `${Date.now()}_${crypto.randomUUID()}_${sanitizeFilename(
    file.originalname
  )}`;
  const storedPath = path.join(uploadsDir, storedName);
  await fs.promises.writeFile(storedPath, file.buffer);

  return storedPath;
}

function createRecordId(sourceName, chunkIndex, chunkText) {
  return crypto
    .createHash("sha1")
    .update(`${sourceName}::${chunkIndex}::${chunkText}`)
    .digest("hex");
}

async function indexFile(file) {
  const savedPath = await saveUploadedFile(file);
  const rawText = await extractTextFromFile(file);
  const chunks = buildSemanticChunks(rawText, CONFIG.chunkLength);
  const collection = await getCollection();

  if (chunks.length === 0) {
    return {
      originalName: file.originalname,
      storedPath: savedPath,
      chunksIndexed: 0,
      message: "No extractable text found.",
    };
  }

  const embeddings = [];
  for (const chunk of chunks) {
    embeddings.push(await embedText(chunk));
  }

  const ids = chunks.map((chunk, index) =>
    createRecordId(file.originalname, index, chunk)
  );
  const metadatas = chunks.map((chunk, index) => ({
    source_file: file.originalname,
    stored_file: savedPath,
    chunk_index: index,
    chunk_length: chunk.length,
    content_type: file.mimetype,
  }));

  await collection.upsert({
    ids,
    documents: chunks,
    embeddings,
    metadatas,
  });

  return {
    originalName: file.originalname,
    storedPath: savedPath,
    chunksIndexed: chunks.length,
  };
}

function formatRetrievedContext(results) {
  const documents = results?.documents?.[0] || [];
  const metadatas = results?.metadatas?.[0] || [];
  const distances = results?.distances?.[0] || [];

  if (!documents.length) {
    return "No uploaded documents have been indexed yet.";
  }

  return documents
    .map((document, index) => {
      const meta = metadatas[index] || {};
      const distance = typeof distances[index] === "number" ? distances[index].toFixed(4) : "n/a";
      return [
        `Source: ${meta.source_file || "unknown"}`,
        `Chunk: ${meta.chunk_index ?? "unknown"}`,
        `Distance: ${distance}`,
        `Content: ${document}`,
      ].join("\n");
    })
    .join("\n\n---\n\n");
}

function extractChatText(responseJson) {
  const candidate = responseJson?.candidates?.[0];
  const parts = candidate?.content?.parts || [];
  const text = parts
    .map((part) => part?.text || "")
    .join("")
    .trim();

  if (!text) {
    const feedback = candidate?.finishReason || "no text returned";
    throw new Error(`Gemini response did not contain text (${feedback}).`);
  }

  return text;
}

async function generateAnswer(query, context) {
  const systemPrompt = [
    "You are a careful retrieval-augmented assistant.",
    "Use the provided context to answer the user's question.",
    "If the context does not contain enough information, say so clearly.",
    "When you use the uploaded context, mention the source file names that support the answer.",
  ].join(" ");

  const userPrompt = [
    `Question: ${query}`,
    "",
    "Context:",
    context,
    "",
    "Answer in a concise but useful way.",
  ].join("\n");

  const response = await callWithTimeout(
    `https://generativelanguage.googleapis.com/v1beta/models/${CONFIG.llmModelName}:generateContent`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": CONFIG.geminiApiKey,
      },
      body: JSON.stringify({
        system_instruction: {
          parts: [{ text: systemPrompt }],
        },
        contents: [
          {
            role: "user",
            parts: [{ text: userPrompt }],
          },
        ],
        generationConfig: {
          maxOutputTokens: 1024,
        },
      }),
    },
    CONFIG.requestTimeoutMs
  );
  if (!response) {
    throw new Error("Gemini request did not return a response.");
  }

  const rawText = await response.text();
  if (!response.ok) {
    throw new Error(`Gemini request failed (${response.status}): ${rawText}`);
  }

  let data;
  try {
    data = JSON.parse(rawText);
  } catch {
    throw new Error("Gemini response was not valid JSON.");
  }

  return extractChatText(data);
}

async function retrieveContext(query) {
  const collection = await getCollection();
  const queryEmbedding = await embedText(query);
  const results = await collection.query({
    queryEmbeddings: [queryEmbedding],
    nResults: CONFIG.topK,
    include: ["documents", "metadatas", "distances"],
  });

  return {
    results,
    context: formatRetrievedContext(results),
  };
}

function asyncHandler(handler) {
  return (req, res, next) => {
    Promise.resolve(handler(req, res, next)).catch(next);
  };
}

app.use(express.json({ limit: "1mb" }));

app.post(
  "/upload",
  upload.array("files"),
  asyncHandler(async (req, res) => {
    const files = Array.isArray(req.files) ? req.files : [];

    if (files.length === 0) {
      return res.status(400).json({
        error: "No files were uploaded. Use multipart/form-data with the field name files.",
      });
    }

    ensureDirSync(CONFIG.ragDataDir);

    const results = [];
    for (const file of files) {
      try {
        const result = await indexFile(file);
        results.push({ file: file.originalname, status: "indexed", ...result });
      } catch (error) {
        results.push({
          file: file.originalname,
          status: "failed",
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    const indexedCount = results.filter((item) => item.status === "indexed").length;
    const failedCount = results.filter((item) => item.status === "failed").length;

    return res.status(failedCount > 0 && indexedCount === 0 ? 500 : 200).json({
      message: "Upload processed.",
      indexedCount,
      failedCount,
      results,
    });
  })
);

app.post(
  "/prompt",
  asyncHandler(async (req, res) => {
    const query = String(req.body?.query || "").trim();

    if (!query) {
      return res.status(400).json({
        error: "Missing required field: query",
      });
    }

    const { results, context } = await retrieveContext(query);
    const answer = await generateAnswer(query, context);

    const sources = Array.from(
      new Set(
        (results?.metadatas?.[0] || [])
          .flatMap((metadata) => (metadata?.source_file ? [metadata.source_file] : []))
      )
    );

    return res.json({
      query,
      answer,
      sources,
      retrievedContext: context,
    });
  })
);

app.get("/health", (_, res) => {
  res.status(200).json({ status: "ok" });
});

app.use((error, _req, res, _next) => {
  console.error(error);
  res.status(500).json({
    error: error instanceof Error ? error.message : "Internal server error",
  });
});

async function bootstrap() {
  ensureDirSync(CONFIG.ragDataDir);
  await getCollection();

  app.listen(CONFIG.serverPort, () => {
    console.log(
      `RAG service running on http://localhost:${CONFIG.serverPort} with Chroma at ${CONFIG.chromaDbHost}:${CONFIG.chromaDbPort}`
    );
  });
}

bootstrap().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
