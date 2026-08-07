import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

import dotenv from 'dotenv';
import mammoth from 'mammoth';
import pdfParse from 'pdf-parse';
import { z } from 'zod';

import { ChatOpenAI } from '@langchain/openai';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { Document } from '@langchain/core/documents';
import { tool } from '@langchain/core/tools';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { MemorySaver } from '@langchain/langgraph';
import { createReactAgent } from '@langchain/langgraph/prebuilt';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, '.env') });

const REQUIRED_ENV = {
  openRouterApiKey: process.env.OPENROUTER_API_KEY,
  hfApiKey: process.env.HF_API_KEY,
};

const missingRequired = Object.entries(REQUIRED_ENV)
  .filter(([, value]) => value == null || String(value).trim() === '')
  .map(([key]) => key);

if (missingRequired.length > 0) {
  console.error(
    `Missing required environment variables: ${missingRequired.join(', ')}`,
  );
  console.error('Set them in a .env file in this folder (see .env.example).');
  process.exit(1);
}

function clamp01(value, fallback) {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : fallback;
}

const CONFIG = {
  openRouterApiKey: REQUIRED_ENV.openRouterApiKey,
  hfApiKey: REQUIRED_ENV.hfApiKey,
  modelName: process.env.MODEL_NAME || 'nvidia/nemotron-3-nano-30b-a3b:free',
  embedModelName:
    process.env.EMBED_MODEL_NAME || 'sentence-transformers/all-MiniLM-L6-v2',
  chromaUrl: process.env.CHROMA_URL || 'http://localhost:8000',
  documentsCollection:
    process.env.CHROMA_COLLECTION_NAME || 'tool_augmented_rag_documents',
  memoryCollection:
    process.env.CHROMA_MEMORY_COLLECTION_NAME ||
    'tool_augmented_rag_conversations',
  dataDir: path.resolve(__dirname, process.env.DATA_DIR || 'data'),
  chunkLength: Math.max(200, Number(process.env.CHUNK_LENGTH) || 1000),
  topK: Math.max(1, Number(process.env.TOP_K) || 4),
  memoryTopK: Math.max(1, Number(process.env.MEMORY_TOP_K) || 3),
  memorySimilarityThreshold: clamp01(
    Number(process.env.MEMORY_SIMILARITY_THRESHOLD),
    0.5,
  ),
  hybridAlpha: clamp01(Number(process.env.HYBRID_ALPHA), 0.6),
  maxOutputTokens: Number(process.env.MAX_OUTPUT_TOKENS) || 2000,
  maxAgentSteps: Number(process.env.MAX_AGENT_STEPS) || 12,
};

const MANIFEST_PATH = path.join(CONFIG.dataDir, '.index-manifest.json');

function ensureDirSync(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function stripNoise(text) {
  return String(text || '')
    .replace(/\u0000/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function sentenceSegments(text) {
  const normalized = String(text || '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .trim();
  if (!normalized) return [];

  if (typeof Intl !== 'undefined' && Intl.Segmenter) {
    const segmenter = new Intl.Segmenter('en', { granularity: 'sentence' });
    return Array.from(segmenter.segment(normalized), (segment) =>
      segment.segment.trim(),
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
  let current = '';

  for (const word of words) {
    if (!current) {
      current = word;
      continue;
    }
    if ((current + ' ' + word).length > maxLength) {
      chunks.push(current);
      current = word;
    } else {
      current += ` ${word}`;
    }
  }
  if (current) chunks.push(current);
  return chunks;
}

function buildSemanticChunks(text, maxLength) {
  const segments = sentenceSegments(text);
  const chunks = [];
  let current = '';

  for (const segment of segments) {
    if (segment.length > maxLength) {
      if (current) {
        chunks.push(current);
        current = '';
      }
      chunks.push(...splitLongSegment(segment, maxLength));
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
  if (current) chunks.push(current);

  return chunks.map(stripNoise).filter(Boolean);
}

async function extractTextFromFile(buffer, fileName) {
  const lower = fileName.toLowerCase();

  if (lower.endsWith('.pdf')) {
    const parsed = await pdfParse(buffer);
    return stripNoise(parsed.text);
  }

  if (lower.endsWith('.docx')) {
    const parsed = await mammoth.extractRawText({ buffer });
    return stripNoise(parsed.value);
  }

  return stripNoise(buffer.toString('utf8'));
}

function tokenize(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

function lexicalOverlapScore(queryTokens, text) {
  if (queryTokens.length === 0) return 0;
  const docTokens = new Set(tokenize(text));
  if (docTokens.size === 0) return 0;
  const matches = queryTokens.filter((token) => docTokens.has(token)).length;
  return matches / queryTokens.length;
}

function createRecordId(...parts) {
  return crypto.createHash('sha1').update(parts.join('::')).digest('hex');
}

// Data folder ingestion (semantic chunking + idempotent upsert into Chroma)

function loadManifest() {
  try {
    return JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'));
  } catch {
    return {};
  }
}

function saveManifest(manifest) {
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
}

async function ingestDataFolder(documentsStore) {
  ensureDirSync(CONFIG.dataDir);

  const manifest = loadManifest();
  const fileNames = fs
    .readdirSync(CONFIG.dataDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && !entry.name.startsWith('.'))
    .map((entry) => entry.name);

  const seen = new Set();
  const summary = [];

  for (const fileName of fileNames) {
    seen.add(fileName);
    const filePath = path.join(CONFIG.dataDir, fileName);
    const buffer = fs.readFileSync(filePath);
    const contentHash = crypto
      .createHash('sha256')
      .update(buffer)
      .digest('hex');
    const previous = manifest[fileName];

    if (previous && previous.contentHash === contentHash) {
      continue;
    }

    if (previous?.ids?.length) {
      await documentsStore.delete({ ids: previous.ids }).catch(() => {});
    }

    const text = await extractTextFromFile(buffer, fileName);
    const chunks = buildSemanticChunks(text, CONFIG.chunkLength);

    if (chunks.length === 0) {
      manifest[fileName] = {
        contentHash,
        ids: [],
        indexedAt: new Date().toISOString(),
      };
      summary.push({
        file: fileName,
        status: 'skipped',
        reason: 'no extractable text',
      });
      continue;
    }

    const ids = chunks.map((chunk, index) =>
      createRecordId(fileName, index, chunk),
    );
    const documents = chunks.map(
      (chunk, index) =>
        new Document({
          pageContent: chunk,
          metadata: { source_file: fileName, chunk_index: index },
        }),
    );

    await documentsStore.addDocuments(documents, { ids });
    manifest[fileName] = {
      contentHash,
      ids,
      indexedAt: new Date().toISOString(),
    };
    summary.push({ file: fileName, status: 'indexed', chunks: chunks.length });
  }

  for (const fileName of Object.keys(manifest)) {
    if (!seen.has(fileName)) {
      const stale = manifest[fileName];
      if (stale?.ids?.length) {
        await documentsStore.delete({ ids: stale.ids }).catch(() => {});
      }
      delete manifest[fileName];
      summary.push({ file: fileName, status: 'removed' });
    }
  }

  saveManifest(manifest);
  return summary;
}

// Hybrid retrieval (vector similarity + lexical keyword overlap)

async function hybridRetrieveDocuments(documentsStore, query, k) {
  const overfetch = Math.max(k * 3, k + 4);
  const candidates = await documentsStore.similaritySearchWithScore(
    query,
    overfetch,
  );
  if (candidates.length === 0) return [];

  const queryTokens = tokenize(query);
  const scored = candidates.map(([doc, distance]) => {
    const vectorSimilarity = 1 / (1 + distance);
    const lexical = lexicalOverlapScore(queryTokens, doc.pageContent);
    const hybrid =
      CONFIG.hybridAlpha * vectorSimilarity +
      (1 - CONFIG.hybridAlpha) * lexical;
    return { doc, vectorSimilarity, lexical, hybrid };
  });

  scored.sort((a, b) => b.hybrid - a.hybrid);
  return scored.slice(0, k);
}

function formatRetrievedContext(items) {
  if (items.length === 0) {
    return 'No relevant internal documents were found in the knowledge base.';
  }

  return items
    .map(({ doc, hybrid, vectorSimilarity, lexical }, index) => {
      const meta = doc.metadata || {};
      return [
        `Result ${index + 1} (hybrid=${hybrid.toFixed(4)}, vector=${vectorSimilarity.toFixed(4)}, keyword=${lexical.toFixed(4)})`,
        `Source: ${meta.source_file || 'unknown'} (chunk ${meta.chunk_index ?? '?'})`,
        `Content: ${doc.pageContent}`,
      ].join('\n');
    })
    .join('\n\n---\n\n');
}

// Long-term conversation memory (separate Chroma collection)

async function recallConversationMemory(memoryStore, query, k) {
  const results = await memoryStore
    .similaritySearchWithScore(query, k)
    .catch(() => []);
  return results
    .map(([doc, distance]) => ({ doc, similarity: 1 / (1 + distance) }))
    .filter(({ similarity }) => similarity >= CONFIG.memorySimilarityThreshold)
    .sort((a, b) => b.similarity - a.similarity);
}

async function saveConversationTurn(memoryStore, humanText, aiText) {
  const timestamp = new Date().toISOString();
  const pageContent = `User: ${humanText}\nAssistant: ${aiText}`;
  const id = createRecordId('conversation', timestamp, humanText);

  await memoryStore.addDocuments(
    [
      new Document({
        pageContent,
        metadata: { type: 'conversation', timestamp, user_message: humanText },
      }),
    ],
    { ids: [id] },
  );
}

// Tool definitions (LangChain `tool()` wrappers)

function getFlightSchedule({ origin, destination }) {
  return {
    route: `${origin} to ${destination}`,
    currency: 'USD',
    outbound: {
      departure: `${origin} 08:10`,
      arrival: `${destination} 14:10`,
      duration_hours: 6,
      price_usd: 420,
      stopovers: 0,
    },
    return: {
      departure: `${destination} 18:40`,
      arrival: `${origin} 00:40`,
      duration_hours: 6,
      price_usd: 430,
      stopovers: 0,
    },
    total_flight_time_hours: 12,
    total_flight_cost_usd: 850,
    note: 'Assumed direct economy round trip for conference travel.',
  };
}

function getHotelBookingSchedule({ city, nights }) {
  const nightlyRateUsd = 175;
  const totalHotelCostUsd = nightlyRateUsd * nights;

  return {
    city,
    currency: 'USD',
    stay_nights: nights,
    check_in: `${city} Day 1`,
    check_out: `${city} Day ${nights + 1}`,
    hotel_name: `${city} Conference Stay`,
    nightly_rate_usd: nightlyRateUsd,
    total_hotel_cost_usd: totalHotelCostUsd,
    note: 'Assumed mid-range conference hotel rate.',
  };
}

function convertCurrency({ amount, from_currency, to_currency }) {
  const rateTable = {
    USD_NGN: 1600,
    NGN_USD: 1 / 1600,
  };

  const key = `${from_currency.toUpperCase()}_${to_currency.toUpperCase()}`;
  const rate = rateTable[key];

  if (!rate) {
    throw new Error(
      `Unsupported currency pair: ${from_currency} to ${to_currency}`,
    );
  }

  const convertedAmount = amount * rate;

  return {
    from_currency: from_currency.toUpperCase(),
    to_currency: to_currency.toUpperCase(),
    amount,
    rate,
    converted_amount: Number(convertedAmount.toFixed(2)),
  };
}

function buildTools(documentsStore) {
  const getFlightScheduleTool = tool(
    async ({ origin, destination, trip_type }) =>
      JSON.stringify(getFlightSchedule({ origin, destination, trip_type })),
    {
      name: 'get_flight_schedule',
      description:
        'Return a round-trip flight schedule between two cities and the total flight cost in USD.',
      schema: z.object({
        origin: z.string().describe('Origin city or airport'),
        destination: z.string().describe('Destination city or airport'),
        trip_type: z.enum(['round_trip']).describe('Trip type'),
      }),
    },
  );

  const getHotelBookingScheduleTool = tool(
    async ({ city, nights }) =>
      JSON.stringify(getHotelBookingSchedule({ city, nights })),
    {
      name: 'get_hotel_booking_schedule',
      description:
        'Return a hotel booking schedule for a stay and the total hotel cost in USD.',
      schema: z.object({
        city: z.string().describe('City for the stay'),
        nights: z.number().int().positive().describe('Number of nights'),
      }),
    },
  );

  const convertCurrencyTool = tool(
    async ({ amount, from_currency, to_currency }) =>
      JSON.stringify(convertCurrency({ amount, from_currency, to_currency })),
    {
      name: 'convert_currency',
      description:
        'Convert an amount from one currency to another using a deterministic exchange rate table.',
      schema: z.object({
        amount: z.number().describe('Amount to convert'),
        from_currency: z.string().describe('Source currency code'),
        to_currency: z.string().describe('Target currency code'),
      }),
    },
  );

  const queryInternalKnowledgeBaseTool = tool(
    async ({ query, top_k }) => {
      const items = await hybridRetrieveDocuments(
        documentsStore,
        query,
        top_k || CONFIG.topK,
      );
      return formatRetrievedContext(items);
    },
    {
      name: 'query_internal_knowledge_base',
      description:
        'Search the internal knowledge base built from files in the data/ folder using hybrid ' +
        '(semantic + keyword) retrieval. Use this whenever the user asks about internal, ' +
        'company-specific, or document-based information that general knowledge would not cover.',
      schema: z.object({
        query: z
          .string()
          .describe(
            'The search query to look up in the internal knowledge base',
          ),
        top_k: z
          .number()
          .int()
          .positive()
          .optional()
          .describe('Number of results to return'),
      }),
    },
  );

  return [
    getFlightScheduleTool,
    getHotelBookingScheduleTool,
    convertCurrencyTool,
    queryInternalKnowledgeBaseTool,
  ];
}

// Conversation trace formatting

function describeMessage(message) {
  const type =
    typeof message._getType === 'function' ? message._getType() : message.type;

  if (type === 'human') {
    return `[User]\n${message.content}`;
  }

  if (type === 'system') {
    return `[System]\n${message.content}`;
  }

  if (type === 'tool') {
    const label = message.name || 'tool';
    return `[Tool Result: ${label}]\n${message.content}`;
  }

  if (type === 'ai') {
    const toolCalls = message.tool_calls || [];
    if (toolCalls.length > 0) {
      const calls = toolCalls
        .map((call) => `  - ${call.name}(${JSON.stringify(call.args)})`)
        .join('\n');
      const header = `[Assistant] requested ${toolCalls.length} tool call(s):`;
      return message.content
        ? `${header}\n${calls}\n${message.content}`
        : `${header}\n${calls}`;
    }
    return `[Assistant]\n${message.content}`;
  }

  return `[${type}]\n${message.content}`;
}

// Main

const SYSTEM_PROMPT = [
  'You are a professional operations assistant with access to internal tools.',
  'Available tools:',
  '- get_flight_schedule: round-trip flight schedule and total flight cost in USD.',
  '- get_hotel_booking_schedule: hotel schedule and total hotel cost in USD for a stay.',
  '- convert_currency: convert an amount between currencies using a fixed rate table.',
  '- query_internal_knowledge_base: search internal documents indexed from the data/ folder.',
  "Only call a tool when it is relevant to the user's request.",
  'If the request could relate to internal or document-based information, check ' +
    'query_internal_knowledge_base before answering from general knowledge.',
  'Be concise, accurate, and state any assumptions you make.',
].join(' ');

async function main() {
  const prompt = process.argv.slice(2).join(' ').trim();

  if (!prompt) {
    console.error('Usage: node main.js "your prompt here"');
    process.exitCode = 1;
    return;
  }

  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: CONFIG.hfApiKey,
    model: CONFIG.embedModelName,
    provider: 'hf-inference',
  });

  const documentsStore = new Chroma(embeddings, {
    collectionName: CONFIG.documentsCollection,
    url: CONFIG.chromaUrl,
  });

  const memoryStore = new Chroma(embeddings, {
    collectionName: CONFIG.memoryCollection,
    url: CONFIG.chromaUrl,
  });

  await ingestDataFolder(documentsStore);

  const llm = new ChatOpenAI({
    apiKey: CONFIG.openRouterApiKey,
    model: CONFIG.modelName,
    temperature: 0,
    maxTokens: CONFIG.maxOutputTokens,
    configuration: {
      baseURL: 'https://openrouter.ai/api/v1',
      defaultHeaders: {
        'HTTP-Referer': 'https://openai.com',
        'X-Title': 'Tool Augmented RAG Agent using LangChain',
      },
      fetch: globalThis.fetch,
    },
  });

  const tools = buildTools(documentsStore);

  const agent = createReactAgent({
    llm,
    tools,
    checkpointSaver: new MemorySaver(),
  });

  const recalled = await recallConversationMemory(
    memoryStore,
    prompt,
    CONFIG.memoryTopK,
  );

  const messages = [new SystemMessage(SYSTEM_PROMPT)];
  if (recalled.length > 0) {
    const memoryContext = recalled
      .map(({ doc }) => doc.pageContent)
      .join('\n\n');
    messages.push(
      new SystemMessage(
        `Relevant past conversation that may help with the current request:\n\n${memoryContext}`,
      ),
    );
  }
  messages.push(new HumanMessage(prompt));

  if (recalled.length > 0) {
    console.log('===== Recalled Long-Term Memory =====\n');
    recalled.forEach(({ doc, similarity }, index) => {
      const savedAt = doc.metadata?.timestamp || 'unknown';
      console.log(
        `Memory ${index + 1} (similarity=${similarity.toFixed(4)}, saved_at=${savedAt})`,
      );
      console.log(doc.pageContent);
      console.log();
    });
  }

  const result = await agent.invoke(
    { messages },
    {
      configurable: { thread_id: 'cli-session' },
      recursionLimit: CONFIG.maxAgentSteps,
    },
  );

  console.log('===== Conversation History =====\n');
  for (const message of result.messages) {
    console.log(describeMessage(message));
    console.log();
  }

  const finalMessage = result.messages[result.messages.length - 1];
  const finalResponse = String(finalMessage?.content ?? '').trim();

  console.log('===== Final Response =====\n');
  console.log(finalResponse);

  await saveConversationTurn(memoryStore, prompt, finalResponse);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  const looksLikeChromaConnectionError =
    /ECONNREFUSED|fetch failed|ChromaConnectionError|Failed to connect to chromadb/i.test(
      message,
    );
  if (looksLikeChromaConnectionError) {
    console.error(
      `\nCould not reach ChromaDB at ${CONFIG.chromaUrl}. Start a Chroma server first, e.g.:\n` +
        '  docker run -p 8000:8000 chromadb/chroma\n' +
        'or:\n' +
        '  pip install chromadb && chroma run',
    );
  }
  process.exitCode = 1;
});
