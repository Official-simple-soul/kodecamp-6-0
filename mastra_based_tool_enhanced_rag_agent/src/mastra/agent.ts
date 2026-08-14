import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { Agent } from "@mastra/core/agent";
import { Memory } from "@mastra/memory";
import { LibSQLStore } from "@mastra/libsql";

import { CONFIG } from "./config.js";
import { allTools } from "./tools.js";

const openrouter = createOpenRouter({ apiKey: CONFIG.openRouterApiKey });

const memory = new Memory({
  storage: new LibSQLStore({ id: "agent-memory-storage", url: CONFIG.memoryDbUrl }),
  // Conversation history is kept as durable thread messages; semantic
  // recall is disabled here because document retrieval is already handled
  // explicitly by the query_internal_knowledge_base tool.
  vector: false,
  options: {
    lastMessages: CONFIG.memoryLastMessages,
  },
});

const SYSTEM_PROMPT = [
  "You are a professional operations assistant with access to internal tools.",
  "Available tools:",
  "- get_flight_schedule: round-trip flight schedule and total flight cost in USD.",
  "- get_hotel_booking_schedule: hotel schedule and total hotel cost in USD for a stay.",
  "- convert_currency: convert an amount between currencies using a fixed rate table.",
  "- query_internal_knowledge_base: search internal documents indexed from the data/ folder.",
  "Only call a tool when it is relevant to the user's request.",
  "If the request could relate to internal or document-based information, check " +
    "query_internal_knowledge_base before answering from general knowledge.",
  "Be concise, accurate, and state any assumptions you make.",
].join(" ");

export const assistantAgent = new Agent({
  id: "assistantAgent",
  name: "Operations Assistant",
  instructions: SYSTEM_PROMPT,
  model: openrouter.chat(CONFIG.modelName),
  tools: allTools,
  memory,
});
