import { Mastra } from "@mastra/core";
import { LibSQLStore } from "@mastra/libsql";
import { PinoLogger } from "@mastra/loggers";

import { assistantAgent } from "./agent.js";
import { CONFIG } from "./config.js";

export const mastra = new Mastra({
  agents: { assistantAgent },
  storage: new LibSQLStore({ id: "mastra-storage", url: CONFIG.memoryDbUrl }),
  logger: new PinoLogger({ name: "mastra-rag-agent", level: "warn" }),
});
