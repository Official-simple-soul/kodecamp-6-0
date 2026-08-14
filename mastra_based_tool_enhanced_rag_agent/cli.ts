#!/usr/bin/env node
import readline from "node:readline/promises";
import { stdin, stdout } from "node:process";

import { CONFIG } from "./src/mastra/config.js";
import { mastra } from "./src/mastra/index.js";
import { ingestDataFolder } from "./src/mastra/rag/store.js";

const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const CYAN = "\x1b[36m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const RED = "\x1b[31m";

const EXIT_COMMANDS = new Set(["exit", "quit", ":q"]);
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function startSpinner(label: string): () => void {
  let frame = 0;
  const interval = setInterval(() => {
    stdout.write(`\r${DIM}${SPINNER_FRAMES[frame % SPINNER_FRAMES.length]} ${label}${RESET}`);
    frame += 1;
  }, 80);
  return () => {
    clearInterval(interval);
    stdout.write(`\r${" ".repeat(label.length + 3)}\r`);
  };
}

/** Turns a raw error into a short, actionable message instead of a stack trace. */
function describeError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);

  if (/ECONNREFUSED|fetch failed|ChromaConnectionError|Failed to connect to chromadb/i.test(message)) {
    return (
      `Could not reach ChromaDB at ${CONFIG.chromaUrl}. Start a Chroma server first, e.g.:\n` +
      `  docker run -p 8000:8000 chromadb/chroma\n` +
      `or:\n` +
      `  pip install chromadb && chroma run`
    );
  }
  if (/401|unauthorized|invalid api key/i.test(message)) {
    return "Authentication failed. Check OPENROUTER_API_KEY / HF_API_KEY in your .env file.";
  }
  if (/429|rate limit/i.test(message)) {
    return "Rate limited by the model provider. Wait a moment and try again.";
  }
  if (/timeout|timed out/i.test(message)) {
    return "The request timed out. Check your connection and try again.";
  }
  return message;
}

async function main(): Promise<void> {
  stdout.write(`${BOLD}Mastra Tool-Enhanced RAG Agent${RESET}\n`);
  stdout.write(`${DIM}Model: ${CONFIG.modelName} · Type "exit" or Ctrl+C to quit.${RESET}\n\n`);

  const stopIngestSpinner = startSpinner("indexing data/ into the knowledge base");
  try {
    await ingestDataFolder();
    stopIngestSpinner();
  } catch (error) {
    stopIngestSpinner();
    stdout.write(`${YELLOW}Warning: could not index data/ at startup.${RESET}\n`);
    stdout.write(`${describeError(error)}\n\n`);
    stdout.write(`${DIM}Continuing without a refreshed knowledge base index.${RESET}\n\n`);
  }

  const agent = mastra.getAgent("assistantAgent");
  const rl = readline.createInterface({ input: stdin, output: stdout });

  // While a generation is in flight, Ctrl+C cancels just that turn instead
  // of killing the process — an idle Ctrl+C exits normally via rl.close().
  let activeReader: ReadableStreamDefaultReader<unknown> | null = null;
  let cancelledByUser = false;
  rl.on("SIGINT", () => {
    if (activeReader) {
      cancelledByUser = true;
      activeReader.cancel("user-interrupted").catch(() => {});
      return;
    }
    rl.close();
  });

  // `close` fires both for intentional exits (rl.close() below) and for
  // stdin hitting EOF (e.g. piped input) — which can happen mid-generation.
  // Defer the actual process exit until the in-flight turn's `finally`
  // block runs, so a response in progress always gets to finish printing.
  let isGenerating = false;
  let exitPending = false;
  function exitCleanly(): void {
    if (isGenerating) {
      exitPending = true;
      return;
    }
    stdout.write(`\n${DIM}Goodbye!${RESET}\n`);
    process.exit(0);
  }
  rl.on("close", exitCleanly);

  let exitRequested = false;
  while (!exitRequested) {
    let input: string;
    try {
      input = await rl.question(`${BOLD}${CYAN}You:${RESET} `);
    } catch {
      // rl was closed (e.g. Ctrl+C at the prompt) — let the "close" handler exit.
      break;
    }

    const prompt = input.trim();
    if (!prompt) continue;
    if (EXIT_COMMANDS.has(prompt.toLowerCase())) {
      exitRequested = true;
      rl.close();
      break;
    }

    let stopSpinner: (() => void) | null = startSpinner("thinking");
    let assistantLabelPrinted = false;

    try {
      const output = await agent.stream(prompt, {
        memory: {
          thread: CONFIG.memoryThreadId,
          resource: CONFIG.memoryResourceId,
        },
        modelSettings: {
          temperature: 0,
          maxOutputTokens: CONFIG.maxOutputTokens,
        },
      });

      isGenerating = true;
      const reader = output.fullStream.getReader();
      activeReader = reader as ReadableStreamDefaultReader<unknown>;
      cancelledByUser = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        if (value.type === "text-delta") {
          if (stopSpinner) {
            stopSpinner();
            stopSpinner = null;
          }
          if (!assistantLabelPrinted) {
            stdout.write(`${BOLD}${GREEN}Assistant:${RESET} `);
            assistantLabelPrinted = true;
          }
          stdout.write(value.payload.text);
        } else if (value.type === "tool-call") {
          if (stopSpinner) {
            stopSpinner();
            stopSpinner = null;
          }
          stdout.write(`${DIM}↳ using ${value.payload.toolName}...${RESET}\n`);
          stopSpinner = startSpinner("thinking");
        }
      }

      if (cancelledByUser) {
        stdout.write(`${YELLOW}⏹ Generation cancelled.${RESET}`);
      }
    } catch (error) {
      if (stopSpinner) stopSpinner();
      if (!assistantLabelPrinted) stdout.write(`${BOLD}${RED}Error:${RESET} `);
      stdout.write(`\n${RED}${describeError(error)}${RESET}`);
    } finally {
      if (stopSpinner) stopSpinner();
      activeReader = null;
      isGenerating = false;
      stdout.write("\n\n");
      if (exitPending) exitCleanly();
    }
  }
}

main().catch((error) => {
  console.error(`${RED}Fatal error:${RESET}`, error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
