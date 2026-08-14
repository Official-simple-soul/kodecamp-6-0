import { InferenceClient } from "@huggingface/inference";

import { CONFIG } from "../config.js";

// Pin the provider explicitly: leaving it unset makes the HF client resolve
// "auto" on every call and log its provider choice to stdout, which would
// pollute the CLI's streamed output.
const hfClient = new InferenceClient(CONFIG.hfApiKey);

function meanPoolTokenEmbeddings(tokenEmbeddings: unknown): number[] {
  if (!Array.isArray(tokenEmbeddings) || tokenEmbeddings.length === 0) {
    throw new Error("Empty HuggingFace embedding response.");
  }

  if (typeof tokenEmbeddings[0] === "number") {
    return tokenEmbeddings as number[];
  }

  const vectors = (tokenEmbeddings as unknown[]).filter(
    (row): row is number[] => Array.isArray(row) && row.every((value) => typeof value === "number"),
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

function normalizeVector(vector: number[]): number[] {
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (!magnitude) return vector;
  return vector.map((value) => value / magnitude);
}

export async function embedText(text: string): Promise<number[]> {
  try {
    const result = await hfClient.featureExtraction({
      model: CONFIG.embedModelName,
      inputs: text,
      provider: "hf-inference",
    });
    return normalizeVector(meanPoolTokenEmbeddings(result));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`HuggingFace embeddings failed: ${message}`);
  }
}
