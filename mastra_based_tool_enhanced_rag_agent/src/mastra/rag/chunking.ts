import mammoth from "mammoth";
// pdf-parse ships no ESM/CJS types that resolve cleanly under NodeNext; the
// default export is the actual parse function at runtime.
// @ts-expect-error -- no bundled type declarations for this entry point
import pdfParse from "pdf-parse";

export function stripNoise(text: string): string {
  return String(text || "")
    .replace(/\u0000/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function sentenceSegments(text: string): string[] {
  const normalized = String(text || "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .trim();
  if (!normalized) return [];

  if (typeof Intl !== "undefined" && "Segmenter" in Intl) {
    const segmenter = new Intl.Segmenter("en", { granularity: "sentence" });
    return Array.from(segmenter.segment(normalized), (segment) => segment.segment.trim()).filter(
      Boolean,
    );
  }

  return normalized
    .split(/(?<=[.!?])\s+|\n+/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function splitLongSegment(segment: string, maxLength: number): string[] {
  const words = segment.split(/\s+/).filter(Boolean);
  const chunks: string[] = [];
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
  if (current) chunks.push(current);
  return chunks;
}

/** Sentence-aware chunking: keeps sentences together up to `maxLength`, splitting only oversized sentences. */
export function buildSemanticChunks(text: string, maxLength: number): string[] {
  const segments = sentenceSegments(text);
  const chunks: string[] = [];
  let current = "";

  for (const segment of segments) {
    if (segment.length > maxLength) {
      if (current) {
        chunks.push(current);
        current = "";
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

export async function extractTextFromFile(buffer: Buffer, fileName: string): Promise<string> {
  const lower = fileName.toLowerCase();

  if (lower.endsWith(".pdf")) {
    const parsed = await pdfParse(buffer);
    return stripNoise(parsed.text);
  }

  if (lower.endsWith(".docx")) {
    const parsed = await mammoth.extractRawText({ buffer });
    return stripNoise(parsed.value);
  }

  return stripNoise(buffer.toString("utf8"));
}

export function tokenize(text: string): string[] {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

export function lexicalOverlapScore(queryTokens: string[], text: string): number {
  if (queryTokens.length === 0) return 0;
  const docTokens = new Set(tokenize(text));
  if (docTokens.size === 0) return 0;
  const matches = queryTokens.filter((token) => docTokens.has(token)).length;
  return matches / queryTokens.length;
}
