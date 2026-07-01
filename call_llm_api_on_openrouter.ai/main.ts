import dotenv from 'dotenv';
import { promptLLM } from './scripts/promptLLM.ts';

dotenv.config();

async function main(): Promise<void> {
  const prompt = process.argv.slice(2).join(' ');

  if (!prompt) {
    console.log('Usage: node main.js "Your prompt here"');
    process.exit(1);
  }

  try {
    const result = await promptLLM(prompt);
    console.log(result);
  } catch (error) {
    const err = error instanceof Error ? error.message : error;
    console.error('Error:', err);
  }
}

main();
