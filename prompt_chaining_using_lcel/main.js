import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import dotenv from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableLambda, RunnableSequence } from '@langchain/core/runnables';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, '.env') });

const PROMPT_DIR = path.join(__dirname, 'prompts');
const CATEGORIES = [
  'Web Application',
  'Mobile Application',
  'API / Backend Service',
  'Data Analytics Platform',
  'AI / Machine Learning System',
  'E-Commerce Platform',
  'Enterprise Management System',
  'System Integration',
  'DevOps / Infrastructure Automation',
  'General Software Project',
];

function loadPrompt(filename) {
  return fs.readFileSync(path.join(PROMPT_DIR, filename), 'utf8').trim();
}

function buildCategoriesList() {
  return CATEGORIES.map((category, index) => `${index + 1}. ${category}`).join(
    '\n',
  );
}

function createStageChain(templateText, llm) {
  return PromptTemplate.fromTemplate(templateText)
    .pipe(llm)
    .pipe(new StringOutputParser());
}

function createStageRunner({ name, chain, updateState }) {
  return RunnableLambda.from(async (state) => {
    const output = await chain.invoke(state);
    console.log(`\n===== ${name} =====`);
    console.log(output);
    return updateState(state, output);
  });
}

async function main() {
  const clientDescription = process.argv.slice(2).join(' ').trim();

  if (!clientDescription) {
    console.error('Usage: node main.js "client project description"');
    process.exitCode = 1;
    return;
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  const modelName = process.env.MODEL_NAME;

  if (!apiKey) {
    console.error('Missing OPENROUTER_API_KEY in .env or environment.');
    process.exitCode = 1;
    return;
  }

  if (!modelName) {
    console.error('Missing MODEL_NAME in .env or environment.');
    process.exitCode = 1;
    return;
  }

  const llm = new ChatOpenAI({
    apiKey,
    model: modelName,
    temperature: 0,
    configuration: {
      baseURL: 'https://openrouter.ai/api/v1',
      defaultHeaders: {
        'HTTP-Referer': 'https://openai.com',
        'X-Title': 'Prompt Chaining using LCEL',
      },
      fetch: globalThis.fetch,
    },
  });

  const categoriesList = buildCategoriesList();

  const stage1Chain = createStageChain(
    loadPrompt('01_interpret_request.txt'),
    llm,
  );
  const stage2Chain = createStageChain(
    loadPrompt('02_identify_possible_categories.txt'),
    llm,
  );
  const stage3Chain = createStageChain(
    loadPrompt('03_select_best_category.txt'),
    llm,
  );
  const stage4Chain = createStageChain(
    loadPrompt('04_extract_missing_requirements.txt'),
    llm,
  );
  const stage5Chain = createStageChain(
    loadPrompt('05_generate_initial_assessment.txt'),
    llm,
  );

  const workflow = RunnableSequence.from([
    createStageRunner({
      name: 'Stage 1: Interpret the Project Request',
      chain: stage1Chain,
      updateState: (state, output) => ({
        ...state,
        stage1_output: output,
      }),
    }),
    createStageRunner({
      name: 'Stage 2: Identify Possible Project Categories',
      chain: stage2Chain,
      updateState: (state, output) => ({
        ...state,
        stage2_output: output,
      }),
    }),
    createStageRunner({
      name: 'Stage 3: Select the Best Category',
      chain: stage3Chain,
      updateState: (state, output) => ({
        ...state,
        stage3_output: output,
      }),
    }),
    createStageRunner({
      name: 'Stage 4: Extract Missing Requirements',
      chain: stage4Chain,
      updateState: (state, output) => ({
        ...state,
        stage4_output: output,
      }),
    }),
    createStageRunner({
      name: 'Stage 5: Generate Initial Assessment',
      chain: stage5Chain,
      updateState: (state, output) => ({
        ...state,
        final_assessment: output,
      }),
    }),
  ]);

  const initialState = {
    client_description: clientDescription,
    categories_list: categoriesList,
  };

  const finalState = await workflow.invoke(initialState);

  console.log('\n===== FINAL PROJECT ASSESSMENT =====');
  console.log(finalState.final_assessment);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
