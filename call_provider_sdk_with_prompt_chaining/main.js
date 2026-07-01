const fs = require("fs");
const path = require("path");

const CATEGORIES = [
  "Account Opening",
  "Billing Issue",
  "Account Access",
  "Transaction Inquiry",
  "Card Services",
  "Account Statement",
  "Loan Inquiry",
  "General Information",
];

const PROMPT_CHAIN = [
  ["1. Interpret Intent", "01_interpret_intent.txt"],
  ["2. Map Possible Categories", "02_map_categories.txt"],
  ["3. Choose Best Category", "03_choose_category.txt"],
  ["4. Extract Additional Details", "04_extract_details.txt"],
  ["5. Generate Short Response", "05_generate_response.txt"],
];

function loadPrompt(promptDir, filename) {
  return fs.readFileSync(path.join(promptDir, filename), "utf8").trim();
}

function buildStageInput(customerQuery, previousOutputs) {
  const lines = [
    "Customer query:",
    customerQuery,
    "",
    "Allowed categories:",
    CATEGORIES.join(", "),
  ];

  if (previousOutputs.length > 0) {
    lines.push("", "Previous chain outputs:");
    for (const [stageName, output] of previousOutputs) {
      lines.push(`${stageName}:`, output, "");
    }
  }

  return lines.join("\n").trim();
}

async function callOpenRouter(apiKey, modelName, prompt, userContent) {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: modelName,
      messages: [
        { role: "system", content: prompt },
        { role: "user", content: userContent },
      ],
      temperature: 0.2,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`OpenRouter request failed (${response.status}): ${errorBody}`);
  }

  const data = await response.json();
  return data.choices[0].message.content.trim();
}

async function main() {
  if (process.argv.length < 3) {
    console.error('Usage: node main.js "customer query"');
    process.exitCode = 1;
    return;
  }

  require("dotenv").config({ path: path.join(__dirname, ".env") });

  const apiKey = process.env.OPENROUTER_API_KEY;
  const modelName = process.env.MODEL_NAME;

  if (!apiKey) {
    console.error("Missing OPENROUTER_API_KEY in environment or .env file.");
    process.exitCode = 1;
    return;
  }

  if (!modelName) {
    console.error("Missing MODEL_NAME in environment or .env file.");
    process.exitCode = 1;
    return;
  }

  const customerQuery = process.argv.slice(2).join(" ").trim();
  const promptDir = path.join(__dirname, "prompts");
  const previousOutputs = [];

  for (const [stageName, promptFile] of PROMPT_CHAIN) {
    const prompt = loadPrompt(promptDir, promptFile);
    const stageInput = buildStageInput(customerQuery, previousOutputs);
    const output = await callOpenRouter(apiKey, modelName, prompt, stageInput);

    console.log(`\n===== ${stageName} =====`);
    console.log(output);

    previousOutputs.push([stageName, output]);
  }

  console.log("\n===== FINAL RESPONSE =====");
  console.log(previousOutputs[previousOutputs.length - 1][1]);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
