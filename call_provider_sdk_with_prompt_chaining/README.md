# Bank Customer Support Prompt Chain

This project implements a five-step prompt chain for classifying and responding to bank customer support queries.

## Files

- `main.js` - command-line script that runs the prompt chain.
- `prompts/01_interpret_intent.txt` - interprets the customer's intent.
- `prompts/02_map_categories.txt` - maps the query to possible categories.
- `prompts/03_choose_category.txt` - selects the best category.
- `prompts/04_extract_details.txt` - extracts provided and missing details.
- `prompts/05_generate_response.txt` - generates the final customer response.
- `package.json` - JavaScript dependency metadata.
- `.env.example` - example environment variable file.

## Setup

Install dependencies:

```bash
npm install
```

Create a `.env` file in the same folder as `main.js`:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
MODEL_NAME=openai/gpt-4o-mini
```

The `.env` file is ignored by git and should not be committed.

## Usage

```bash
node main.js "I was charged twice for my card payment yesterday"
```

The script prints the LLM response after each prompt stage, then prints the final response clearly at the end.
