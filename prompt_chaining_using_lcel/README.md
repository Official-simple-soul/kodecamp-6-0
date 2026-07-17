# Prompt Chaining using LCEL

This task implements a five-stage software requirements analysis chain using LangChain LCEL.

## What it does

The script:

1. Interprets the client's project request.
2. Identifies possible project categories.
3. Selects the best category.
4. Extracts missing requirements.
5. Generates an initial project assessment.

## Files

- `main.js` - CLI entry point and LCEL workflow.
- `prompts/01_interpret_request.txt` - Stage 1 prompt.
- `prompts/02_identify_possible_categories.txt` - Stage 2 prompt.
- `prompts/03_select_best_category.txt` - Stage 3 prompt.
- `prompts/04_extract_missing_requirements.txt` - Stage 4 prompt.
- `prompts/05_generate_initial_assessment.txt` - Stage 5 prompt.
- `package.json` - Dependencies and scripts.
- `.env.example` - Example environment variables.

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

## Usage

```bash
node main.js "I need a web platform for schools to manage attendance, homework, and parent communication"
```

The script prints the output from each stage before moving to the next stage, then prints the final project assessment at the end.
