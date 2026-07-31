# Building Tools for an LLM

This task demonstrates function calling with an LLM using the OpenAI SDK and an OpenRouter-compatible model.

## What it does

- Provides a flight booking schedule tool with USD pricing
- Provides a hotel booking schedule tool with USD pricing
- Provides a currency conversion tool
- Sends the required prompt to the LLM
- Completes the full tool-calling conversation
- Prints the final LLM response to standard output

## Files

- `main.js` - CLI entry point, tool definitions, and conversation loop
- `package.json` - dependencies and scripts
- `.env.example` - example environment variables

## Setup

Install dependencies:

```bash
npm install
```

Create a `.env` file in the same folder as `main.js`:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL_NAME=openai/gpt-4o-mini
```

## Run

```bash
node main.js
```

The program prints the final response from the LLM after the tool calls complete.
