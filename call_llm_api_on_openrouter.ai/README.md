# KODECAMP 6.0 AI FOR DEVELOPER TASK 1 (Call an LLM API on Openrouter.ai)

A simple TypeScript command-line tool that sends a user prompt to an AI model through the OpenRouter API and prints the response in the terminal.

## Features

- Accepts a prompt from the command line
- Sends the prompt to an LLM API
- Uses environment variables for secret configuration
- Loads environment variables from a `.env` file
- Does not hardcode API keys
- Limits response length using `max_tokens`

## Project Structure

```bash
.
├── main.ts
├── scripts
│   └── promptLLM.ts
├── package.json
├── .env
└── README.md
```

## Setup

### Install dependencies:

```bash
npm install
```

#### Create .env file in the root and add the below

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
MODEL_NAME=your_model_name_here
API_URL=https://openrouter.ai/api/v1/chat/completions
```

#### Example Model

```bash
MODEL_NAME=openai/gpt-5.2
```

### How to Test

Test with a simple prompt

```bash
npm start -- "What is TypeScript?"
```

Note that in the above, the text in quotation is your prompt. You can always decide what you want to ask
