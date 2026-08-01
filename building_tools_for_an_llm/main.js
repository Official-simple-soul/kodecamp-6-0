import path from 'path';
import { fileURLToPath } from 'url';

import dotenv from 'dotenv';
import OpenAI from 'openai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, '.env') });

const openRouterApiKey = process.env.OPENROUTER_API_KEY;
const geminiApiKey = process.env.GEMINI_API_KEY;
const modelName = process.env.LLM_MODEL_NAME || 'openai/gpt-4o-mini';
const maxOutputTokens = Number(process.env.MAX_OUTPUT_TOKENS) || 2000;

if (!openRouterApiKey && !geminiApiKey) {
  console.error('Missing OPENROUTER_API_KEY or GEMINI_API_KEY in .env file.');
  process.exit(1);
}

const client = new OpenAI({
  apiKey: openRouterApiKey || geminiApiKey,
  baseURL: 'https://openrouter.ai/api/v1',
  defaultHeaders: {
    'HTTP-Referer': 'http://localhost',
    'X-Title': 'Building Tools for an LLM',
  },
});

const requiredToolNames = [
  'get_flight_schedule',
  'get_hotel_booking_schedule',
  'convert_currency',
];

const tools = [
  {
    type: 'function',
    function: {
      name: 'get_flight_schedule',
      description:
        'Return a round-trip flight schedule between two cities and the total flight cost in USD.',
      parameters: {
        type: 'object',
        properties: {
          origin: { type: 'string', description: 'Origin city or airport' },
          destination: {
            type: 'string',
            description: 'Destination city or airport',
          },
          trip_type: {
            type: 'string',
            description: 'Trip type',
            enum: ['round_trip'],
          },
        },
        required: ['origin', 'destination', 'trip_type'],
        additionalProperties: false,
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_hotel_booking_schedule',
      description:
        'Return a hotel booking schedule for a stay and the total hotel cost in USD.',
      parameters: {
        type: 'object',
        properties: {
          city: { type: 'string', description: 'City for the stay' },
          nights: { type: 'integer', description: 'Number of nights' },
        },
        required: ['city', 'nights'],
        additionalProperties: false,
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'convert_currency',
      description:
        'Convert an amount from one currency to another using a deterministic exchange rate table.',
      parameters: {
        type: 'object',
        properties: {
          amount: { type: 'number', description: 'Amount to convert' },
          from_currency: {
            type: 'string',
            description: 'Source currency code',
          },
          to_currency: { type: 'string', description: 'Target currency code' },
        },
        required: ['amount', 'from_currency', 'to_currency'],
        additionalProperties: false,
      },
    },
  },
];

function getFlightSchedule({ origin, destination }) {
  return {
    route: `${origin} to ${destination}`,
    currency: 'USD',
    outbound: {
      departure: `${origin} 08:10`,
      arrival: `${destination} 14:10`,
      duration_hours: 6,
      price_usd: 420,
      stopovers: 0,
    },
    return: {
      departure: `${destination} 18:40`,
      arrival: `${origin} 00:40`,
      duration_hours: 6,
      price_usd: 430,
      stopovers: 0,
    },
    total_flight_time_hours: 12,
    total_flight_cost_usd: 850,
    note: 'Assumed direct economy round trip for conference travel.',
  };
}

function getHotelBookingSchedule({ city, nights }) {
  const nightlyRateUsd = 175;
  const totalHotelCostUsd = nightlyRateUsd * nights;

  return {
    city,
    currency: 'USD',
    stay_nights: nights,
    check_in: `${city} Day 1`,
    check_out: `${city} Day ${nights + 1}`,
    hotel_name: `${city} Conference Stay`,
    nightly_rate_usd: nightlyRateUsd,
    total_hotel_cost_usd: totalHotelCostUsd,
    note: 'Assumed mid-range conference hotel rate.',
  };
}

function convertCurrency({ amount, from_currency, to_currency }) {
  const rateTable = {
    USD_NGN: 1600,
    NGN_USD: 1 / 1600,
  };

  const key = `${from_currency.toUpperCase()}_${to_currency.toUpperCase()}`;
  const rate = rateTable[key];

  if (!rate) {
    throw new Error(
      `Unsupported currency pair: ${from_currency} to ${to_currency}`,
    );
  }

  const convertedAmount = amount * rate;

  return {
    from_currency: from_currency.toUpperCase(),
    to_currency: to_currency.toUpperCase(),
    amount,
    rate,
    converted_amount: Number(convertedAmount.toFixed(2)),
  };
}

function runTool(name, args) {
  switch (name) {
    case 'get_flight_schedule':
      return getFlightSchedule(args);
    case 'get_hotel_booking_schedule':
      return getHotelBookingSchedule(args);
    case 'convert_currency':
      return convertCurrency(args);
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

function buildMessages() {
  return [
    {
      role: 'system',
      content:
        'You are a travel logistics assistant. You must call all three tools at least once before answering: get_flight_schedule, get_hotel_booking_schedule, and convert_currency. Use the flight and hotel tools to compute total cost in USD. Then use convert_currency to convert the final total from USD to NGN. After the tool calls are complete, give a concise answer with the total round-trip flight time, the total logistics cost in USD, and the NGN equivalent. If any tool output is missing, continue the conversation until you have all the data.',
    },
    {
      role: 'user',
      content:
        "I'm taking a flight from Lagos to Nairobi for a conference. I would like to know the total flight time back and forth, and the total cost of logistics for this conference if I'm staying for three days.",
    },
  ];
}

async function completeConversation() {
  const messages = buildMessages();
  const usedTools = new Set();
  const maxTurns = 8;

  for (let turn = 0; turn < maxTurns; turn += 1) {
    const response = await client.chat.completions.create({
      model: modelName,
      messages,
      tools,
      tool_choice: 'auto',
      temperature: 0,
      max_tokens: maxOutputTokens,
    });

    const assistantMessage = response.choices[0].message;
    messages.push({
      role: 'assistant',
      content: assistantMessage.content ?? null,
      tool_calls: assistantMessage.tool_calls ?? undefined,
    });

    const toolCalls = assistantMessage.tool_calls || [];

    if (toolCalls.length === 0) {
      const hasAllTools = requiredToolNames.every((name) =>
        usedTools.has(name),
      );
      if (hasAllTools && assistantMessage.content) {
        return assistantMessage.content;
      }

      const missingTools = requiredToolNames.filter(
        (name) => !usedTools.has(name),
      );
      messages.push({
        role: 'user',
        content: `You still need to use these tools before answering: ${missingTools.join(
          ', ',
        )}. Call them now and then finish the response.`,
      });
      continue;
    }

    for (const toolCall of toolCalls) {
      const toolName = toolCall.function.name;
      const toolArgs = JSON.parse(toolCall.function.arguments || '{}');
      const toolResult = runTool(toolName, toolArgs);
      usedTools.add(toolName);

      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        content: JSON.stringify(toolResult),
      });
    }
  }

  throw new Error(
    'The model did not complete the tool-calling conversation in time.',
  );
}

async function main() {
  try {
    const finalResponse = await completeConversation();
    process.stdout.write(`${finalResponse}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(message);
    process.exitCode = 1;
  }
}

main();
