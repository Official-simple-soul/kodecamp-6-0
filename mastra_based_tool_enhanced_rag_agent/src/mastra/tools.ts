import { createTool } from '@mastra/core/tools';
import { z } from 'zod';

import { CONFIG } from './config.js';
import {
  formatRetrievedContext,
  hybridRetrieveDocuments,
} from './rag/store.js';

const EXCHANGE_RATES: Record<string, number> = {
  USD_NGN: 1600,
  NGN_USD: 1 / 1600,
};

export const getFlightScheduleTool = createTool({
  id: 'get_flight_schedule',
  description:
    'Return a round-trip flight schedule between two cities and the total flight cost in USD.',
  inputSchema: z.object({
    origin: z.string().describe('Origin city or airport'),
    destination: z.string().describe('Destination city or airport'),
    trip_type: z.enum(['round_trip']).describe('Trip type'),
  }),
  execute: async ({ origin, destination }) => ({
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
  }),
});

export const getHotelBookingScheduleTool = createTool({
  id: 'get_hotel_booking_schedule',
  description:
    'Return a hotel booking schedule for a stay and the total hotel cost in USD.',
  inputSchema: z.object({
    city: z.string().describe('City for the stay'),
    nights: z.number().int().positive().describe('Number of nights'),
  }),
  execute: async ({ city, nights }) => {
    const nightlyRateUsd = 175;
    return {
      city,
      currency: 'USD',
      stay_nights: nights,
      check_in: `${city} Day 1`,
      check_out: `${city} Day ${nights + 1}`,
      hotel_name: `${city} Conference Stay`,
      nightly_rate_usd: nightlyRateUsd,
      total_hotel_cost_usd: nightlyRateUsd * nights,
      note: 'Assumed mid-range conference hotel rate.',
    };
  },
});

export const convertCurrencyTool = createTool({
  id: 'convert_currency',
  description:
    'Convert an amount from one currency to another using a deterministic exchange rate table.',
  inputSchema: z.object({
    amount: z.number().describe('Amount to convert'),
    from_currency: z.string().describe('Source currency code'),
    to_currency: z.string().describe('Target currency code'),
  }),
  execute: async ({ amount, from_currency, to_currency }) => {
    const key = `${from_currency.toUpperCase()}_${to_currency.toUpperCase()}`;
    const rate = EXCHANGE_RATES[key];
    if (!rate) {
      return {
        error: `Unsupported currency pair: ${from_currency} to ${to_currency}`,
      };
    }
    return {
      from_currency: from_currency.toUpperCase(),
      to_currency: to_currency.toUpperCase(),
      amount,
      rate,
      converted_amount: Number((amount * rate).toFixed(2)),
    };
  },
});

export const queryInternalKnowledgeBaseTool = createTool({
  id: 'query_internal_knowledge_base',
  description:
    'Search the internal knowledge base built from files in the data/ folder using hybrid ' +
    '(semantic + keyword) retrieval. Use this whenever the user asks about internal, ' +
    'company-specific, or document-based information that general knowledge would not cover.',
  inputSchema: z.object({
    query: z
      .string()
      .describe('The search query to look up in the internal knowledge base'),
    top_k: z
      .number()
      .int()
      .positive()
      .optional()
      .describe('Number of results to return'),
  }),
  execute: async ({ query, top_k }) => {
    try {
      const items = await hybridRetrieveDocuments(query, top_k || CONFIG.topK);
      return { result: formatRetrievedContext(items) };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        error: `Internal knowledge base is currently unavailable: ${message}`,
      };
    }
  },
});

export const allTools = {
  get_flight_schedule: getFlightScheduleTool,
  get_hotel_booking_schedule: getHotelBookingScheduleTool,
  convert_currency: convertCurrencyTool,
  query_internal_knowledge_base: queryInternalKnowledgeBaseTool,
};
