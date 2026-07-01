export async function promptLLM(prompt: string): Promise<string> {
  const apiKey = process.env.OPENROUTER_API_KEY;
  const model = process.env.MODEL_NAME;
  const chatUrl = `${process.env.API_URL}/chat/completions`;

  if (!apiKey) {
    throw new Error('OPENROUTER_API_KEY is missing');
  }

  if (!model) {
    throw new Error('MODEL_NAME is missing');
  }

  if (!chatUrl) {
    throw new Error('API_URL is missing');
  }

  const response = await fetch(chatUrl, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      max_tokens: 500,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data.error?.message || 'Hey, something wrong. Please try again',
    );
  }

  return data.choices[0].message.content;
}
