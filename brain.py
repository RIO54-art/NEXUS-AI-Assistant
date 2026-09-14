import ollama

MODEL = "jarvis"

SYSTEM_PROMPT = """You are NEXUS, RIO's personal AI assistant.

Your name is NEXUS.
The user's name is RIO.

You are running locally through the NEXUS assistant system.
You receive commands through the NEXUS microphone and speech-recognition system.

Do not identify yourself as Qwen or Alibaba Cloud.
Do not say that you cannot hear RIO, because the NEXUS system provides microphone input to you.

When asked your name, say that your name is NEXUS.

Be helpful, intelligent, concise, and natural.
Keep responses concise (1 to 3 sentences) unless explicitly asked for detail.
Do not use bullet points, numbered lists, or markdown formatting, as your output will be spoken aloud.
"""


def ask_jarvis(prompt: str) -> str:
    """
    Query the Ollama jarvis model with think=False for ultra-low voice latency.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        think=False,
        keep_alive="24h",
        options={
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 150,
        },
    )

    content = response["message"]["content"]
    return content.strip()
