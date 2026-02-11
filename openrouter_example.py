import os
import requests
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENROUTER_API_KEY in environment (.env)")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
payload = {
    "model": "openai/gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
}

response = requests.post(url, headers=headers, json=payload, timeout=30)
response.raise_for_status()

print(response.json()["choices"][0]["message"]["content"])
