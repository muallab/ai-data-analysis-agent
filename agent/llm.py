from __future__ import annotations
import os
from typing import List
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # load .env at import time

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")

@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

class LLMClient:
    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_MODEL
        self.client = OpenAI()  # picks up OPENAI_API_KEY from env

    def chat(self, messages: List[ChatMessage], temperature: float = 0.2) -> str:
        """
        Minimal chat wrapper. Returns assistant text.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return resp.choices[0].message.content.strip()

# convenience singleton
_client: LLMClient | None = None

def get_client(model: str | None = None) -> LLMClient:
    global _client
    if _client is None or (model and model != _client.model):
        _client = LLMClient(model=model)
    return _client