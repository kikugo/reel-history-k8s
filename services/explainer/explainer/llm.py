"""Client for the Phase 2 vLLM server on Modal (OpenAI-compatible chat completions).

Each attempt has its own timeout, and all attempts share one deadline, so a cold Modal
container (minutes to start) costs a request at most DEADLINE_S before the template answers.
Timeouts, connection errors, 429 and 5xx are retried with exponential backoff and jitter;
other 4xx are not, since sending the same request again won't fix them.
"""

import asyncio
import random
import re
import time

import httpx

THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


class LLMError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class LLMClient:
    def __init__(self, client: httpx.AsyncClient, base_url: str, api_key: str, model: str,
                 timeout_s: float = 15.0, deadline_s: float = 30.0, retries: int = 2,
                 backoff_s: float = 0.5, max_tokens: int = 200):
        self.client = client
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.deadline_s = deadline_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.max_tokens = max_tokens

    def body(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},  # Qwen3 thinks by default
        }

    async def complete(self, prompt: str) -> str:
        start = time.monotonic()
        reason = "unknown"
        for attempt in range(self.retries + 1):
            remaining = self.deadline_s - (time.monotonic() - start)
            if remaining <= 0:
                raise LLMError("deadline")
            try:
                resp = await self.client.post(
                    self.url, json=self.body(prompt), timeout=min(self.timeout_s, remaining),
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            except httpx.TimeoutException:
                reason = "timeout"
            except httpx.TransportError:
                reason = "connect"
            else:
                if resp.status_code == 200:
                    return self._text(resp)
                if resp.status_code != 429 and resp.status_code < 500:
                    raise LLMError(f"http_{resp.status_code}")
                reason = f"http_{resp.status_code}"
            if attempt < self.retries:
                await asyncio.sleep(self.backoff_s * 2**attempt + random.uniform(0, self.backoff_s))
        raise LLMError(reason)

    @staticmethod
    def _text(resp: httpx.Response) -> str:
        try:
            content = resp.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError("bad_response") from None
        text = THINK_RE.sub("", content).strip()
        if not text:
            raise LLMError("empty")
        return text
