"""
Shared chat-completion client for Hugging Face Inference and local Ollama.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Optional


def _normalize_base_url(base_url: Optional[str]) -> str:
    return (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")


def _call_ollama_chat(
    model_id: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    base_url: Optional[str] = None,
    timeout_s: int = 180,
) -> str:
    endpoint = f"{_normalize_base_url(base_url)}/api/chat"
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not connect to Ollama at {endpoint}. "
            f"Ensure 'ollama serve' is running and the model is pulled."
        ) from e

    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected Ollama response shape: {data}")
    return content


def chat_completion(
    model_id: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    provider: str = "auto",
    hf_token: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    Run one chat completion against either HF Inference API or local Ollama.
    """
    if provider == "ollama":
        return _call_ollama_chat(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )

    try:
        from huggingface_hub import InferenceClient
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "huggingface_hub is required only for non-Ollama providers. "
            "Install it with: pip install huggingface_hub"
        ) from e

    client = InferenceClient(provider=provider, api_key=hf_token)
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content
