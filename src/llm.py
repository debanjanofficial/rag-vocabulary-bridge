"""Ollama helper — CPU-only, JSON output, thinking disabled."""

from __future__ import annotations

import json
import re

import ollama

from config import GEN_MODEL

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_BASE_OPTS = {"num_gpu": 0}


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def chat_json(prompt: str, model: str | None = None,
              temperature: float = 0.4, num_predict: int = 600,
              retries: int = 2) -> dict | None:
    model = model or GEN_MODEL
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format="json", think=False,
                options={**_BASE_OPTS, "temperature": temperature,
                           "num_predict": num_predict},
            )
            return json.loads(_strip_think(resp["message"]["content"]))
        except (json.JSONDecodeError, KeyError) as e:
            last_err = e
            prompt += "\n\nReturn ONLY valid JSON. No prose, no markdown."
        except Exception as e:
            last_err = e
            break
    print(f"    [llm] JSON parse failed: {last_err}")
    return None


def chat_text(prompt: str, model: str | None = None,
              temperature: float = 0.4, num_predict: int = 600) -> str:
    model = model or GEN_MODEL
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        think=False,
        options={**_BASE_OPTS, "temperature": temperature, "num_predict": num_predict},
    )
    return _strip_think(resp["message"]["content"])
