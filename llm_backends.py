"""LLM backend abstractions for debate layer."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Dict, List

import requests


class LLMBackend(ABC):
    """Abstract backend interface."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Execute chat completion and return the response text."""
        raise NotImplementedError


class GeminiBackend(LLMBackend):
    """Google Gemini backend using google-generativeai."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key
        self.model_name = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("google-generativeai package is required for GeminiBackend") from exc

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        # Flatten messages into a single prompt for simplicity.
        prompt_parts = []
        for m in messages:
            prefix = m.get("role", "user")
            prompt_parts.append(f"[{prefix.upper()}]\n{m.get('content','')}")
        prompt = "\n\n".join(prompt_parts)
        resp = model.generate_content(prompt)
        return resp.text or ""


class LocalLLMBackend(LLMBackend):
    """Local LLM backend (e.g., Ollama / vLLM HTTP endpoint)."""

    def __init__(
        self,
        model: str,
        endpoint: str = "http://localhost:11434/api/chat",
        soft_fallback: bool = False,
    ) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.soft_fallback = soft_fallback

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:  # pragma: no cover - network issues
            if self.soft_fallback:
                # Graceful fallback so UI continues even if local LLM is down.
                return (
                    "[LOCAL_LLM_FALLBACK] 로컬 LLM 엔드포인트에 연결할 수 없어 기본 응답을 반환합니다. "
                    "LOCAL_LLM_ENDPOINT/LOCAL_LLM_MODEL 설정 또는 서버 실행 상태를 확인하세요. "
                    f"에러: {exc}"
                )
            raise RuntimeError(
                f"Local LLM endpoint 오류. 서버가 실행 중인지 확인하거나 LOCAL_LLM_ENDPOINT를 설정하세요. ({exc})"
            ) from exc

        data = resp.json()
        # Ollama-style response: {"message": {"content": "..."}}
        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "")
        # Fallback generic
        return data.get("content") or json.dumps(data)
