"""Agent definitions for debate layer."""

from __future__ import annotations

from typing import Dict, List

from llm_backends import LLMBackend


class Agent:
    """A role-specific LLM agent."""

    def __init__(self, name: str, role_prompt: str, backend: LLMBackend) -> None:
        self.name = name
        self.role_prompt = role_prompt
        self.backend = backend

    def run(self, global_prompt: str, user_prompt: str) -> str:
        """
        Run a single turn for this agent.

        Args:
            global_prompt: Shared system context.
            user_prompt: Turn-specific user prompt.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": global_prompt},
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.backend.chat(messages)
