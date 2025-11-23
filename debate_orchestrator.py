"""Debate orchestrator for multi-agent LLM discussion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from agents import Agent
from prompts import (
    DEBATE_ROUND_PROMPT_TEMPLATE,
    FINAL_REPORT_PROMPT,
    GLOBAL_SYSTEM_PROMPT,
)


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Parse JSON with graceful fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _normalize_round_response(raw_resp: Any) -> Dict[str, Any]:
    """
    Normalize agent response into a consistent dict so UI doesn't show None.
    Falls back to treating raw text as the position.
    """
    if isinstance(raw_resp, dict):
        position = raw_resp.get("position") or raw_resp.get("raw") or ""
        agreements = raw_resp.get("agreements") or []
        disagreements = raw_resp.get("disagreements") or []
        questions = raw_resp.get("questions_to_others") or raw_resp.get("questions") or []
    else:
        position, agreements, disagreements, questions = str(raw_resp or ""), [], [], []

    # Ensure list types
    def _as_list(val: Any) -> List[str]:
        if isinstance(val, list):
            return [str(x) for x in val if x is not None]
        if val is None:
            return []
        return [str(val)]

    return {
        "position": position,
        "agreements": _as_list(agreements),
        "disagreements": _as_list(disagreements),
        "questions_to_others": _as_list(questions),
    }


@dataclass
class DebateSession:
    """Manage multi-round debate among agents."""

    agents: Dict[str, Agent]
    global_prompt: str = GLOBAL_SYSTEM_PROMPT
    round_history: List[Dict[str, Any]] = field(default_factory=list)

    def run_round(self, data_summary: str, prev_round_summary: str) -> Dict[str, Dict[str, Any]]:
        """Run one debate round."""
        round_result: Dict[str, Dict[str, Any]] = {}
        for name, agent in self.agents.items():
            user_prompt = DEBATE_ROUND_PROMPT_TEMPLATE.format(
                data_summary=data_summary,
                round_summary=prev_round_summary,
                role_name=name,
            )
            raw = agent.run(self.global_prompt, user_prompt)
            parsed = _safe_json_loads(raw)
            round_result[name] = _normalize_round_response(parsed)
        # build simple summary for next round (positions only)
        summary_items = []
        for n, res in round_result.items():
            pos = res.get("position") if isinstance(res, dict) else None
            summary_items.append(f"{n}: {pos}")
        round_summary = "; ".join(summary_items)
        self.round_history.append({"round_result": round_result, "round_summary": round_summary})
        return round_result

    def run_multi_rounds(self, data_summary: str, num_rounds: int = 2) -> Dict[str, Any]:
        """Execute multiple rounds and collect results."""
        prev_summary = ""
        all_rounds: List[Dict[str, Dict[str, Any]]] = []
        for _ in range(num_rounds):
            res = self.run_round(data_summary=data_summary, prev_round_summary=prev_summary)
            all_rounds.append(res)
            prev_summary = self.round_history[-1]["round_summary"]
        return {"rounds": all_rounds, "history": self.round_history}

    def build_report_input(self) -> str:
        """Flatten debate history into a summary string for REPORT_WRITER."""
        parts: List[str] = []
        for idx, item in enumerate(self.round_history, start=1):
            parts.append(f"[Round {idx}] {item.get('round_summary','')}")
        return "\n".join(parts)


def generate_final_report(
    session: DebateSession,
    corp_name: str,
    corp_code: str,
    years_range: str,
    industry: str,
    extra_info: str,
    data_summary: str,
) -> str:
    """
    Ask REPORT_WRITER agent to craft final Markdown report.

    Args:
        session: DebateSession containing agents with REPORT role.
        extra_info: Additional context (e.g., filings).
    """
    report_agent = session.agents.get("REPORT_WRITER")
    if report_agent is None:
        raise ValueError("REPORT_WRITER agent not provided.")

    debate_summary = session.build_report_input()
    user_prompt = FINAL_REPORT_PROMPT.format(
        corp_name=corp_name,
        corp_code=corp_code,
        years_range=years_range,
        industry=industry,
        data_summary=data_summary + "\n" + extra_info,
        debate_summary=debate_summary,
    )
    return report_agent.run(session.global_prompt, user_prompt)
