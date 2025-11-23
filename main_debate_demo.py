"""Demo script: fetch DART ratios and run multi-agent debate."""

from __future__ import annotations

import os
from datetime import datetime

from backend.financial_analysis import calc_yoy_changes, get_yearly_ratios
from backend.opendart_client import OpenDartClient
from agents import Agent
from debate_orchestrator import DebateSession, generate_final_report
from llm_backends import GeminiBackend, LocalLLMBackend
from prompts import (
    GLOBAL_SYSTEM_PROMPT,
    ROLE_PROMPT_CAPITAL_STRUCTURE,
    ROLE_PROMPT_EQUITY_ANALYST,
    ROLE_PROMPT_FS_ANALYST,
    ROLE_PROMPT_REPORT_WRITER,
    ROLE_PROMPT_RISK_MANAGER,
)


def build_data_summary(ratios_by_year: dict, yoy_changes: dict) -> str:
    """Create a compact textual summary from ratios."""
    years = sorted(ratios_by_year.keys())
    parts = [f"{years[0]}~{years[-1]} 재무 요약:"]
    for key in ["current_ratio", "debt_ratio", "roa", "roe"]:
        seq = [ratios_by_year[y].get(key) for y in years]
        parts.append(f"- {key}: " + " → ".join(f"{v:.2f}" if v is not None else "NA" for v in seq))
    parts.append("YoY 변화:")
    for key, changes in yoy_changes.items():
        lines = []
        for yr, info in changes.items():
            v = info.get("pct_change")
            lines.append(f"{yr}: {v:.1f}% {info.get('direction')}" if v is not None else f"{yr}: NA")
        parts.append(f"- {key}: " + "; ".join(lines))
    return "\n".join(parts)


def main() -> None:
    target_name = "삼성전자"
    start_year, end_year = 2021, datetime.now().year

    client = OpenDartClient()
    corps = client.find_corp_by_name(target_name, exact=True)
    if not corps:
        raise SystemExit(f"{target_name} not found in corpCode table.")
    corp = corps[0]

    ratios, missing_years = get_yearly_ratios(client, corp_code=corp.corp_code, start_year=start_year, end_year=end_year)
    if not ratios:
        raise SystemExit(f"선택한 기간({start_year}~{end_year})에 제출된 재무제표가 없습니다. DART 제출 연도를 확인하세요.")
    if missing_years:
        print(f"[INFO] DART에서 찾을 수 없는 연도: {', '.join(map(str, missing_years))}")
    yoy = calc_yoy_changes(ratios, ratio_keys=["current_ratio", "debt_ratio", "roa", "roe"])
    data_summary = build_data_summary(ratios, yoy)

    # Choose backend: Gemini if key available, else local.
    if os.getenv("GOOGLE_API_KEY"):
        backend = GeminiBackend(api_key=os.getenv("GOOGLE_API_KEY"), model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    else:
        backend = LocalLLMBackend(model="llama3")

    agents = {
        "FS_ANALYST": Agent("FS_ANALYST", ROLE_PROMPT_FS_ANALYST, backend),
        "CAPITAL_STRUCTURE": Agent("CAPITAL_STRUCTURE", ROLE_PROMPT_CAPITAL_STRUCTURE, backend),
        "EQUITY_ANALYST": Agent("EQUITY_ANALYST", ROLE_PROMPT_EQUITY_ANALYST, backend),
        "RISK_MANAGER": Agent("RISK_MANAGER", ROLE_PROMPT_RISK_MANAGER, backend),
        "REPORT_WRITER": Agent("REPORT_WRITER", ROLE_PROMPT_REPORT_WRITER, backend),
    }

    session = DebateSession(agents=agents, global_prompt=GLOBAL_SYSTEM_PROMPT)
    session.run_multi_rounds(data_summary=data_summary, num_rounds=2)

    report = generate_final_report(
        session=session,
        corp_name=corp.corp_name,
        corp_code=corp.corp_code,
        years_range=f"{start_year}~{end_year}",
        industry="N/A",
        extra_info="",
        data_summary=data_summary,
    )
    print("\n=== 최종 리포트 (Markdown) ===\n")
    print(report)


if __name__ == "__main__":
    main()
