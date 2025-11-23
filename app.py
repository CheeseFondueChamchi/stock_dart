"""Streamlit dashboard for DART-based financial analysis with multi-agent LLM debate.

Run: `streamlit run app.py`
"""

from __future__ import annotations

import datetime as dt
import os
import re
import importlib
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple
import importlib

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import requests
try:  # Optional dependency
    from pykrx import stock as krx_stock
except ImportError:  # pragma: no cover
    krx_stock = None
from collections import Counter

from agents import Agent
from backend.financial_analysis import (
    calc_debt_structure_summary,
    calc_yoy_changes,
    get_yearly_ratios,
    get_quarterly_ratios,
    calc_quarterly_yoy_changes,
)
from backend.opendart_client import OpenDartClient
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

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

PBLNTF_LABELS = {
    "All": "전체 공시",
    None: "미분류",
    "A": "정기보고서(사업/분기/반기)",
    "B": "주요사항/의사결정",
    "C": "발행공시(유상·주식 발행 등)",
    "D": "지분공시(대량보유 등)",
    "E": "기타공시",
}

# --------------------------------------------------------------------------- #
# LLM helpers
# --------------------------------------------------------------------------- #


def normalize_gemini_model(raw: str) -> str:
    """Normalize Gemini model name to API-accepted format."""
    name = (raw or "").strip()
    if not name:
        return "models/gemini-2.0-flash"
    # If user passes something like models/gemini/gemini-2.5-flash -> take last segment
    if "/" in name:
        name = name.split("/")[-1]
    if not name.startswith("models/"):
        name = f"models/{name}"
    return name


def check_local_llm(endpoint: str) -> Optional[str]:
    """Quick health check for local LLM endpoint; return error message or None."""
    try:
        # Ollama supports GET /api/tags. For OpenAI-style, /v1/models might exist.
        base = endpoint.rsplit("/", 1)[0]
        resp = requests.get(f"{base}/tags", timeout=5)
        if resp.status_code == 404:
            return "로컬 LLM 엔드포인트가 /api/chat 경로를 지원하지 않습니다. LOCAL_LLM_ENDPOINT를 올바른 chat 경로로 설정하세요."
        resp.raise_for_status()
        return None
    except Exception as exc:
        return f"로컬 LLM 엔드포인트에 연결할 수 없습니다: {exc}"


# --------------------------------------------------------------------------- #
# Filing helpers
# --------------------------------------------------------------------------- #

FILING_NAME_PATTERNS = [
    ("정기보고서", [r"분기보고서", r"반기보고서", r"사업보고서"]),
    ("임원/주주 지분 보고", [r"임원[ㆍ·]주요주주특정증권등소유상황보고서", r"최대주주등소유주식변동신고서"]),
    ("자기주식/처분", [r"자기주식처분결과보고서", r"자기주식[취득|처분]"]),
    ("특수관계인 증여", [r"특수관계인에대한증여"]),
    ("기타경영사항(자율공시)", [r"기타경영사항"]),
    ("장래사업·경영계획", [r"장래사업[ㆍ·]경영계획", r"공정공시"]),
    ("배당/결정", [r"현금[ㆍ·]현물배당결정", r"배당"]),
    ("잠정실적/정정", [r"잠정\)실적", r"\[기재정정\].*잠정"]),
]


def categorize_filing(report_nm: str) -> str:
    """Rudimentary categorization by report name keywords."""
    title = report_nm or ""
    for label, patterns in FILING_NAME_PATTERNS:
        for pat in patterns:
            if re.search(pat, title):
                return label
    return "기타/미분류"


# --------------------------------------------------------------------------- #
# Market data helpers (pykrx)
# --------------------------------------------------------------------------- #


def _ensure_pykrx() -> None:
    """Ensure pykrx is importable (lazy import for Streamlit reruns)."""
    global krx_stock  # noqa: PLW0603
    if krx_stock is None:
        try:
            krx_stock = importlib.import_module("pykrx").stock  # type: ignore
        except Exception as exc:
            raise RuntimeError("pykrx가 설치되어 있지 않습니다. `pip install pykrx` 후 다시 시도하세요.") from exc


def fetch_price_short_option(
    stock_code: str,
    days: int = 365,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch OHLCV, short-selling status, and option trading value (market-level) for the last `days`."""
    _ensure_pykrx()
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    price_df = krx_stock.get_market_ohlcv_by_date(start_str, end_str, stock_code)
    short_df = krx_stock.get_shorting_status_by_date(start_str, end_str, stock_code)
    # Option trading value is market-wide sentiment (KOSPI/KOSDAQ options)
    option_df = pd.DataFrame()
    try:
        # Some pykrx versions expect positional args only
        option_df = krx_stock.get_market_trading_value_by_date(start_str, end_str, "옵션")
    except TypeError:
        try:
            option_df = krx_stock.get_market_trading_value_by_date(start_str, end_str)
        except Exception:
            option_df = pd.DataFrame()
    return price_df, short_df, option_df


def summarize_market_data(
    price_df: pd.DataFrame, short_df: pd.DataFrame, option_df: pd.DataFrame
) -> Dict[str, Any]:
    """Compute quick stats for LLM and UI."""
    summary: Dict[str, Any] = {}
    if not price_df.empty:
        closes = price_df["종가"]
        rets = closes.pct_change().dropna()
        summary["price_start"] = float(closes.iloc[0])
        summary["price_end"] = float(closes.iloc[-1])
        summary["price_return_pct"] = float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)
        for window in [20, 60, 120, 252]:
            if len(closes) >= window:
                summary[f"return_{window}d_pct"] = float((closes.iloc[-1] / closes.iloc[-window] - 1) * 100)
        if not rets.empty:
            summary["volatility_252d_pct"] = float(rets.std() * sqrt(252) * 100)
            summary["max_drawdown_pct"] = float(((closes / closes.cummax() - 1).min()) * 100)

    if not short_df.empty:
        # 공매도비중 열이 있으면 사용, 없으면 거래량 비율 계산
        ratio_col = "공매도비중" if "공매도비중" in short_df.columns else None
        if ratio_col:
            short_ratio = short_df[ratio_col].astype(float)
        else:
            vol = short_df["공매도거래량"] if "공매도거래량" in short_df.columns else short_df.get("공매도거래대금")
            total_vol = short_df["거래량"] if "거래량" in short_df.columns else short_df.get("거래대금")
            if vol is not None and total_vol is not None:
                vol_f = vol.astype(float)
                tot_f = total_vol.replace(0, pd.NA).astype(float)
                short_ratio = (vol_f / tot_f) * 100
            else:
                short_ratio = pd.Series(dtype=float)
        if not short_ratio.empty:
            summary["short_ratio_avg_pct"] = float(short_ratio.mean())
            summary["short_ratio_recent_pct"] = float(short_ratio.iloc[-1])

    if not option_df.empty:
        # 최근 20일 옵션 순매수 강도(외인/기관/개인)
        recent_opt = option_df.tail(20)
        for investor in ["개인", "외국인", "기관계"]:
            if investor in recent_opt.columns:
                try:
                    summary[f"option_{investor}_net_recent"] = float(recent_opt[investor].astype(float).sum())
                except Exception:
                    continue
    else:
        summary["option_data_missing"] = True

    return summary

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def init_state() -> None:
    """Initialize session state keys."""
    defaults = {
        "corp_info": None,
        "ratios_by_year": None,
        "yoy_changes": None,
        "debt_summary": None,
        "debt_summary_prev": None,
        "filings": None,
        "debate_result": None,
        "final_report": None,
        "debate_data_summary": None,
        "market_price": None,
        "market_short": None,
        "market_option": None,
        "market_summary": None,
        "quarterly_ratios": None,
        "quarterly_yoy": None,
        "quarterly_missing": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def build_data_summary(ratios_by_year: Dict[int, Dict[str, Any]], yoy_changes: Dict[str, Any]) -> str:
    """Create compact textual summary for LLM debate."""
    if not ratios_by_year:
        return "데이터 없음"
    years = sorted(ratios_by_year.keys())
    parts = [f"{years[0]}~{years[-1]} 재무 요약"]
    for key, label in [
        ("current_ratio", "유동비율"),
        ("debt_ratio", "부채비율"),
        ("roa", "ROA"),
        ("roe", "ROE"),
    ]:
        seq = []
        for y in years:
            val = ratios_by_year[y].get(key)
            seq.append(f"{val:.2f}" if val is not None else "NA")
        parts.append(f"- {label}: " + " → ".join(seq))
    parts.append("전년 대비 변화")
    for key, change in yoy_changes.items():
        lines = []
        for y, info in change.items():
            pct = info.get("pct_change")
            direction = info.get("direction")
            lines.append(f"{y}: {pct:.1f}% {direction}" if pct is not None else f"{y}: NA")
        parts.append(f"- {key}: " + "; ".join(lines))
    return "\n".join(parts)


def build_debate_context(
    ratios_by_year: Dict[int, Dict[str, Any]],
    yoy_changes: Dict[str, Any],
    debt_latest: Optional[Dict[str, Any]],
    debt_prev: Optional[Dict[str, Any]],
    filings: List[Dict[str, Any]],
    market_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Augment ratio summary with 자본구조 변화와 최근 1년 공시 그룹 요약."""
    base = build_data_summary(ratios_by_year, yoy_changes)
    parts = [base]

    # 자본구조/부채 변화 요약
    if debt_latest:
        latest_year = max(ratios_by_year.keys())
        prev_year = latest_year - 1 if ratios_by_year else None

        def fmt(val: Any) -> str:
            return f"{val:,.0f}" if isinstance(val, (int, float)) else "NA"

        def delta(cur: Optional[float], prev: Optional[float]) -> str:
            if cur is None or prev in (None, 0):
                return "N/A"
            diff = cur - prev
            pct = (diff / prev) * 100
            return f"{diff:,.0f} ({pct:+.1f}%)"

        total_liab = debt_latest.get("total_liabilities")
        short = debt_latest.get("short_term_debt")
        long = debt_latest.get("long_term_debt")
        bond = debt_latest.get("bond_balance")

        parts.append(f"\n자본구조/부채 요약 ({latest_year}):")
        parts.append(
            f"- 총부채: {fmt(total_liab)}, 단기부채: {fmt(short)}, 장기부채: {fmt(long)}, 사채/회사채: {fmt(bond)}"
        )

        if debt_prev:
            parts.append(f"- 전년({prev_year}) 대비 총부채 증감: {delta(total_liab, debt_prev.get('total_liabilities'))}")
            parts.append(f"- 단기부채 증감: {delta(short, debt_prev.get('short_term_debt'))}")
            parts.append(f"- 장기부채 증감: {delta(long, debt_prev.get('long_term_debt'))}")

    # 최근 1년 공시 그룹 요약
    if filings:
        ty_counter: Counter = Counter()
        name_counter: Counter = Counter()
        for item in filings:
            ty_counter[item.get("pblntf_ty")] += 1
            name_counter[categorize_filing(item.get("report_nm", ""))] += 1
        parts.append("\n최근 1년 공시 요약 (종류별 건수):")
        for code, count in ty_counter.most_common():
            label = PBLNTF_LABELS.get(code, f"기타({code})")
            parts.append(f"- {label}: {count}건")
        # Top 3 latest filings overall
        latest_top = sorted(filings, key=lambda x: x.get("rcept_dt", ""), reverse=True)[:3]
        if latest_top:
            parts.append("최근 공시 TOP3:")
            for f in latest_top:
                parts.append(f"- {f.get('rcept_dt')}: {f.get('report_nm')}")
        # Top 3 per 공시 유형 (정기/주요/발행/지분/기타)
        parts.append("공시 유형별 최근 3건:")
        filings_sorted = sorted(filings, key=lambda x: x.get("rcept_dt", ""), reverse=True)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for f in filings_sorted:
            code = f.get("pblntf_ty")
            grouped.setdefault(code, []).append(f)
        for code, items in grouped.items():
            label = PBLNTF_LABELS.get(code, f"기타({code})")
            parts.append(f"- {label}:")
            for f in items[:3]:
                parts.append(f"  • {f.get('rcept_dt')}: {f.get('report_nm')}")
        # Name-based category counts to capture 세부 보고서명 패턴
        parts.append("\n보고서명 기반 카테고리(최근 1년) 건수:")
        for cat, count in name_counter.most_common():
            parts.append(f"- {cat}: {count}건")
        parts.append("보고서명 기반 카테고리별 최근 3건:")
        grouped_name: Dict[str, List[Dict[str, Any]]] = {}
        for f in filings_sorted:
            cat = categorize_filing(f.get("report_nm", ""))
            grouped_name.setdefault(cat, []).append(f)
        for cat, items in grouped_name.items():
            parts.append(f"- {cat}:")
            for f in items[:3]:
                parts.append(f"  • {f.get('rcept_dt')}: {f.get('report_nm')}")

    # 주가/공매도/옵션 간단 요약
    if market_summary:
        parts.append("\n주가/공매도/옵션 요약:")
        price_ret = market_summary.get("price_return_pct")
        if price_ret is not None:
            parts.append(f"- 1년 주가 수익률: {price_ret:+.1f}% (시작 {market_summary.get('price_start'):,}, "
                         f"현재 {market_summary.get('price_end'):,})")
        for window in [20, 60, 120, 252]:
            key = f"return_{window}d_pct"
            if key in market_summary:
                parts.append(f"- 최근 {window}일 수익률: {market_summary[key]:+.1f}%")
        if market_summary.get("volatility_252d_pct") is not None:
            parts.append(f"- 연환산 변동성(252d): {market_summary['volatility_252d_pct']:.1f}%")
        if market_summary.get("max_drawdown_pct") is not None:
            parts.append(f"- 최대 낙폭: {market_summary['max_drawdown_pct']:.1f}%")
        if market_summary.get("short_ratio_avg_pct") is not None:
            parts.append(
                f"- 평균 공매도 비중: {market_summary['short_ratio_avg_pct']:.2f}%, "
                f"최근: {market_summary.get('short_ratio_recent_pct', 0):.2f}%"
            )
        opt_parts = []
        for inv in ["개인", "외국인", "기관계"]:
            key = f"option_{inv}_net_recent"
            if key in market_summary:
                opt_parts.append(f"{inv} {market_summary[key]:,.0f}")
        if opt_parts:
            parts.append("- 최근 20일 옵션 순매수(시장 단위): " + "; ".join(opt_parts))
        elif market_summary.get("option_data_missing"):
            parts.append("- 옵션 데이터 없음(해당 기간)")
        if market_summary.get("short_ratio_avg_pct") is None:
            parts.append("- 공매도 데이터 없음(해당 기간)")

    return "\n".join(parts)


def format_date(dt_obj: dt.date) -> str:
    """Format date to YYYYMMDD for DART API."""
    return dt_obj.strftime("%Y%m%d")


def fetch_filings(client: OpenDartClient, corp_code: str, start_date: dt.date, end_date: dt.date, pblntf_ty: Optional[str]) -> List[Dict[str, Any]]:
    """Fetch filings list from DART with simple pagination and dedup by 접수번호."""
    results: Dict[str, Dict[str, Any]] = {}
    page_no = 1
    page_count = 100
    max_pages = 5  # up to ~500 filings per query window
    while page_no <= max_pages:
        try:
            data = client.search_filings(
                corp_code=corp_code,
                bgn_de=format_date(start_date),
                end_de=format_date(end_date),
                pblntf_ty=None if pblntf_ty == "All" else pblntf_ty,
                page_no=page_no,
                page_count=page_count,
            )
        except RuntimeError as exc:
            # DART returns status 013 when there are no filings in the range; treat as empty instead of failing the app.
            if "013" in str(exc):
                break
            raise
        items = data.get("list") or []
        if not items:
            break
        for item in items:
            rno = item.get("rcept_no")
            if rno:
                results[rno] = item
        if len(items) < page_count:
            break
        page_no += 1
    return list(results.values())


def ratios_to_dataframe(ratios_by_year: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """Convert ratios dict into DataFrame for charting."""
    rows = []
    for year, data in ratios_by_year.items():
        row = {"year": year}
        row.update({k: v for k, v in data.items() if isinstance(v, (int, float)) or v is None})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year")


def yoy_to_dataframe(yoy_changes: Dict[str, Any]) -> pd.DataFrame:
    """Flatten YoY changes for display."""
    rows = []
    for metric, values in yoy_changes.items():
        for year, info in values.items():
            rows.append(
                {
                    "metric": metric,
                    "year": year,
                    "pct_change": info.get("pct_change"),
                    "direction": info.get("direction"),
                }
            )
    return pd.DataFrame(rows)


def create_line_chart(df: pd.DataFrame, metric: str, title: str) -> alt.Chart:
    """Build altair line chart."""
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(x="year:O", y=alt.Y(metric, title=title))
        .properties(height=250)
    )


def create_yoy_bar(df: pd.DataFrame, title: str) -> alt.Chart:
    """Bar chart for YoY changes."""
    color_scale = alt.Scale(
        domain=["up", "flat", "down"],
        range=["#2ca02c", "#9e9e9e", "#d62728"],
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="year:O",
            y=alt.Y("pct_change", title="pct_change (%)"),
            color=alt.Color("direction", scale=color_scale),
            tooltip=["metric", "year", "pct_change", "direction"],
        )
        .properties(height=250, title=title)
    )


def filings_dataframe(filings: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert filings list to DataFrame."""
    records = []
    for item in filings:
        records.append(
            {
                "접수번호": item.get("rcept_no"),
                "접수일": item.get("rcept_dt"),
                "보고서명": item.get("report_nm"),
                "공시구분": item.get("pblntf_ty"),
                "링크": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item.get('rcept_no')}",
            }
        )
    return pd.DataFrame(records)


def render_metrics(latest_year: int, ratios: Dict[str, Any]) -> None:
    """Render metric cards for latest year."""
    cols = st.columns(4)
    cols[0].metric("유동비율", f"{(ratios.get('current_ratio') or 0)*100:.1f}%")
    cols[1].metric("부채비율", f"{(ratios.get('debt_ratio') or 0)*100:.1f}%")
    cols[2].metric("ROE", f"{(ratios.get('roe') or 0)*100:.1f}%")
    cols[3].metric("ROA", f"{(ratios.get('roa') or 0)*100:.1f}%")
    st.caption(f"{latest_year}년 기준 핵심 지표")


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="DART Financial Analyzer", layout="wide")
    init_state()

    st.sidebar.header("입력")
    corp_name = st.sidebar.text_input("회사명", value=st.session_state.get("corp_name", "삼성전자"))
    stock_code = st.sidebar.text_input("종목코드 (옵션)", value=st.session_state.get("stock_code", ""))
    current_year = dt.datetime.now().year
    start_year = st.sidebar.number_input("시작 연도", min_value=2000, max_value=current_year, value=current_year - 3)
    end_year = st.sidebar.number_input("종료 연도", min_value=2000, max_value=current_year, value=current_year)

    preset_col1, preset_col2 = st.sidebar.columns(2)
    if preset_col1.button("최근 3년"):
        start_year = max(current_year - 2, 2000)
    if preset_col2.button("최근 5년"):
        start_year = max(current_year - 4, 2000)

    llm_choice = st.sidebar.selectbox(
        "LLM Backend",
        [
            "Local (free: ollama/gpt-oss)",
            "Gemini (Google AI)",
        ],
        index=0,
    )
    include_quarterly = st.sidebar.checkbox("분기 데이터 포함(요약)", value=False)
    run_button = st.sidebar.button("데이터 로드 및 분석 실행", use_container_width=True)

    if run_button:
        st.session_state["corp_name"] = corp_name
        st.session_state["stock_code"] = stock_code
        st.session_state["debate_result"] = None
        st.session_state["final_report"] = None
        try:
            client = OpenDartClient()
        except Exception as exc:
            st.error(f"OpenDART 클라이언트 초기화 실패: {exc}")
            return

        try:
            with st.spinner("기업 조회 중..."):
                corp_info = None
                if stock_code.strip():
                    corp_info = client.find_corp_by_stock_code(stock_code.strip())
                if corp_info is None and corp_name.strip():
                    matches = client.find_corp_by_name(corp_name.strip(), exact=False)
                    corp_info = matches[0] if matches else None
                if corp_info is None:
                    st.error("해당 이름/종목코드로 회사를 찾을 수 없습니다.")
                    return
                st.session_state["corp_info"] = corp_info
                # If user only provided 회사명, auto-fill 종목코드 for convenience
                if corp_info.stock_code and not stock_code.strip():
                    st.session_state["stock_code"] = corp_info.stock_code

            with st.spinner("재무 데이터 로드 및 지표 계산 중..."):
                try:
                    ratios_by_year, missing_years = get_yearly_ratios(
                        client,
                        corp_code=corp_info.corp_code,
                        start_year=int(start_year),
                        end_year=int(end_year),
                    )
                except RuntimeError as exc:
                    if "013" in str(exc):
                        st.error("조회된 재무제표 데이터가 없습니다. 연도 범위를 다시 선택하거나 공시 제출 여부를 확인하세요.")
                        return
                    raise

                if not ratios_by_year:
                    missing_msg = ""
                    if missing_years:
                        missing_msg = f" (DART에서 조회되지 않은 연도: {', '.join(str(y) for y in missing_years)})"
                    st.error(f"선택한 기간에 재무제표 데이터를 찾을 수 없습니다.{missing_msg}")
                    return

                if missing_years:
                    st.warning(
                        f"DART에서 데이터를 찾지 못한 연도: {', '.join(str(y) for y in missing_years)}. "
                        "해당 연도의 사업/분기보고서 제출 여부를 확인하거나 연도 범위를 조정하세요."
                    )

                yoy_changes = calc_yoy_changes(ratios_by_year, ratio_keys=["current_ratio", "debt_ratio", "roa", "roe"])
                debt_summary = None
                debt_summary_prev = None
                latest_year = max(ratios_by_year.keys()) if ratios_by_year else None
                if latest_year and latest_year in ratios_by_year:
                    debt_summary = calc_debt_structure_summary(
                        client.get_single_fs_full(corp_info.corp_code, latest_year)
                    )
                    # 전년 부채 요약(있으면)도 같이 구해 변화량 제공
                    prev_year = latest_year - 1
                    if prev_year >= start_year:
                        try:
                            prev_rows = client.get_single_fs_full(corp_info.corp_code, prev_year)
                            if prev_rows:
                                debt_summary_prev = calc_debt_structure_summary(prev_rows)
                        except RuntimeError:
                            debt_summary_prev = None
                # 분기 재무 지표 (선택)
                if include_quarterly:
                    try:
                        q_ratios, q_missing = get_quarterly_ratios(
                            client,
                            corp_code=corp_info.corp_code,
                            start_year=int(start_year),
                            end_year=int(end_year),
                            fs_div="CFS",
                        )
                        q_yoy = calc_quarterly_yoy_changes(
                            q_ratios, ratio_keys=["current_ratio", "debt_ratio", "roa", "roe"]
                        )
                        st.session_state["quarterly_ratios"] = q_ratios
                        st.session_state["quarterly_yoy"] = q_yoy
                        st.session_state["quarterly_missing"] = q_missing
                        if q_missing:
                            st.warning(f"분기 데이터 누락: {', '.join(q_missing)}")
                    except Exception as exc:
                        st.warning(f"분기 재무 데이터를 불러오지 못했습니다: {exc}")
                        st.session_state["quarterly_ratios"] = None
                        st.session_state["quarterly_yoy"] = None
                        st.session_state["quarterly_missing"] = None
                st.session_state["ratios_by_year"] = ratios_by_year
                st.session_state["yoy_changes"] = yoy_changes
                st.session_state["debt_summary"] = debt_summary
                st.session_state["debt_summary_prev"] = debt_summary_prev
                # 분기 재무 지표 (선택적)
                if include_quarterly:
                    try:
                        q_ratios, q_missing = get_quarterly_ratios(
                            client,
                            corp_code=corp_info.corp_code,
                            start_year=int(start_year),
                            end_year=int(end_year),
                            fs_div="CFS",
                        )
                        q_yoy = calc_quarterly_yoy_changes(
                            q_ratios, ratio_keys=["current_ratio", "debt_ratio", "roa", "roe"]
                        )
                        st.session_state["quarterly_ratios"] = q_ratios
                        st.session_state["quarterly_yoy"] = q_yoy
                        st.session_state["quarterly_missing"] = q_missing
                        if q_missing:
                            st.warning(f"분기 데이터 누락: {', '.join(q_missing)}")
                    except Exception as exc:
                        st.warning(f"분기 재무 데이터를 불러오지 못했습니다: {exc}")
                        st.session_state["quarterly_ratios"] = None
                        st.session_state["quarterly_yoy"] = None
                        st.session_state["quarterly_missing"] = None

            with st.spinner("공시 데이터 로드 중..."):
                start_dt = dt.date(end_year - 1, 1, 1)
                end_dt = dt.date(end_year, 12, 31)
                filings = fetch_filings(client, corp_info.corp_code, start_dt, end_dt, pblntf_ty=None)
                st.session_state["filings"] = filings
            with st.spinner("주가/공매도/옵션 데이터 로드 중..."):
                if corp_info.stock_code:
                    try:
                        _ensure_pykrx()
                        price_df, short_df, option_df = fetch_price_short_option(corp_info.stock_code)
                        market_summary = summarize_market_data(price_df, short_df, option_df)
                        st.session_state["market_price"] = price_df
                        st.session_state["market_short"] = short_df
                        st.session_state["market_option"] = option_df
                        st.session_state["market_summary"] = market_summary
                    except RuntimeError as exc:
                        st.warning(str(exc))
                        st.session_state["market_price"] = None
                        st.session_state["market_short"] = None
                        st.session_state["market_option"] = None
                        st.session_state["market_summary"] = None
                    except Exception as exc:
                        st.warning(f"주가/공매도/옵션 데이터를 불러오지 못했습니다: {exc}")
                        st.session_state["market_price"] = None
                        st.session_state["market_short"] = None
                        st.session_state["market_option"] = None
                        st.session_state["market_summary"] = None
                else:
                    st.info("종목코드가 없어 주가/공매도 데이터를 불러오지 않습니다.")
            st.success("데이터 로드 완료")
        except RuntimeError as exc:
            st.error(f"DART API 에러: {exc}")
        except Exception as exc:  # pragma: no cover - runtime protection
            st.error(f"알 수 없는 오류가 발생했습니다: {exc}")

    # Early exit if no data yet
    if not st.session_state.get("ratios_by_year"):
        st.info("좌측에서 회사명과 연도 범위를 입력 후 '데이터 로드 및 분석 실행'을 눌러주세요.")
        return

    corp_info = st.session_state["corp_info"]
    ratios_by_year = st.session_state["ratios_by_year"]
    yoy_changes = st.session_state["yoy_changes"] or {}
    debt_summary = st.session_state["debt_summary"]
    filings = st.session_state.get("filings") or []
    market_price = st.session_state.get("market_price")
    market_short = st.session_state.get("market_short")
    market_option = st.session_state.get("market_option")
    market_summary = st.session_state.get("market_summary")
    quarterly_ratios = st.session_state.get("quarterly_ratios") or {}
    quarterly_yoy = st.session_state.get("quarterly_yoy") or {}

    tabs = st.tabs(
        ["Overview", "Financial Ratios", "Capital Structure & Debt", "Filings", "가격/공매도/옵션", "AI Report & Debate"]
    )

    # Overview tab
    with tabs[0]:
        st.subheader("기업 기본 정보")
        if corp_info:
            info_cols = st.columns(3)
            info_cols[0].markdown(f"**회사명**: {corp_info.corp_name}")
            info_cols[1].markdown(f"**종목코드**: {corp_info.stock_code or '-'}")
            info_cols[2].markdown(f"**기업구분**: {corp_info.corp_cls or '-'}")
        latest_year = max(ratios_by_year.keys())
        prev_year = sorted(ratios_by_year.keys())[-2] if len(ratios_by_year) >= 2 else None
        st.divider()
        st.subheader(f"{latest_year}년 핵심 지표")
        render_metrics(latest_year, ratios_by_year[latest_year])
        st.write(
            "요약: 유동비율과 수익성(ROE/ROA)을 확인해 재무 건전성을 빠르게 파악하세요. "
            "부채비율이 높거나 급변한 경우 추가 확인이 필요합니다."
        )
        # 간단 YoY 하이라이트
        if yoy_changes:
            st.markdown("**전년 대비 변화 (주요 지표)**")
            yoy_lines = []
            for metric in ["current_ratio", "debt_ratio", "roa", "roe"]:
                if metric not in yoy_changes:
                    continue
                parts_change = []
                for year, info in sorted(yoy_changes[metric].items()):
                    pct = info.get("pct_change")
                    direction = info.get("direction")
                    parts_change.append(f"{year}: {pct:+.1f}% {direction}" if pct is not None else f"{year}: NA")
                yoy_lines.append(f"- {metric}: " + "; ".join(parts_change))
            if yoy_lines:
                st.markdown("\n".join(yoy_lines))
        # 최근 추이 테이블(매출/순이익/부채비율/ROE/ROA)
        trend_rows = []
        for y in sorted(ratios_by_year.keys())[-3:]:
            row = ratios_by_year[y]
            trend_rows.append(
                {
                    "연도": y,
                    "매출": row.get("revenue"),
                    "순이익": row.get("net_income"),
                    "부채비율": row.get("debt_ratio"),
                    "ROE": row.get("roe"),
                    "ROA": row.get("roa"),
                }
            )
        if trend_rows:
            st.markdown("**최근 추이 요약**")
            df_trend = pd.DataFrame(trend_rows)
            st.dataframe(df_trend, use_container_width=True, hide_index=True)
        # 리스크/체크 포인트 자동 요약
        checklist = []
        if prev_year:
            debt_change = yoy_changes.get("debt_ratio", {}).get(prev_year + 1, {})
            if debt_change:
                pct = debt_change.get("pct_change")
                if pct and pct > 50:
                    checklist.append(f"부채비율 급증(+{pct:.1f}%) → 차입/만기 구조 점검 필요")
        latest = ratios_by_year[latest_year]
        if (latest.get("debt_ratio") or 0) > 2.0:
            checklist.append(f"부채비율 {latest.get('debt_ratio')*100:.0f}% → 재무레버리지 높음")
        if (latest.get("roa") or 0) < 0:
            checklist.append("ROA 음수 → 영업/순이익 적자")
        if checklist:
            st.markdown("**체크 포인트**")
            st.write("\n".join(f"- {c}" for c in checklist))
        # 상세 테이블: 최신 연도 vs 직전 연도
        def _fmt_pct(val: Optional[float]) -> str:
            return f"{val*100:.1f}%" if isinstance(val, (int, float)) else "NA"

        def _fmt_num(val: Optional[float]) -> str:
            return f"{val:,.0f}" if isinstance(val, (int, float)) else "NA"

        if latest_year and (latest_year - 1) in ratios_by_year:
            prev_year = latest_year - 1
            st.markdown("**연도별 핵심 지표 비교**")
            comp_df = pd.DataFrame(
                [
                    {
                        "지표": "유동비율",
                        f"{prev_year}": _fmt_pct(ratios_by_year[prev_year].get("current_ratio")),
                        f"{latest_year}": _fmt_pct(ratios_by_year[latest_year].get("current_ratio")),
                    },
                    {
                        "지표": "부채비율",
                        f"{prev_year}": _fmt_pct(ratios_by_year[prev_year].get("debt_ratio")),
                        f"{latest_year}": _fmt_pct(ratios_by_year[latest_year].get("debt_ratio")),
                    },
                    {
                        "지표": "ROE",
                        f"{prev_year}": _fmt_pct(ratios_by_year[prev_year].get("roe")),
                        f"{latest_year}": _fmt_pct(ratios_by_year[latest_year].get("roe")),
                    },
                    {
                        "지표": "ROA",
                        f"{prev_year}": _fmt_pct(ratios_by_year[prev_year].get("roa")),
                        f"{latest_year}": _fmt_pct(ratios_by_year[latest_year].get("roa")),
                    },
                    {
                        "지표": "매출액",
                        f"{prev_year}": _fmt_num(ratios_by_year[prev_year].get("revenue")),
                        f"{latest_year}": _fmt_num(ratios_by_year[latest_year].get("revenue")),
                    },
                    {
                        "지표": "당기순이익",
                        f"{prev_year}": _fmt_num(ratios_by_year[prev_year].get("net_income")),
                        f"{latest_year}": _fmt_num(ratios_by_year[latest_year].get("net_income")),
                    },
                ]
            )
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        # 부채 요약 첨부
        if debt_summary:
            st.markdown("**부채/자본 요약 (최신 연도)**")
            debt_df = pd.DataFrame(
                [
                    {
                        "총부채": _fmt_num(debt_summary.get("total_liabilities")),
                        "자본총계": _fmt_num(ratios_by_year[latest_year].get("equity_total") if latest_year else None),
                        "부채비율": _fmt_pct(ratios_by_year[latest_year].get("debt_ratio") if latest_year else None),
                        "단기부채": _fmt_num(debt_summary.get("short_term_debt")),
                        "장기부채": _fmt_num(debt_summary.get("long_term_debt")),
                        "사채/회사채": _fmt_num(debt_summary.get("bond_balance")),
                    }
                ]
            )
            st.dataframe(debt_df, use_container_width=True, hide_index=True)
        # 시장 요약 첨부
        if market_summary:
            st.markdown("**시장 요약 (최근 1년)**")
            ms_rows = []
            for key, label in [
                ("price_return_pct", "주가 수익률"),
                ("volatility_252d_pct", "연환산 변동성"),
                ("max_drawdown_pct", "최대 낙폭"),
                ("short_ratio_avg_pct", "평균 공매도비중"),
                ("short_ratio_recent_pct", "최근 공매도비중"),
            ]:
                val = market_summary.get(key)
                if val is not None:
                    suffix = "%" if "pct" in key else ""
                    ms_rows.append({"지표": label, "값": f"{val:.2f}{suffix}"})
            if ms_rows:
                st.dataframe(ms_rows, use_container_width=True, hide_index=True)
        # 분기 하이라이트
        if quarterly_ratios:
            st.markdown("**최근 분기 스냅샷**")
            # 최근 분기 key
            latest_q = sorted(quarterly_ratios.keys())[-1]
            latest_q_data = quarterly_ratios[latest_q]
            q_cols = st.columns(4)
            q_cols[0].metric(f"{latest_q} 유동비율", _fmt_pct(latest_q_data.get("current_ratio")))
            q_cols[1].metric(f"{latest_q} 부채비율", _fmt_pct(latest_q_data.get("debt_ratio")))
            q_cols[2].metric(f"{latest_q} ROE", _fmt_pct(latest_q_data.get("roe")))
            q_cols[3].metric(f"{latest_q} ROA", _fmt_pct(latest_q_data.get("roa")))
            if quarterly_yoy:
                st.caption("동일 분기 전년 대비 YoY가 별도 표/차트에서 제공됩니다.")
        # 추가 컨텍스트: 주가/공매도/공시 요약
        info_cols2 = st.columns(3)
        # 주가/공매도
        if market_summary and market_summary.get("price_return_pct") is not None:
            price_ret = market_summary["price_return_pct"]
            short_pct = market_summary.get("short_ratio_recent_pct")
            info_cols2[0].markdown(
                f"**최근 1년 주가**: {price_ret:+.1f}%"
                + (f" / 공매도비중 최근: {short_pct:.2f}%"
                   if short_pct is not None else "")
            )
        else:
            info_cols2[0].markdown("**주가/공매도**: 데이터 없음")
        # 공시 건수
        if filings:
            info_cols2[1].markdown(f"**최근 1년 공시 건수**: {len(filings)}건")
        else:
            info_cols2[1].markdown("**최근 1년 공시**: 없음")
        # 분기 데이터 존재 여부
        if quarterly_ratios:
            latest_q = sorted(quarterly_ratios.keys())[-1]
            info_cols2[2].markdown(f"**분기 데이터**: {latest_q}까지 확보")
        else:
            info_cols2[2].markdown("**분기 데이터**: 불러오지 않음")
        # 최신 연도 상세 테이블 + 전년 대비
        prev_year = latest_year - 1 if (latest_year - 1) in ratios_by_year else None
        detail_rows = []
        labels = {
            "current_ratio": "유동비율",
            "debt_ratio": "부채비율",
            "roe": "ROE",
            "roa": "ROA",
            "revenue": "매출액",
            "net_income": "당기순이익",
            "total_assets": "총자산",
            "total_liabilities": "총부채",
            "equity_total": "자본총계",
        }
        def _fmt(v: Optional[float], pct: bool = False) -> str:
            if v is None:
                return "NA"
            return f"{v*100:.1f}%" if pct else f"{v:,.0f}"

        for key, label in labels.items():
            cur = ratios_by_year[latest_year].get(key)
            prev = ratios_by_year[prev_year].get(key) if prev_year else None
            change = None
            if cur is not None and prev not in (None, 0):
                change = (cur - prev) / prev * 100
            is_pct = key in ["current_ratio", "debt_ratio", "roe", "roa"]
            detail_rows.append(
                {
                    "지표": label,
                    "현재": _fmt(cur, pct=is_pct),
                    "전년": _fmt(prev, pct=is_pct) if prev_year else "-",
                    "전년 대비": f"{change:+.1f}%" if change is not None else "NA",
                }
            )
        if detail_rows:
            st.markdown("**핵심 재무 상세 (전년 대비)**")
            st.dataframe(detail_rows, use_container_width=True, hide_index=True)
        # 강화된 개요: 최신 연도 vs 직전 연도, 추이 미니차트, 데이터 커버리지
        prev_year = latest_year - 1
        if prev_year in ratios_by_year:
            st.markdown("**최신 연도 vs 직전 연도**")
            rows = []
            for key, label in [
                ("current_ratio", "유동비율"),
                ("debt_ratio", "부채비율"),
                ("roe", "ROE"),
                ("roa", "ROA"),
            ]:
                cur = ratios_by_year[latest_year].get(key)
                prv = ratios_by_year[prev_year].get(key)
                pct = None
                if cur is not None and prv not in (None, 0):
                    pct = (cur / prv - 1) * 100
                rows.append(
                    {
                        "지표": label,
                        f"{prev_year}": prv,
                        f"{latest_year}": cur,
                        "증감(%)": pct,
                    }
                )
            st.dataframe(rows, hide_index=True, use_container_width=True)
        # 최근 4개 연도 스파크라인
        if len(ratios_by_year) >= 2:
            st.markdown("**최근 연도 추이(미니 차트)**")
            years_sorted = sorted(ratios_by_year.keys())[-4:]
            spark_df = []
            for y in years_sorted:
                r = ratios_by_year[y]
                spark_df.append(
                    {
                        "year": y,
                        "current_ratio": r.get("current_ratio"),
                        "debt_ratio": r.get("debt_ratio"),
                        "roe": r.get("roe"),
                        "roa": r.get("roa"),
                    }
                )
            spark_df = pd.DataFrame(spark_df)
            spark_cols = st.columns(2)
            for idx, (col_name, label) in enumerate(
                [("current_ratio", "유동비율"), ("debt_ratio", "부채비율"), ("roe", "ROE"), ("roa", "ROA")]
            ):
                if col_name not in spark_df:
                    continue
                chart = (
                    alt.Chart(spark_df)
                    .mark_line(point=True)
                    .encode(x="year:O", y=alt.Y(col_name, title=label))
                    .properties(height=180, title=label)
                )
                spark_cols[idx // 2].altair_chart(chart, use_container_width=True)
        # 데이터 커버리지 안내
        missing_years = st.session_state.get("quarterly_missing") or []
        if missing_years:
            st.info(f"분기 데이터 누락: {', '.join(missing_years)}")
        # 상세 스냅샷: 최신/전년 값과 YoY
        years_sorted = sorted(ratios_by_year.keys())
        if len(years_sorted) >= 2:
            prev_year = years_sorted[-2]
            latest = ratios_by_year[latest_year]
            prev = ratios_by_year[prev_year]

            def _fmt(v: Optional[float], pct: bool = False) -> str:
                if v is None:
                    return "NA"
                return f"{v*100:.1f}%" if pct else f"{v:.2f}"

            rows = []
            for key, label, is_pct in [
                ("current_ratio", "유동비율", True),
                ("debt_ratio", "부채비율", True),
                ("roe", "ROE", True),
                ("roa", "ROA", True),
                ("net_margin", "순이익률", True),
            ]:
                yoy_info = yoy_changes.get(key, {}).get(latest_year, {})
                rows.append(
                    {
                        "지표": label,
                        f"{prev_year}": _fmt(prev.get(key), is_pct),
                        f"{latest_year}": _fmt(latest.get(key), is_pct),
                        "YoY 변화": f"{(yoy_info.get('pct_change') or 0):+.1f}%" if yoy_info else "NA",
                    }
                )
            st.markdown("**핵심 지표 스냅샷 (전년 대비)**")
            st.dataframe(rows, hide_index=True, use_container_width=True)

        # 시장 스냅샷 (있을 경우)
        if market_summary:
            st.markdown("**시장/공매도 스냅샷 (최근 1년)**")
            m_rows = []
            for key, label in [
                ("price_return_pct", "주가 수익률"),
                ("volatility_252d_pct", "연환산 변동성"),
                ("max_drawdown_pct", "최대 낙폭"),
                ("short_ratio_recent_pct", "최근 공매도 비중"),
            ]:
                if market_summary.get(key) is not None:
                    m_rows.append({"지표": label, "값": f"{market_summary[key]:+.1f}%"})
            if m_rows:
                st.dataframe(m_rows, hide_index=True, use_container_width=True)

    # Financial Ratios tab
    with tabs[1]:
        st.subheader("재무 비율 추이")
        df_ratios = ratios_to_dataframe(ratios_by_year)
        chart_cols = st.columns(2)
        chart_cols[0].altair_chart(create_line_chart(df_ratios, "current_ratio", "유동비율"), use_container_width=True)
        chart_cols[1].altair_chart(create_line_chart(df_ratios, "debt_ratio", "부채비율"), use_container_width=True)
        chart_cols2 = st.columns(2)
        chart_cols2[0].altair_chart(create_line_chart(df_ratios, "roe", "ROE"), use_container_width=True)
        chart_cols2[1].altair_chart(create_line_chart(df_ratios, "roa", "ROA"), use_container_width=True)

        st.subheader("전년 대비 변화율")
        df_yoy = yoy_to_dataframe(yoy_changes)
        if not df_yoy.empty:
            st.altair_chart(create_yoy_bar(df_yoy, "YoY % Changes"), use_container_width=True)
            st.dataframe(df_yoy, use_container_width=True, hide_index=True)
            # 연도별/지표별 피벗 테이블로 한눈에 비교
            pivot = df_yoy.pivot_table(index="year", columns="metric", values="pct_change")
            st.markdown("#### YoY 변화율 피벗")
            st.dataframe(pivot, use_container_width=True)
        else:
            st.info("YoY 변화 데이터가 없습니다.")
        if quarterly_ratios:
            st.subheader("분기 재무 비율 요약")
            q_df = pd.DataFrame(
                [
                    {"분기": k, **{kk: v for kk, v in val.items() if isinstance(v, (int, float)) or v is None}}
                    for k, val in quarterly_ratios.items()
                ]
            )
            # 분기 순서 정렬을 위한 키 추가
            def _order_key(qstr: str) -> int:
                try:
                    year = int(qstr[:4])
                    q = qstr[4:]
                    q_idx = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}.get(q, 5)
                    return year * 10 + q_idx
                except Exception:
                    return 0

            q_df["order"] = q_df["분기"].apply(_order_key)
            q_df = q_df.sort_values("order")
            st.dataframe(q_df.drop(columns=["order"]), use_container_width=True, hide_index=True)

            # 핵심 지표 분기 추이 시각화
            metric_labels = {
                "current_ratio": "유동비율",
                "debt_ratio": "부채비율",
                "roe": "ROE",
                "roa": "ROA",
            }
            chart_cols = st.columns(2)
            charts = []
            for idx, (key, label) in enumerate(metric_labels.items()):
                if key in q_df.columns:
                    chart = (
                        alt.Chart(q_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("분기:N", sort=list(q_df["분기"]), title="분기"),
                            y=alt.Y(f"{key}:Q", title=label),
                        )
                        .properties(height=250, title=f"{label} (분기)")
                    )
                    charts.append(chart)
                    chart_cols[idx % 2].altair_chart(chart, use_container_width=True)
            if quarterly_yoy:
                st.markdown("#### 분기 YoY 변화 (동일 분기 전년 대비)")
                rows = []
                for metric, items in quarterly_yoy.items():
                    for qkey, info in items.items():
                        rows.append(
                            {
                                "지표": metric,
                                "분기": qkey,
                                "변화율(%)": info.get("pct_change"),
                                "방향": info.get("direction"),
                            }
                        )
                if rows:
                    yoy_df = pd.DataFrame(rows).sort_values("분기")
                    st.dataframe(yoy_df, use_container_width=True, hide_index=True)
                    # 시각화: 분기별 YoY % 변화
                    yoy_chart = (
                        alt.Chart(yoy_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("분기:N", sort=list(yoy_df["분기"]), title="분기"),
                            y=alt.Y("변화율(%):Q", title="YoY 변화율(%)"),
                            color="지표:N",
                        )
                        .properties(height=250)
                    )
                    st.altair_chart(yoy_chart, use_container_width=True)

    # Capital Structure tab
    with tabs[2]:
        st.subheader("자본구조 및 부채")
        if debt_summary:
            st.table(
                pd.DataFrame(
                    [
                        {
                            "총부채": debt_summary.get("total_liabilities"),
                            "사채/회사채": debt_summary.get("bond_balance"),
                            "사채비율": debt_summary.get("bond_ratio"),
                            "단기부채": debt_summary.get("short_term_debt"),
                            "장기부채": debt_summary.get("long_term_debt"),
                        }
                    ]
                )
            )
            # 전년 대비 부채 변화(요약)
            prev = st.session_state.get("debt_summary_prev")
            if prev:
                def _delta(cur: Optional[float], prv: Optional[float]) -> str:
                    if cur is None or prv in (None, 0):
                        return "N/A"
                    diff = cur - prv
                    pct = (diff / prv) * 100
                    return f"{diff:,.0f} ({pct:+.1f}%)"

                st.markdown("**전년 대비 부채 증감**")
                st.write(
                    f"- 총부채 증감: {_delta(debt_summary.get('total_liabilities'), prev.get('total_liabilities'))}"
                )
                st.write(
                    f"- 단기부채 증감: {_delta(debt_summary.get('short_term_debt'), prev.get('short_term_debt'))}"
                )
                st.write(
                    f"- 장기부채 증감: {_delta(debt_summary.get('long_term_debt'), prev.get('long_term_debt'))}"
                )
            debt_df = pd.DataFrame(
                [
                    {"항목": "사채/회사채", "비율": debt_summary.get("bond_ratio")},
                    {"항목": "단기부채", "비율": debt_summary.get("short_term_ratio")},
                    {"항목": "장기부채", "비율": debt_summary.get("long_term_ratio")},
                ]
            ).dropna()
            if not debt_df.empty:
                st.altair_chart(
                    alt.Chart(debt_df).mark_bar().encode(x="항목", y="비율", tooltip=["항목", "비율"]).properties(height=250),
                    use_container_width=True,
                )
        else:
            st.info("부채 요약 정보를 불러오지 못했습니다.")

        st.subheader("최근 공시 (발행공시/기타 포함)")
        filings_df = filings_dataframe(filings)
        st.dataframe(filings_df, use_container_width=True, hide_index=True)

    # Filings tab
    with tabs[3]:
        st.subheader("공시 검색")
        col_a, col_b = st.columns(2)
        # Friendly labels so 비전문가도 쉽게 선택 가능
        pblntf_options = [
            ("전체 공시", "All"),
            ("정기보고서(사업/분기/반기)", "A"),
            ("주요사항/의사결정 보고", "B"),
            ("발행공시(유상·주식 발행 등)", "C"),
            ("지분공시(대량보유 등)", "D"),
            ("기타공시", "E"),
        ]
        selected_pblntf = col_a.selectbox("공시 타입", pblntf_options, format_func=lambda x: x[0], index=0)
        pblntf_ty = selected_pblntf[1]
        months_back = col_b.selectbox("기간", [3, 6, 12], format_func=lambda x: f"최근 {x}개월", index=2)
        if st.button("공시 업데이트"):
            try:
                client = OpenDartClient()
                end_dt = dt.date.today()
                start_dt = end_dt - dt.timedelta(days=30 * months_back)
                with st.spinner("공시 조회 중..."):
                    filings = fetch_filings(client, corp_info.corp_code, start_dt, end_dt, pblntf_ty=pblntf_ty)
                st.session_state["filings"] = filings
                filings_df = filings_dataframe(filings)
                st.dataframe(filings_df, use_container_width=True, hide_index=True)
            except Exception as exc:  # pragma: no cover - safety
                st.error(f"공시 조회 실패: {exc}")
        else:
            filings_df = filings_dataframe(st.session_state.get("filings") or [])
            st.dataframe(filings_df, use_container_width=True, hide_index=True)

    # Price/Short/Option tab
    with tabs[4]:
        st.subheader("가격/공매도/옵션 (최근 1년)")
        if market_price is None:
            st.info("주가/공매도/옵션 데이터를 불러오지 못했습니다.")
        else:
            price_df = market_price.reset_index().rename(columns={"날짜": "date"})
            if "종가" in price_df.columns:
                st.altair_chart(
                    alt.Chart(price_df).mark_line().encode(x="date:T", y="종가:Q").properties(title="종가 추이", height=250),
                    use_container_width=True,
                )
            if market_short is not None and not market_short.empty:
                short_df = market_short.reset_index().rename(columns={"날짜": "date"})
                ratio_col = "공매도비중" if "공매도비중" in short_df.columns else None
                if ratio_col:
                    short_df[ratio_col] = short_df[ratio_col].astype(float)
                    st.altair_chart(
                        alt.Chart(short_df)
                        .mark_line(color="#d62728")
                        .encode(x="date:T", y=alt.Y(f"{ratio_col}:Q", title="공매도비중(%)"))
                        .properties(title="공매도 비중", height=250),
                        use_container_width=True,
                    )
                else:
                    st.info("공매도 비중 계산 가능한 열이 없습니다.")
            else:
                st.info("해당 기간 공매도 데이터가 없습니다.")
            if market_option is not None and not market_option.empty:
                st.markdown("옵션 투자자 순매수 (최근 20일 합계)")
                opt_rows = []
                for inv in ["개인", "외국인", "기관계"]:
                    col = f"option_{inv}_net_recent"
                    if market_summary and col in market_summary:
                        opt_rows.append({"투자자": inv, "순매수": market_summary[col]})
                if opt_rows:
                    st.dataframe(opt_rows, hide_index=True, use_container_width=True)
                else:
                    st.info("옵션 순매수 요약을 계산할 수 없습니다.")
            else:
                st.info("옵션 데이터가 없습니다.")
            if market_summary:
                st.markdown("요약 지표")
                summary_rows = []
                for key, label in [
                    ("price_return_pct", "1년 수익률(%)"),
                    ("return_20d_pct", "20일 수익률(%)"),
                    ("return_60d_pct", "60일 수익률(%)"),
                    ("return_120d_pct", "120일 수익률(%)"),
                    ("volatility_252d_pct", "연환산 변동성(%)"),
                    ("max_drawdown_pct", "최대 낙폭(%)"),
                    ("short_ratio_avg_pct", "평균 공매도 비중(%)"),
                    ("short_ratio_recent_pct", "최근 공매도 비중(%)"),
                ]:
                    if market_summary.get(key) is not None:
                        summary_rows.append({"지표": label, "값": market_summary[key]})
                if summary_rows:
                    st.dataframe(summary_rows, hide_index=True, use_container_width=True)

    # AI Report & Debate tab
    with tabs[5]:
        st.subheader("Multi-Agent Debate")
        if st.button("Multi-Agent 분석 실행"):
            backend = None
            try:
                if llm_choice.startswith("Local"):
                    # Default to a free local model; override via LOCAL_LLM_MODEL env var
                    local_model = os.getenv("LOCAL_LLM_MODEL", "exaone3.5:7.8b")
                    local_endpoint = os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:11434/api/chat")
                    health_err = check_local_llm(local_endpoint)
                    if health_err:
                        st.error(
                            f"로컬 LLM에 연결할 수 없습니다. 서버를 실행하거나 LOCAL_LLM_ENDPOINT를 수정하세요.\n{health_err}"
                        )
                        return
                    backend = LocalLLMBackend(model=local_model, endpoint=local_endpoint, soft_fallback=False)
                else:
                    api_key = os.getenv("GOOGLE_API_KEY")
                    # Prefer newer Gemini models by default
                    model = normalize_gemini_model(os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
                    if not api_key:
                        st.error("GOOGLE_API_KEY가 설정되지 않았습니다. 무료 로컬 모델을 선택하세요.")
                        return
                    backend = GeminiBackend(api_key=api_key, model=model)
            except Exception as exc:
                st.warning(f"선택한 백엔드를 초기화하지 못했습니다({exc}). 무료 로컬 모델로 대체합니다.")
                backend = LocalLLMBackend(
                    model=os.getenv("LOCAL_LLM_MODEL", "exaone3.5:7.8b"),
                    endpoint=os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:11434/api/chat"),
                    soft_fallback=False,
                )

            agents = {
                "FS_ANALYST": Agent("FS_ANALYST", ROLE_PROMPT_FS_ANALYST, backend),
                "CAPITAL_STRUCTURE": Agent("CAPITAL_STRUCTURE", ROLE_PROMPT_CAPITAL_STRUCTURE, backend),
                "EQUITY_ANALYST": Agent("EQUITY_ANALYST", ROLE_PROMPT_EQUITY_ANALYST, backend),
                "RISK_MANAGER": Agent("RISK_MANAGER", ROLE_PROMPT_RISK_MANAGER, backend),
                "REPORT_WRITER": Agent("REPORT_WRITER", ROLE_PROMPT_REPORT_WRITER, backend),
            }

            session = DebateSession(agents=agents, global_prompt=GLOBAL_SYSTEM_PROMPT)
            debt_prev = st.session_state.get("debt_summary_prev")
            data_summary = build_debate_context(
                ratios_by_year,
                yoy_changes,
                st.session_state.get("debt_summary"),
                debt_prev,
                st.session_state.get("filings") or [],
                st.session_state.get("market_summary"),
            )
            with st.spinner("LLM 디베이트 진행 중..."):
                debate = session.run_multi_rounds(data_summary=data_summary, num_rounds=2)
                report = generate_final_report(
                    session=session,
                    corp_name=corp_info.corp_name,
                    corp_code=corp_info.corp_code,
                    years_range=f"{min(ratios_by_year.keys())}~{max(ratios_by_year.keys())}",
                    industry=corp_info.corp_cls or "N/A",
                    extra_info="",
                    data_summary=data_summary,
                )
            st.session_state["debate_result"] = debate
            st.session_state["final_report"] = report
            st.session_state["debate_data_summary"] = data_summary

        debate_result = st.session_state.get("debate_result")
        final_report = st.session_state.get("final_report")
        if debate_result:
            st.markdown("### 에이전트별 포인트")
            rounds = debate_result.get("rounds", [])
            if rounds:
                if st.session_state.get("debate_data_summary"):
                    with st.expander("이번 디베이트에 사용된 핵심 데이터 요약"):
                        st.markdown(st.session_state["debate_data_summary"])
                latest_round = rounds[-1]
                for name, res in latest_round.items():
                    with st.expander(name):
                        st.write("- **Position**: ", res.get("position"))
                        st.write("- **Agreements**: ", res.get("agreements"))
                        st.write("- **Disagreements**: ", res.get("disagreements"))
                        st.write("- **Questions**: ", res.get("questions_to_others"))
            # Visualize debate flow across rounds for easier digestion
            history = debate_result.get("history", [])
            if history:
                st.markdown("### 라운드별 요약")
                for idx, h in enumerate(history, start=1):
                    st.markdown(f"- Round {idx}: {h.get('round_summary', '')}")
                rows = []
                for r_idx, round_data in enumerate(rounds, start=1):
                    for agent_name, res in round_data.items():
                        rows.append(
                            {
                                "라운드": r_idx,
                                "에이전트": agent_name,
                                "핵심 의견": res.get("position"),
                                "동의 포인트": "\n• " + "\n• ".join(res.get("agreements") or []),
                                "이견/우려": "\n• " + "\n• ".join(res.get("disagreements") or []),
                                "다른 에이전트 질문": "\n• " + "\n• ".join(res.get("questions_to_others") or []),
                            }
                        )
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
        if final_report:
            st.markdown("### 최종 리포트")
            st.markdown(final_report)
            st.download_button(
                label="리포트 다운로드 (md)",
                data=final_report,
                file_name="dart_report.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
