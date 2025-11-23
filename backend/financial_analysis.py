"""Financial analysis helpers for OpenDART data.

This module computes basic liquidity/leverage/profitability ratios from
OpenDART financial statement rows and offers year-over-year change utilities.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from backend.opendart_client import OpenDartClient

load_dotenv()

# 분기 보고서 코드 매핑 (DART)
QUARTER_REPORT_CODES: List[Tuple[str, str]] = [
    ("Q1", "11013"),  # 1분기
    ("Q2", "11012"),  # 반기
    ("Q3", "11014"),  # 3분기
    ("Q4", "11011"),  # 사업보고서(연간)
]


def _to_float(value: Any) -> Optional[float]:
    """Convert OpenDART numeric fields (often strings with commas) to float."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).replace(",", "").strip()
        if not text:
            return None
        return float(text)
    except (ValueError, TypeError):
        return None


def find_amount(
    fs_rows: Iterable[Dict[str, Any]],
    sj_div: Optional[str],
    account_name_candidates: List[str],
    use_acc_id: Optional[str] = None,
    field: str = "thstrm_amount",
) -> Optional[float]:
    """
    Locate an account amount from financial statement rows.

    Args:
        fs_rows: Rows from fnlttSinglAcntAll.
        sj_div: Section code (e.g., BS, IS, CIS, CF). If None, search all.
        account_name_candidates: List of possible account names to match partially.
        use_acc_id: If provided, match on account_id first.
        field: Which numeric field to pull (thstrm_amount, frmtrm_amount, etc.).

    Returns:
        Amount as float if found, otherwise None.
    """
    def _norm(text: str) -> str:
        return (text or "").replace(" ", "").lower()

    normalized_candidates = [_norm(name) for name in account_name_candidates]

    for row in fs_rows:
        if sj_div and row.get("sj_div") != sj_div:
            continue
        if use_acc_id and row.get("account_id") == use_acc_id:
            return _to_float(row.get(field))

        acc_name = _norm(str(row.get("account_nm", "")))
        for cand in normalized_candidates:
            if cand in acc_name:
                return _to_float(row.get(field))
    return None


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divide with None/zero guarding."""
    if numerator is None or denominator in (None, 0, 0.0):
        return None
    return numerator / denominator


def calc_basic_ratios_from_fs(fs_rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Calculate basic financial ratios from fnlttSinglAcntAll rows.

    Ratios: current_ratio, quick_ratio, debt_ratio, roa, roe, net_margin.
    """
    current_assets = find_amount(fs_rows, "BS", ["유동자산"])
    current_liabilities = find_amount(fs_rows, "BS", ["유동부채"])
    inventory = find_amount(fs_rows, "BS", ["재고자산"])
    total_liabilities = find_amount(fs_rows, "BS", ["부채총계", "총부채"])
    equity_total = find_amount(fs_rows, "BS", ["자본총계", "총자본", "지배기업의소유주지분"])
    total_assets = find_amount(fs_rows, "BS", ["자산총계", "총자산"])

    cash = find_amount(fs_rows, "BS", ["현금및현금성자산"])
    short_invest = find_amount(fs_rows, "BS", ["단기금융상품", "단기투자자산"])
    receivables = find_amount(fs_rows, "BS", ["매출채권", "받을어음및받을채권", "단기매출채권"])
    quick_assets_candidates = [cash, short_invest, receivables]
    quick_assets = sum(x for x in quick_assets_candidates if x is not None)
    if all(x is None for x in quick_assets_candidates):
        quick_assets = None

    revenue = find_amount(fs_rows, "IS", ["매출액", "수익(매출액)", "영업수익"])
    net_income = find_amount(fs_rows, "IS", ["당기순이익", "당기순손익"])

    current_ratio = _safe_div(current_assets, current_liabilities)
    quick_ratio = _safe_div(quick_assets, current_liabilities)
    debt_ratio = _safe_div(total_liabilities, equity_total)
    roa = _safe_div(net_income, total_assets)
    roe = _safe_div(net_income, equity_total)
    net_margin = _safe_div(net_income, revenue)

    return {
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "debt_ratio": debt_ratio,
        "roa": roa,
        "roe": roe,
        "net_margin": net_margin,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "total_liabilities": total_liabilities,
        "equity_total": equity_total,
        "total_assets": total_assets,
        "revenue": revenue,
        "net_income": net_income,
        "inventory": inventory,
    }


def get_yearly_ratios(
    client: OpenDartClient,
    corp_code: str,
    start_year: int,
    end_year: int,
    reprt_code: str = "11011",
    fs_div: str = "CFS",
    verbose: bool = False,
) -> Tuple[Dict[int, Dict[str, Optional[float]]], List[int]]:
    """
    Fetch financial statements for each year and compute basic ratios.

    Returns:
        (Mapping of year -> ratio dict, missing_years list)
    Raises:
        RuntimeError propagates from OpenDartClient on API errors.
    """
    results: Dict[int, Dict[str, Optional[float]]] = {}
    missing_years: List[int] = []
    for year in range(start_year, end_year + 1):
        try:
            fs_rows = client.get_single_fs_full(
                corp_code=corp_code,
                bsns_year=year,
                reprt_code=reprt_code,
                fs_div=fs_div,
            )
        except RuntimeError as exc:
            # DART status 013: no data for that year/report; skip the year instead of failing all
            if "013" in str(exc):
                if verbose:
                    print(f"[WARN] {year}년 재무제표 없음(013) – 스킵합니다.")
                missing_years.append(year)
                continue
            raise
        if not fs_rows:
            # No rows returned; treat as missing data for that year
            if verbose:
                print(f"[WARN] {year}년 재무제표 데이터 없음 – 스킵합니다.")
            missing_years.append(year)
            continue
        ratios = calc_basic_ratios_from_fs(fs_rows)
        results[year] = ratios
    return results, missing_years


def calc_yoy_changes(
    yearly_ratios: Dict[int, Dict[str, Optional[float]]],
    ratio_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[int, Dict[str, Optional[float]]]]:
    """
    Calculate year-over-year changes for selected ratios.

    Returns:
        Nested dict: ratio_key -> year -> {value, prev, abs_change, pct_change, direction}
    """
    if not yearly_ratios:
        return {}

    sorted_years = sorted(yearly_ratios.keys())
    sample_year = sorted_years[0]
    if ratio_keys is None:
        ratio_keys = list(yearly_ratios[sample_year].keys())

    result: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {k: {} for k in ratio_keys}

    for idx in range(1, len(sorted_years)):
        year = sorted_years[idx]
        prev_year = sorted_years[idx - 1]
        current = yearly_ratios.get(year, {})
        previous = yearly_ratios.get(prev_year, {})

        for key in ratio_keys:
            cur_val = current.get(key)
            prev_val = previous.get(key)
            if cur_val is None or prev_val in (None, 0):
                result[key][year] = {
                    "value": cur_val,
                    "prev": prev_val,
                    "abs_change": None,
                    "pct_change": None,
                    "direction": None,
                }
                continue

            abs_change = cur_val - prev_val
            pct_change = (abs_change / prev_val) * 100.0
            direction = "flat"
            if not math.isclose(abs_change, 0.0, rel_tol=1e-9, abs_tol=1e-6):
                direction = "up" if abs_change > 0 else "down"

            result[key][year] = {
                "value": cur_val,
                "prev": prev_val,
                "abs_change": abs_change,
                "pct_change": pct_change,
                "direction": direction,
            }
        return result


def get_quarterly_ratios(
    client: OpenDartClient,
    corp_code: str,
    start_year: int,
    end_year: int,
    fs_div: str = "CFS",
    verbose: bool = False,
) -> Tuple[Dict[str, Dict[str, Optional[float]]], List[str]]:
    """
    Fetch quarterly financials (분기/반기/3분기/사업보고서) and compute ratios.

    Returns:
        - Mapping of "YYYYQn" -> ratio dict
        - missing keys list (e.g., "2023Q1")
    """
    results: Dict[str, Dict[str, Optional[float]]] = {}
    missing: List[str] = []
    for year in range(start_year, end_year + 1):
        for quarter_label, reprt_code in QUARTER_REPORT_CODES:
            key = f"{year}{quarter_label}"
            try:
                fs_rows = client.get_single_fs_full(
                    corp_code=corp_code,
                    bsns_year=year,
                    reprt_code=reprt_code,
                    fs_div=fs_div,
                )
            except RuntimeError as exc:
                if "013" in str(exc):
                    if verbose:
                        print(f"[WARN] {key} 재무제표 없음(013) – 스킵합니다.")
                    missing.append(key)
                    continue
                raise
            if not fs_rows:
                if verbose:
                    print(f"[WARN] {key} 재무제표 데이터 없음 – 스킵합니다.")
                missing.append(key)
                continue
            results[key] = calc_basic_ratios_from_fs(fs_rows)
    return results, missing


def calc_quarterly_yoy_changes(
    quarterly_ratios: Dict[str, Dict[str, Optional[float]]],
    ratio_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """
    Calculate YoY change by matching same quarter across years (e.g., 2024Q1 vs 2023Q1).
    """
    if not quarterly_ratios:
        return {}

    # Parse keys into sortable tuples
    parsed: List[Tuple[int, int, str, Dict[str, Optional[float]]]] = []
    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    for k, v in quarterly_ratios.items():
        year = int(k[:4])
        q = k[4:]
        if q not in quarter_order:
            continue
        parsed.append((year, quarter_order.index(q), q, v))
    parsed.sort()

    if ratio_keys is None and parsed:
        ratio_keys = list(parsed[0][3].keys())

    result: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {k: {} for k in ratio_keys or []}
    # Build lookup by (year, quarter)
    data_by_yq: Dict[Tuple[int, str], Dict[str, Optional[float]]] = {}
    for y, _, q, data in parsed:
        data_by_yq[(y, q)] = data

    for y, _, q, data in parsed:
        prev_key = (y - 1, q)
        if prev_key not in data_by_yq:
            continue
        prev_data = data_by_yq[prev_key]
        for key in ratio_keys or []:
            cur_val = data.get(key)
            prev_val = prev_data.get(key)
            if cur_val is None or prev_val in (None, 0):
                result[key][f"{y}{q}"] = {
                    "value": cur_val,
                    "prev": prev_val,
                    "abs_change": None,
                    "pct_change": None,
                    "direction": None,
                }
                continue
            abs_change = cur_val - prev_val
            pct_change = (abs_change / prev_val) * 100.0
            direction = "flat"
            if not math.isclose(abs_change, 0.0, rel_tol=1e-9, abs_tol=1e-6):
                direction = "up" if abs_change > 0 else "down"
            result[key][f"{y}{q}"] = {
                "value": cur_val,
                "prev": prev_val,
                "abs_change": abs_change,
                "pct_change": pct_change,
                "direction": direction,
            }
    return result
def calc_debt_structure_summary(fs_rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Approximate debt structure summary from balance sheet accounts.

    Returns:
        Dict with total_liabilities, bond_balance, bond_ratio, short_term_debt,
        long_term_debt, short_term_ratio, long_term_ratio.
    """
    total_liabilities = find_amount(fs_rows, "BS", ["부채총계", "총부채"])
    bond_balance = find_amount(fs_rows, "BS", ["사채", "회사채", "전환사채", "신주인수권부사채"])

    short_term_candidates = [
        find_amount(fs_rows, "BS", ["단기차입금"]),
        find_amount(fs_rows, "BS", ["유동성장기부채"]),
        find_amount(fs_rows, "BS", ["단기사채"]),
    ]
    long_term_candidates = [
        find_amount(fs_rows, "BS", ["장기차입금"]),
        find_amount(fs_rows, "BS", ["사채", "회사채"]),
    ]

    short_term_debt = sum(x for x in short_term_candidates if x is not None)
    if all(x is None for x in short_term_candidates):
        short_term_debt = None

    long_term_debt = sum(x for x in long_term_candidates if x is not None)
    if all(x is None for x in long_term_candidates):
        long_term_debt = None

    bond_ratio = _safe_div(bond_balance, total_liabilities)
    short_ratio = _safe_div(short_term_debt, total_liabilities)
    long_ratio = _safe_div(long_term_debt, total_liabilities)

    return {
        "total_liabilities": total_liabilities,
        "bond_balance": bond_balance,
        "bond_ratio": bond_ratio,
        "short_term_debt": short_term_debt,
        "short_term_ratio": short_ratio,
        "long_term_debt": long_term_debt,
        "long_term_ratio": long_ratio,
    }


if __name__ == "__main__":
    # Example: Samsung Electronics 2021~2024 ratios and YoY changes
    client = OpenDartClient()
    samsungs = client.find_corp_by_name("삼성전자", exact=True)
    if not samsungs:
        raise SystemExit("삼성전자 corp_code not found. Ensure corpCode.xml is reachable.")

    corp_code = samsungs[0].corp_code
    ratios_by_year, missing_years = get_yearly_ratios(client, corp_code=corp_code, start_year=2021, end_year=2024)
    if missing_years:
        print(f"[INFO] Missing years (no DART data): {', '.join(map(str, missing_years))}")
    if not ratios_by_year:
        raise SystemExit("No ratios available in the requested range.")
    print("Basic ratios by year:")
    for yr, ratios in ratios_by_year.items():
        print(yr, ratios)

    yoy = calc_yoy_changes(ratios_by_year, ratio_keys=["current_ratio", "debt_ratio", "roa", "roe"])
    print("\nYoY changes:")
    for key, values in yoy.items():
        print(f"{key}: {values}")
