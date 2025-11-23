# filing_agent_tools.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any

from opendart_client import OpenDartClient, CorpInfo


class FilingType(str, Enum):
    """
    DART 공시 구분값 (pblntf_ty)
    A: 정기공시(사업/분기/반기보고서 등)
    B: 주요사항보고
    C: 발행공시
    D: 지분공시
    E: 기타공시
    """
    REGULAR = "A"        # 정기공시
    MATERIAL = "B"       # 주요사항보고
    SECURITY = "C"       # 발행공시
    OWNERSHIP = "D"      # 지분공시
    OTHER = "E"          # 기타공시


@dataclass
class Filing:
    """에이전트가 쓰기 좋은 형태로 정리한 공시 한 건"""
    corp_code: str
    corp_name: str
    rcept_no: str
    report_nm: str
    rcept_dt: str      # YYYYMMDD
    flr_nm: str        # 제출인
    rm: Optional[str]  # 비고
    pblntf_ty: str     # A~E
    pblntf_detail_ty: Optional[str] = None

    @property
    def url(self) -> str:
        # DART 원문 조회 URL (브라우저에서 바로 보기)
        return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={self.rcept_no}"


def _date_to_yyyymmdd(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def _resolve_corp(
    client: OpenDartClient,
    corp_name: Optional[str] = None,
    stock_code: Optional[str] = None,
) -> CorpInfo:
    """
    에이전트가 회사 식별을 위해 사용하는 헬퍼:
    - 우선 종목코드(stock_code)로 찾고,
    - 없으면 회사명으로 검색.
    """
    if stock_code:
        info = client.find_corp_by_stock_code(stock_code)
        if not info:
            raise ValueError(f"종목코드 {stock_code}로 회사를 찾을 수 없습니다.")
        return info

    if not corp_name:
        raise ValueError("corp_name 또는 stock_code 둘 중 하나는 반드시 필요합니다.")

    matches = client.find_corp_by_name(corp_name, exact=False)
    if not matches:
        raise ValueError(f"회사명 '{corp_name}' 으로 회사를 찾을 수 없습니다.")

    # 정확히 일치하는 것이 있으면 그걸 우선
    exact = [m for m in matches if m.corp_name == corp_name]
    if exact:
        return exact[0]
    # 그 외에는 첫 번째 후보 사용 (필요시 에이전트가 후속 질문해서 disambiguation 가능)
    return matches[0]


def _search_filings_raw(
    client: OpenDartClient,
    *,
    corp_code: Optional[str],
    filing_type: FilingType,
    start_date: str,
    end_date: str,
    page_no: int = 1,
    page_count: int = 100,
    pblntf_detail_ty: Optional[str] = None,
) -> Dict[str, Any]:
    """
    list.json을 직접 호출하는 로우 헬퍼.
    에이전트는 보통 아래의 타입별 wrapper를 쓸 것.
    """
    return client.search_filings(
        corp_code=corp_code,
        bgn_de=start_date,
        end_de=end_date,
        pblntf_ty=filing_type.value,
        pblntf_detail_ty=pblntf_detail_ty,
        last_reprt_at=None,
        sort="date",        # 최신 공시 우선
        page_no=page_no,
        page_count=page_count,
    )


def _convert_to_filings(
    corp: Optional[CorpInfo],
    raw: Dict[str, Any]
) -> List[Filing]:
    """list.json 결과를 Filing dataclass 리스트로 변환."""
    corp_name_fallback = corp.corp_name if corp else ""
    items = raw.get("list") or []
    filings: List[Filing] = []

    for it in items:
        filings.append(
            Filing(
                corp_code=it.get("corp_code", corp.corp_code if corp else ""),
                corp_name=it.get("corp_name", corp_name_fallback),
                rcept_no=it.get("rcept_no", ""),
                report_nm=it.get("report_nm", ""),
                rcept_dt=it.get("rcept_dt", ""),
                flr_nm=it.get("flr_nm", ""),
                rm=it.get("rm") or None,
                pblntf_ty=it.get("pblntf_ty", ""),
                pblntf_detail_ty=it.get("pblntf_detail_ty") or None,
            )
        )
    return filings


# =============================================================================
# 1) 정기공시 (A) – 사업/분기/반기보고서 등
# =============================================================================
def get_regular_filings(
    client: OpenDartClient,
    *,
    corp_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    days: int = 365,
    limit: int = 50,
) -> List[Filing]:
    """
    A: 정기공시 (사업보고서, 분기/반기보고서 등)
    - 최근 N일(days) 동안의 공시를 가져온 뒤 limit 개수만 반환.
    """
    corp = None
    corp_code = None
    if corp_name or stock_code:
        corp = _resolve_corp(client, corp_name=corp_name, stock_code=stock_code)
        corp_code = corp.corp_code

    end = datetime.today()
    start = end - timedelta(days=days)

    raw = _search_filings_raw(
        client,
        corp_code=corp_code,
        filing_type=FilingType.REGULAR,
        start_date=_date_to_yyyymmdd(start),
        end_date=_date_to_yyyymmdd(end),
        page_no=1,
        page_count=min(limit, 100),
    )
    filings = _convert_to_filings(corp, raw)
    return filings[:limit]


# =============================================================================
# 2) 주요사항보고 (B) – 합병, 분할, 자산양수도 등
# =============================================================================
def get_material_event_filings(
    client: OpenDartClient,
    *,
    corp_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    days: int = 90,
    limit: int = 50,
    detail_code: Optional[str] = None,
) -> List[Filing]:
    """
    B: 주요사항보고
    - 합병, 분할, 자산양수도, 영업양수도 등
    - detail_code(세부유형 코드)를 알고 있으면 필터링 가능
    """
    corp = None
    corp_code = None
    if corp_name or stock_code:
        corp = _resolve_corp(client, corp_name=corp_name, stock_code=stock_code)
        corp_code = corp.corp_code

    end = datetime.today()
    start = end - timedelta(days=days)

    raw = _search_filings_raw(
        client,
        corp_code=corp_code,
        filing_type=FilingType.MATERIAL,
        start_date=_date_to_yyyymmdd(start),
        end_date=_date_to_yyyymmdd(end),
        page_no=1,
        page_count=min(limit, 100),
        pblntf_detail_ty=detail_code,
    )
    filings = _convert_to_filings(corp, raw)
    return filings[:limit]


# =============================================================================
# 3) 발행공시 (C) – 유상증자, 회사채 등 자금조달 관련
# =============================================================================
def get_security_filings(
    client: OpenDartClient,
    *,
    corp_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    days: int = 180,
    limit: int = 50,
    detail_code: Optional[str] = None,
) -> List[Filing]:
    """
    C: 발행공시
    - 유상증자, 무상증자, 회사채 발행 등 자금 조달 관련
    """
    corp = None
    corp_code = None
    if corp_name or stock_code:
        corp = _resolve_corp(client, corp_name=corp_name, stock_code=stock_code)
        corp_code = corp.corp_code

    end = datetime.today()
    start = end - timedelta(days=days)

    raw = _search_filings_raw(
        client,
        corp_code=corp_code,
        filing_type=FilingType.SECURITY,
        start_date=_date_to_yyyymmdd(start),
        end_date=_date_to_yyyymmdd(end),
        page_no=1,
        page_count=min(limit, 100),
        pblntf_detail_ty=detail_code,
    )
    filings = _convert_to_filings(corp, raw)
    return filings[:limit]


# =============================================================================
# 4) 지분공시 (D) – 대량보유/임원주식 거래 등
# =============================================================================
def get_ownership_filings(
    client: OpenDartClient,
    *,
    corp_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    days: int = 90,
    limit: int = 50,
    detail_code: Optional[str] = None,
) -> List[Filing]:
    """
    D: 지분공시
    - 대량보유 상황보고, 임원/주요주주 주식 거래 등
    """
    corp = None
    corp_code = None
    if corp_name or stock_code:
        corp = _resolve_corp(client, corp_name=corp_name, stock_code=stock_code)
        corp_code = corp.corp_code

    end = datetime.today()
    start = end - timedelta(days=days)

    raw = _search_filings_raw(
        client,
        corp_code=corp_code,
        filing_type=FilingType.OWNERSHIP,
        start_date=_date_to_yyyymmdd(start),
        end_date=_date_to_yyyymmdd(end),
        page_no=1,
        page_count=min(limit, 100),
        pblntf_detail_ty=detail_code,
    )
    filings = _convert_to_filings(corp, raw)
    return filings[:limit]


# =============================================================================
# 5) 기타공시 (E)
# =============================================================================
def get_other_filings(
    client: OpenDartClient,
    *,
    corp_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    days: int = 90,
    limit: int = 50,
    detail_code: Optional[str] = None,
) -> List[Filing]:
    """
    E: 기타공시
    - 배당관련 공시 일부, 기타 법령에 따른 공시 등
    """
    corp = None
    corp_code = None
    if corp_name or stock_code:
        corp = _resolve_corp(client, corp_name=corp_name, stock_code=stock_code)
        corp_code = corp.corp_code

    end = datetime.today()
    start = end - timedelta(days=days)

    raw = _search_filings_raw(
        client,
        corp_code=corp_code,
        filing_type=FilingType.OTHER,
        start_date=_date_to_yyyymmdd(start),
        end_date=_date_to_yyyymmdd(end),
        page_no=1,
        page_count=min(limit, 100),
        pblntf_detail_ty=detail_code,
    )
    filings = _convert_to_filings(corp, raw)
    return filings[:limit]


# =============================================================================
# 6) 에이전트용 통합 진입점 – 자연어 명령에서 바로 쓰기
# =============================================================================
def agent_get_filings_by_category(
    client: OpenDartClient,
    *,
    corp_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    category: str,
    days: int = 90,
    limit: int = 50,
    detail_code: Optional[str] = None,
) -> List[Filing]:
    """
    에이전트가 쓰기 좋은 진입점.
    category 예시:
        - "정기공시", "A"
        - "주요사항보고", "B"
        - "발행공시", "C"
        - "지분공시", "D"
        - "기타공시", "E"
    LLM이 category만 정해주면 이 함수 하나로 라우팅할 수 있다.
    """
    # 한글 → 코드 매핑
    mapping = {
        "정기공시": FilingType.REGULAR,
        "주요사항보고": FilingType.MATERIAL,
        "발행공시": FilingType.SECURITY,
        "지분공시": FilingType.OWNERSHIP,
        "기타공시": FilingType.OTHER,
        "A": FilingType.REGULAR,
        "B": FilingType.MATERIAL,
        "C": FilingType.SECURITY,
        "D": FilingType.OWNERSHIP,
        "E": FilingType.OTHER,
    }

    cat = category.strip().upper()
    if category in mapping:
        ftype = mapping[category]
    elif cat in mapping:
        ftype = mapping[cat]
    else:
        raise ValueError(f"알 수 없는 category: {category}")

    if ftype == FilingType.REGULAR:
        return get_regular_filings(
            client,
            corp_name=corp_name,
            stock_code=stock_code,
            days=days,
            limit=limit,
        )
    if ftype == FilingType.MATERIAL:
        return get_material_event_filings(
            client,
            corp_name=corp_name,
            stock_code=stock_code,
            days=days,
            limit=limit,
            detail_code=detail_code,
        )
    if ftype == FilingType.SECURITY:
        return get_security_filings(
            client,
            corp_name=corp_name,
            stock_code=stock_code,
            days=days,
            limit=limit,
            detail_code=detail_code,
        )
    if ftype == FilingType.OWNERSHIP:
        return get_ownership_filings(
            client,
            corp_name=corp_name,
            stock_code=stock_code,
            days=days,
            limit=limit,
            detail_code=detail_code,
        )
    if ftype == FilingType.OTHER:
        return get_other_filings(
            client,
            corp_name=corp_name,
            stock_code=stock_code,
            days=days,
            limit=limit,
            detail_code=detail_code,
        )

    # 여기까지 오지 않지만 safety용
    raise ValueError(f"지원하지 않는 category: {category}")
