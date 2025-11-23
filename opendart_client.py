"""
opendart_client.py

OpenDART API 모듈형 클라이언트

- 기업 식별(고유번호/종목코드)
- 기업개황(기본정보)
- 재무제표 / 주요계정 / 다중회사 비교
- 주주구조, 배당, 자기주식, 증자·감자
- 임원·주요주주 소유보고
- 공시검색 및 XBRL 원본파일 다운로드

참고:
- 공시검색: list.json :contentReference[oaicite:0]{index=0}
- 기업개황: company.json :contentReference[oaicite:1]{index=1}
- 고유번호 파일: corpCode.xml (zip+XML) :contentReference[oaicite:2]{index=2}
- 단일회사 전체 재무제표: fnlttSinglAcntAll.json :contentReference[oaicite:3]{index=3}
- 단일회사 주요계정: fnlttSinglAcnt.json :contentReference[oaicite:4]{index=4}
- 다중회사 주요계정: fnlttMultiAcnt.json :contentReference[oaicite:5]{index=5}
- 최대주주/변동/소액/임원 등: hyslrSttus, hyslrChgSttus, mrhlSttus, exctvSttus 등 :contentReference[oaicite:6]{index=6}
- 배당에 관한 사항: alotMatter.json :contentReference[oaicite:7]{index=7}
- 자기주식 취득·처분: tesstkAcqsDspsSttus.json :contentReference[oaicite:8]{index=8}
- 증자(감자) 현황: irdsSttus.json :contentReference[oaicite:9]{index=9}
- 임원·주요주주 소유보고: elestock.json :contentReference[oaicite:10]{index=10}
- 공시서류 원본(XBRL 포함): document.xml (zip) :contentReference[oaicite:11]{index=11}
"""

from __future__ import annotations
from dotenv import load_dotenv
import io
import os
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Any, Iterable

import requests

load_dotenv( override=False)
DART_BASE_URL = "https://opendart.fss.or.kr/api"


@dataclass
class CorpInfo:
    """고유번호 파일에서 추출한 기업 기본 식별정보"""
    corp_code: str
    corp_name: str
    stock_code: Optional[str] = None
    corp_eng_name: Optional[str] = None
    corp_cls: Optional[str] = None  # Y/K/N/E 등


class OpenDartClient:
    """
    OpenDART API 모듈형 클라이언트.

    에이전트가 쓰기 좋게 "기능별 메소드"를 제공한다.
    """

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        self.api_key = api_key or os.getenv("DART_API_KEY")
        if not self.api_key:
            raise ValueError("DART_API_KEY가 설정되지 않았습니다. 환경변수 또는 api_key 인자를 사용하세요.")
        self.session = session or requests.Session()

    def get_yearly_ratios(
        self,
        corp_code: str,
        start_year: int,
        end_year: int,
        reprt_code: str = "11011",
        fs_div: str = "CFS",
        verbose: bool = False,
    ) -> Dict[int, Dict[str, Optional[float]]]:
        """
        여러 년도에 걸쳐 기본 재무비율(유동비율, 부채비율, ROA, ROE 등)을 연도별로 계산.

        반환 형태:
        {
          2021: { "liquidity_ratio": ..., "debt_ratio": ..., ... },
          2022: { ... },
          2023: { ... },
        }
        """
        results: Dict[int, Dict[str, Optional[float]]] = {}
        for year in range(start_year, end_year + 1):
            try:
                fs_rows = self.get_single_fs_full(
                    corp_code=corp_code,
                    bsns_year=str(year),
                    reprt_code=reprt_code,
                    fs_div=fs_div,
                )
            except Exception as e:
                # 해당 연도 데이터 없거나 오류 시 스킵
                if verbose:
                    print(f"[WARN] {year}년 재무제표 조회 실패: {e}")
                continue

            if not fs_rows:
                continue

            ratios = self.calc_basic_ratios_from_fs(fs_rows)
            results[year] = ratios
        return results

    def calc_yoy_changes(
        self,
        yearly_ratios: Dict[int, Dict[str, Optional[float]]],
        ratio_keys: Optional[List[str]] = None,
    ) -> Dict[str, Dict[int, Dict[str, Optional[float]]]]:
        """
        연도별 재무비율(yearly_ratios)을 입력받아 전년 대비 변화율(%)을 계산.

        - yearly_ratios:
            get_yearly_ratios() 결과
        - ratio_keys:
            변화율을 계산할 지표 키 리스트 (기본값: 유동비율/부채비율/ROA/ROE)

        반환 형태:
        {
          "liquidity_ratio": {
            2022: {"value": 236.88, "prev": 200.0, "abs_change": 36.88, "pct_change": 18.44},
            2023: {...},
          },
          "debt_ratio": {
            2022: {...},
          },
          ...
        }
        """
        if ratio_keys is None:
            ratio_keys = ["liquidity_ratio", "debt_ratio", "roa", "roe"]

        # 연도 정렬
        years = sorted(yearly_ratios.keys())
        result: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {
            k: {} for k in ratio_keys
        }

        for i in range(1, len(years)):
            year = years[i]
            prev_year = years[i - 1]
            cur_data = yearly_ratios.get(year, {})
            prev_data = yearly_ratios.get(prev_year, {})

            for key in ratio_keys:
                cur_val = cur_data.get(key)
                prev_val = prev_data.get(key)

                if cur_val is None or prev_val in (None, 0):
                    result[key][year] = {
                        "value": cur_val,
                        "prev": prev_val,
                        "abs_change": None,
                        "pct_change": None,
                        "direction": None,  # "up", "down", "flat" 등
                    }
                    continue

                abs_change = cur_val - prev_val
                pct_change = (abs_change / prev_val) * 100.0

                if abs_change > 0:
                    direction = "up"
                elif abs_change < 0:
                    direction = "down"
                else:
                    direction = "flat"

                result[key][year] = {
                    "value": cur_val,
                    "prev": prev_val,
                    "abs_change": abs_change,
                    "pct_change": pct_change,
                    "direction": direction,
                }

        return result
    # -------------------------------------------------------------------------
    # 내부 공통 유틸
    # -------------------------------------------------------------------------
    def _request_json(self, path: str, **params) -> Dict[str, Any]:
        """JSON 응답을 리턴하는 일반 API 호출"""
        full_url = f"{DART_BASE_URL}/{path}"
        payload = {"crtfc_key": self.api_key}
        payload.update({k: v for k, v in params.items() if v is not None})

        resp = self.session.get(full_url, params=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")
        if status not in (None, "000"):
            raise RuntimeError(f"DART API error {status}: {data.get('message')}")
        return data

    def _request_binary(self, path: str, **params) -> bytes:
        """Zip 등 바이너리 응답을 리턴하는 API 호출 (corpCode.xml, document.xml 등)"""
        full_url = f"{DART_BASE_URL}/{path}"
        payload = {"crtfc_key": self.api_key}
        payload.update(params)

        resp = self.session.get(full_url, params=payload, timeout=60)
        resp.raise_for_status()
        return resp.content

    # -------------------------------------------------------------------------
    # 0. 고유번호(corpCode) 처리 모듈
    # -------------------------------------------------------------------------
    @lru_cache(maxsize=1)
    def get_corp_code_table(self) -> List[CorpInfo]:
        """
        DART 고유번호 전체 파일(corpCode.xml)을 다운로드하여 메모리에 로딩.

        반환: CorpInfo 객체 리스트 (상장/비상장 모두 포함)
        """
        raw = self._request_binary("corpCode.xml")
        # 에러 발생 시에는 zip이 아니라 XML로 status/message만 내려올 수 있음 :contentReference[oaicite:12]{index=12}
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                xml_bytes = zf.read("CORPCODE.xml")
        except zipfile.BadZipFile:
            # 에러 XML 파싱
            root = ET.fromstring(raw)
            status = root.findtext("status")
            msg = root.findtext("message")
            raise RuntimeError(f"corpCode 요청 실패 {status}: {msg}")

        root = ET.fromstring(xml_bytes)
        corps: List[CorpInfo] = []
        for elem in root.findall("list"):
            corp_code = (elem.findtext("corp_code") or "").strip()
            corp_name = (elem.findtext("corp_name") or "").strip()
            stock_code = (elem.findtext("stock_code") or "").strip() or None
            corp_eng_name = (elem.findtext("corp_eng_name") or "").strip() or None
            corp_cls = (elem.findtext("corp_cls") or "").strip() or None
            corps.append(
                CorpInfo(
                    corp_code=corp_code,
                    corp_name=corp_name,
                    stock_code=stock_code,
                    corp_eng_name=corp_eng_name,
                    corp_cls=corp_cls,
                )
            )
        return corps

    def find_corp_by_name(self, name: str, exact: bool = True) -> List[CorpInfo]:
        """
        회사명으로 고유번호 검색.

        exact=True  : 완전일치
        exact=False : 부분일치(대소문자 구분 X, 공백 포함 검색)
        """
        name_norm = name.strip()
        corps = self.get_corp_code_table()

        if exact:
            return [c for c in corps if c.corp_name == name_norm]
        # 부분 일치
        lower = name_norm.lower()
        return [c for c in corps if lower in c.corp_name.lower()]

    def find_corp_by_stock_code(self, stock_code: str) -> Optional[CorpInfo]:
        """
        종목코드(6자리)로 고유번호 검색. 없으면 None.
        """
        code = stock_code.strip()
        for c in self.get_corp_code_table():
            if c.stock_code == code:
                return c
        return None

    # -------------------------------------------------------------------------
    # 1. 기업개황 / 기본정보 모듈
    # -------------------------------------------------------------------------
    def get_company_overview(self, corp_code: str) -> Dict[str, Any]:
        """
        기업개황 API: 회사명, 대표자명, 설립일, 산업코드, 주소 등 기본 프로필 조회 :contentReference[oaicite:13]{index=13}
        """
        return self._request_json("company.json", corp_code=corp_code)

    # -------------------------------------------------------------------------
    # 2. 재무제표 / 재무비율 모듈
    # -------------------------------------------------------------------------
    def get_single_fs_full(
        self,
        corp_code: str,
        bsns_year: str,
        reprt_code: str = "11011",
        fs_div: str = "CFS",
    ) -> List[Dict[str, Any]]:
        """
        단일회사 전체 재무제표(fnlttSinglAcntAll.json) 조회 :contentReference[oaicite:14]{index=14}

        - corp_code: 고유번호 (8자리)
        - bsns_year: 사업연도 (예: '2023')
        - reprt_code:
            '11013': 1분기보고서
            '11012': 반기보고서
            '11014': 3분기보고서
            '11011': 사업보고서
        - fs_div:
            'CFS' : 연결
            'OFS' : 개별
        """
        data = self._request_json(
            "fnlttSinglAcntAll.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
            fs_div=fs_div,
        )
        return data.get("list", []) or []

    def get_single_fs_major_accounts(
        self,
        corp_code: str,
        bsns_year: str,
        reprt_code: str = "11011",
    ) -> List[Dict[str, Any]]:
        """
        단일회사 주요계정(fnlttSinglAcnt.json) 조회 – BS/IS 주요 계정값 :contentReference[oaicite:15]{index=15}
        """
        data = self._request_json(
            "fnlttSinglAcnt.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
        )
        return data.get("list", []) or []

    def get_multi_fs_major_accounts(
        self,
        corp_codes: Iterable[str],
        bsns_year: str,
        reprt_code: str = "11011",
    ) -> List[Dict[str, Any]]:
        """
        다중회사 주요계정(fnlttMultiAcnt.json) 조회 – 최대 100개 기업까지 한번에 비교 :contentReference[oaicite:16]{index=16}
        """
        corp_code_param = ",".join(corp_codes)
        data = self._request_json(
            "fnlttMultiAcnt.json",
            corp_code=corp_code_param,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
        )
        return data.get("list", []) or []

    @staticmethod
    def _find_amount(
        fs_rows: List[Dict[str, Any]],
        *,
        sj_div: Optional[str],
        account_name_candidates: List[str],
        use_acc_id: Optional[List[str]] = None,
        field: str = "thstrm_amount",
    ) -> Optional[float]:
        """
        전체 재무제표 row 리스트에서 특정 계정의 금액을 찾는 헬퍼.

        - sj_div: 'BS', 'IS', 'CIS', 'CF', 'SCE' 중 하나 또는 None
        - account_name_candidates: account_nm에 대해 우선순위로 매칭할 한글 명칭 목록
        - use_acc_id: XBRL 표준 account_id 후보 (있으면 account_id 우선 사용) :contentReference[oaicite:17]{index=17}
        """
        def norm(x: str) -> str:
            return (x or "").replace(" ", "").strip()

        candidate_ids = {norm(i) for i in (use_acc_id or [])}
        candidate_names = [norm(n) for n in account_name_candidates]

        # 1) account_id 기준 검색
        for row in fs_rows:
            if sj_div and row.get("sj_div") != sj_div:
                continue
            acc_id = norm(row.get("account_id", ""))
            if acc_id and acc_id in candidate_ids:
                val = row.get(field)
                if val is None or val == "":
                    continue
                try:
                    return float(str(val).replace(",", ""))
                except ValueError:
                    continue

        # 2) account_nm 기준 느슨한 매칭
        for row in fs_rows:
            if sj_div and row.get("sj_div") != sj_div:
                continue
            name = norm(row.get("account_nm", ""))
            for cand in candidate_names:
                if cand and name == cand:
                    val = row.get(field)
                    if val is None or val == "":
                        break
                    try:
                        return float(str(val).replace(",", ""))
                    except ValueError:
                        break

        # 3) 부분 포함 매칭(마지막 fallback)
        for row in fs_rows:
            if sj_div and row.get("sj_div") != sj_div:
                continue
            name = norm(row.get("account_nm", ""))
            for cand in candidate_names:
                if cand and cand in name:
                    val = row.get(field)
                    if val is None or val == "":
                        break
                    try:
                        return float(str(val).replace(",", ""))
                    except ValueError:
                        break
        return None

    def calc_basic_ratios_from_fs(
        self,
        fs_rows: List[Dict[str, Any]],
    ) -> Dict[str, Optional[float]]:
        """
        단일회사 전체 재무제표(fnlttSinglAcntAll.json) 결과로부터 기본 재무비율 계산.

        계산 항목:
        - 유동비율 = 유동자산 / 유동부채 * 100
        - 당좌비율 = (당좌자산 or (유동자산 - 재고자산)) / 유동부채 * 100
        - 부채비율 = 부채총계 / 자본총계 * 100
        - ROA = 당기순이익 / 자산총계
        - ROE = 당기순이익 / 자본총계 (단순 버전, 평균자본총계는 사용자가 직접 계산 권장)
        """
        # 재무상태표(BS)
        current_assets = self._find_amount(
            fs_rows,
            sj_div="BS",
            account_name_candidates=["유동자산"],
        )
        current_liabilities = self._find_amount(
            fs_rows,
            sj_div="BS",
            account_name_candidates=["유동부채"],
        )
        quick_assets = self._find_amount(
            fs_rows,
            sj_div="BS",
            account_name_candidates=["당좌자산"],
        )
        inventory = self._find_amount(
            fs_rows,
            sj_div="BS",
            account_name_candidates=["재고자산"],
        )
        total_liab = self._find_amount(
            fs_rows,
            sj_div="BS",
            account_name_candidates=["부채총계"],
        )
        total_equity = self._find_amount(
            fs_rows,
            sj_div="BS",
            account_name_candidates=["자본총계"],
        )
        total_assets = self._find_amount(
            fs_rows,
            sj_div="BS",
            account_name_candidates=["자산총계"],
        )

        # 손익계산서(IS)
        revenue = self._find_amount(
            fs_rows,
            sj_div="IS",
            account_name_candidates=["매출액", "수익"],
        )
        operating_income = self._find_amount(
            fs_rows,
            sj_div="IS",
            account_name_candidates=["영업이익"],
        )
        net_income = self._find_amount(
            fs_rows,
            sj_div="IS",
            account_name_candidates=["당기순이익", "당기순이익(손실)"],
        )

        def safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
            if num is None or den in (None, 0):
                return None
            return num / den

        liquidity_ratio = safe_ratio(current_assets, current_liabilities)
        quick_base = quick_assets
        if quick_base is None and current_assets is not None and inventory is not None:
            quick_base = current_assets - inventory
        quick_ratio = safe_ratio(quick_base, current_liabilities)

        debt_ratio = safe_ratio(total_liab, total_equity)
        roa = safe_ratio(net_income, total_assets)
        roe = safe_ratio(net_income, total_equity)

        return {
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "quick_assets": quick_assets,
            "inventory": inventory,
            "total_liabilities": total_liab,
            "total_equity": total_equity,
            "total_assets": total_assets,
            "revenue": revenue,
            "operating_income": operating_income,
            "net_income": net_income,
            "liquidity_ratio": None if liquidity_ratio is None else liquidity_ratio * 100,
            "quick_ratio": None if quick_ratio is None else quick_ratio * 100,
            "debt_ratio": None if debt_ratio is None else debt_ratio * 100,
            "roa": roa,
            "roe": roe,
        }

    # -------------------------------------------------------------------------
    # 3. 지배구조 / 주주 구조 모듈 (정기보고서 주요정보 + 지분공시 종합정보)
    # -------------------------------------------------------------------------
    def get_major_shareholders(
        self, corp_code: str, bsns_year: str, reprt_code: str = "11011"
    ) -> List[Dict[str, Any]]:
        """최대주주 현황 (hyslrSttus.json) :contentReference[oaicite:18]{index=18}"""
        data = self._request_json(
            "hyslrSttus.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
        )
        return data.get("list", []) or []

    def get_major_shareholders_changes(
        self, corp_code: str, bsns_year: str, reprt_code: str = "11011"
    ) -> List[Dict[str, Any]]:
        """최대주주 변동현황 (hyslrChgSttus.json) :contentReference[oaicite:19]{index=19}"""
        data = self._request_json(
            "hyslrChgSttus.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
        )
        return data.get("list", []) or []

    def get_minority_shareholders(
        self, corp_code: str, bsns_year: str, reprt_code: str = "11011"
    ) -> List[Dict[str, Any]]:
        """소액주주 현황 (mrhlSttus.json) :contentReference[oaicite:20]{index=20}"""
        data = self._request_json(
            "mrhlSttus.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
        )
        return data.get("list", []) or []

    def get_executives(
        self, corp_code: str, bsns_year: str, reprt_code: str = "11011"
    ) -> List[Dict[str, Any]]:
        """임원 현황 (exctvSttus.json) :contentReference[oaicite:21]{index=21}"""
        data = self._request_json(
            "exctvSttus.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
        )
        return data.get("list", []) or []

    def get_executive_and_major_holder_ownership(
        self, corp_code: str
    ) -> List[Dict[str, Any]]:
        """
        임원·주요주주 소유보고 (elestock.json) – 지분공시 종합정보 :contentReference[oaicite:22]{index=22}

        bsns_year / reprt_code 없이 전체 소유보고 이력이 내려온다.
        """
        data = self._request_json(
            "elestock.json",
            corp_code=corp_code,
        )
        return data.get("list", []) or []

    # -------------------------------------------------------------------------
    # 4. 배당 / 자본 구조 / 자기주식 모듈
    # -------------------------------------------------------------------------
    def get_dividend_info(
        self, corp_code: str, bsns_year: str, reprt_code: str = "11011"
    ) -> List[Dict[str, Any]]:
        """
        배당에 관한 사항 (alotMatter.json) – 배당금, 배당률, 배당성향 등 :contentReference[oaicite:23]{index=23}
        """
        data = self._request_json(
            "alotMatter.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
        )
        return data.get("list", []) or []

    def get_capital_change(
        self, corp_code: str, bsns_year: str, reprt_code: str = "11011"
    ) -> List[Dict[str, Any]]:
        """
        증자(감자) 현황 (irdsSttus.json) :contentReference[oaicite:24]{index=24}
        """
        data = self._request_json(
            "irdsSttus.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
        )
        return data.get("list", []) or []

    def get_treasury_stock_changes(
        self, corp_code: str, bsns_year: str, reprt_code: str = "11011"
    ) -> List[Dict[str, Any]]:
        """
        자기주식 취득 및 처분 현황 (tesstkAcqsDspsSttus.json) :contentReference[oaicite:25]{index=25}
        """
        data = self._request_json(
            "tesstkAcqsDspsSttus.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
        )
        return data.get("list", []) or []

    def get_total_stock_status(
        self, corp_code: str, bsns_year: str, reprt_code: str = "11011"
    ) -> List[Dict[str, Any]]:
        """
        주식의 총수 현황 (stockTotqySttus.json)
        - 발행주식수, 보통주/우선주 구분 등 :contentReference[oaicite:26]{index=26}
        """
        data = self._request_json(
            "stockTotqySttus.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
        )
        return data.get("list", []) or []

    # -------------------------------------------------------------------------
    # 5. 공시검색 / 모니터링 모듈
    # -------------------------------------------------------------------------
    def search_filings(
        self,
        corp_code: Optional[str] = None,
        bgn_de: Optional[str] = None,
        end_de: Optional[str] = None,
        pblntf_ty: Optional[str] = None,
        pblntf_detail_ty: Optional[str] = None,
        last_reprt_at: Optional[str] = None,
        sort: Optional[str] = None,
        page_no: Optional[int] = None,
        page_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        공시검색(list.json) – 특정 기간, 회사, 공시유형별 공시 목록 조회 :contentReference[oaicite:27]{index=27}

        주요 파라미터:
        - corp_code: 고유번호(없으면 전체)
        - bgn_de, end_de: YYYYMMDD (기간 지정)
        - pblntf_ty:
            A: 정기공시(사업/분기/반기보고서 등)
            B: 주요사항보고
            C: 발행공시
            D: 지분공시
            E: 기타공시
        - pblntf_detail_ty: 세부 유형 코드 (예: A001 = 사업보고서 등 :contentReference[oaicite:28]{index=28})
        - last_reprt_at: 'Y'인 경우 최종보고서만
        """
        return self._request_json(
            "list.json",
            corp_code=corp_code,
            bgn_de=bgn_de,
            end_de=end_de,
            pblntf_ty=pblntf_ty,
            pblntf_detail_ty=pblntf_detail_ty,
            last_reprt_at=last_reprt_at,
            sort=sort,
            page_no=page_no,
            page_count=page_count,
        )

    # -------------------------------------------------------------------------
    # 6. 공시서류 원본(XBRL 포함) 다운로드 모듈
    # -------------------------------------------------------------------------
    def download_document_zip(self, rcept_no: str) -> bytes:
        """
        공시서류원본파일(document.xml) – ZIP 바이너리 그대로 반환 :contentReference[oaicite:29]{index=29}

        - rcept_no: 접수번호(14자리)
        - 반환: zip 파일 bytes (호출 측에서 파일로 저장하거나 zipfile.ZipFile로 처리)
        """
        return self._request_binary("document.xml", rcept_no=rcept_no)

    def extract_xbrl_from_document(
        self, doc_zip_bytes: bytes
    ) -> Dict[str, bytes]:
        """
        document.xml ZIP 안에서 XBRL / XML 파일들을 메모리상 dict로 반환.

        key: 파일명 (예: 'xbrl/....xbrl', '....xml')
        val: 파일 바이너리

        Arelle 등 외부 라이브러리로 파싱할 때 이 dict를 넘겨 사용하면 된다.
        """
        result: Dict[str, bytes] = {}
        with zipfile.ZipFile(io.BytesIO(doc_zip_bytes)) as zf:
            for info in zf.infolist():
                # 재무제표 관련 XBRL/XML만 필터링할 수도 있음 (원하면 호출자가 필터링)
                if info.filename.lower().endswith((".xbrl", ".xml", ".htm", ".html")):
                    result[info.filename] = zf.read(info.filename)
        return result


# -----------------------------------------------------------------------------
# 사용 예시 (에이전트가 호출하는 형태를 가정한 샘플)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 환경변수에 DART_API_KEY가 있다고 가정
    client = OpenDartClient()

    # 1) 회사 검색
    samsungs = client.find_corp_by_name("다날")
    if not samsungs:
        raise SystemExit("삼성전자를 찾을 수 없습니다.")
    samsung = samsungs[0]
    print("삼성전자 corp_code:", samsung.corp_code, "stock_code:", samsung.stock_code)

    # 2) 기업개황
    overview = client.get_company_overview(samsung.corp_code)
    print("대표자명:", overview.get("ceo_nm"))

    # 3) 2023년 사업보고서 기준 전체 재무제표 + 기본 재무비율
    fs_rows = client.get_single_fs_full(samsung.corp_code, bsns_year="2023", reprt_code="11011", fs_div="CFS")
    ratios = client.calc_basic_ratios_from_fs(fs_rows)
    print("유동비율:", ratios["liquidity_ratio"])
    print("부채비율:", ratios["debt_ratio"])
    print("ROE:", ratios["roe"])

    # 4) 최대주주 및 소액주주 현황
    major = client.get_major_shareholders(samsung.corp_code, bsns_year="2023")
    minor = client.get_minority_shareholders(samsung.corp_code, bsns_year="2023")
    print("최대주주 건수:", len(major))
    print("소액주주 건수:", len(minor))

    # 5) 최근 3개월간 사업보고서/분기보고서 공시 검색
    filings = client.search_filings(
        corp_code=samsung.corp_code,
        bgn_de="20250101",
        end_de="20251231",
        pblntf_ty="A",
        last_reprt_at="Y",
    )
    print("정기보고서 검색 결과 건수:", filings.get("total_count"))

    # 6) 첫 번째 공시의 XBRL 원본 ZIP 다운로드 및 추출
    if filings.get("list"):
        first = filings["list"][0]
        rcept_no = first["rcept_no"]
        zip_bytes = client.download_document_zip(rcept_no)
        xbrl_files = client.extract_xbrl_from_document(zip_bytes)
        print("ZIP 내 파일 수:", len(xbrl_files))
        # 여기서 Arelle 등을 이용해 XBRL 파싱 가능
