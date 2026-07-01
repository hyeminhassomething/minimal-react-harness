# tools.py
# -----------------------------------------------------------------------------
# 에이전트가 호출할 "도구(tool)"들을 정의한다.
# 핵심 원칙: 도구는 그냥 평범한 파이썬 함수다. 모델은 "어떤 도구를 언제 쓸지"만
#            판단하고, 실제 사실 조회/계산은 이 함수들이 책임진다.
# -----------------------------------------------------------------------------
import ast
import operator

# --- 도구 1) calculate : 숫자 계산을 정확히 (모델 암산 대신) -------------------
# eval()은 보안상 위험하므로, ast로 수식을 파싱해 허용된 연산만 직접 수행한다.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("허용되지 않은 식")

def calculate(expr: str) -> str:
    """사칙연산/거듭제곱/나머지를 정확히 계산. 예: calculate[2026 - 1889]"""
    try:
        return str(_eval(ast.parse(expr, mode="eval").body))
    except Exception as e:
        return f"계산 오류: {e}"

# --- 도구 2) search : 로컬 지식베이스 조회 (외부 API/네트워크 없음) ------------
# ※ 재현성을 위해 작은 사전(dict)으로 만든 '스텁'이다. 실제 프로젝트에서는
#   여기를 진짜 검색/RAG 리트리버로 갈아끼우면 된다. (인터페이스는 그대로)
_KB = {
    "에펠탑": "에펠탑은 1889년에 완공되었다.",
    "경복궁": "경복궁은 1395년에 창건되었다.",
    "파이썬": "파이썬(Python)은 1991년에 처음 공개되었다.",
}

def search(query: str) -> str:
    """로컬 지식베이스에서 사실을 찾는다. 예: search[에펠탑]"""
    for key, fact in _KB.items():
        if key in query:
            return fact
    return "검색 결과 없음 (로컬 지식베이스에 해당 항목이 없습니다)."

# =============================================================================
# 시나리오: 온디바이스 "지원사업 추천" — 자격 대조/계산은 모델이 아니라 '코드'가 한다.
# 모델은 자연어 조건을 도구에 넘기기만 하고, 필터링·자격검증·근거생성은 아래 함수들이 책임.
# =============================================================================
import os
import re

_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "programs.csv")
_PROGRAMS = None  # pandas DataFrame 캐시 (최초 1회 로드)

# income_max/age_max에 쓰는 '무관(상한 없음)' 관례값
_NO_LIMIT_INCOME = 999999
_NO_LIMIT_AGE = 200


def _load_programs():
    """programs.csv를 pandas로 1회 로드해 캐시. pandas는 여기서만 지연 import."""
    global _PROGRAMS
    if _PROGRAMS is None:
        import pandas as pd
        _PROGRAMS = pd.read_csv(_CSV_PATH)
    return _PROGRAMS


def _parse_profile(text: str) -> dict:
    """사용자 발화/조건 문자열을 표준 프로필 dict로 정규화(코드로 처리).

    지원 형태(둘 다):
      - 자연어: "서울 사는 27살 1인가구 월소득 250만원, 주거 지원 알아봐줘"
      - 반정형: "지역=서울, 나이=27, 가구=1인, 소득=250, 관심=주거"
    누락 필드는 None(=무관)으로 둔다.
    """
    p = {"region": None, "age": None, "income": None, "household": None, "interest": None}

    # region: 알려진 지역 키워드
    for r in ["서울", "경기", "부산", "전국"]:
        if r in text:
            p["region"] = r
            break

    # age: '27살' '27세' '나이=27' 등에서 첫 숫자
    m = re.search(r"(?:나이\s*=?\s*)?(\d{1,3})\s*(?:살|세)", text) or re.search(r"나이\s*=?\s*(\d{1,3})", text)
    if m:
        p["age"] = int(m.group(1))

    # income: '월소득 250' '소득=250' '250만원' (단위: 만원)
    m = re.search(r"(?:월?\s*소득\s*=?\s*)(\d{2,5})", text) or re.search(r"(\d{2,5})\s*만원", text)
    if m:
        p["income"] = int(m.group(1))

    # household: 가구 형태 키워드
    for h in ["1인", "한부모", "다자녀", "다문화"]:
        if h in text:
            p["household"] = h
            break

    # interest: 관심 분야(선택) — 검색/근거 표시에만 사용
    for k in ["주거", "월세", "일자리", "취업", "금융", "자산", "돌봄", "양육"]:
        if k in text:
            p["interest"] = k
            break

    return p


def dataset_search(arg: str) -> str:
    """조건에 맞는 지원사업 후보를 programs.csv에서 1차 필터링한다.
    region(사용자 지역 또는 전국) + age 범위로만 좁힌다(소득/가구는 검증 단계에서).
    사용법: dataset_search[서울 27살 1인가구]
    """
    try:
        df = _load_programs()
    except Exception as e:
        return f"데이터 로드 오류: {e}"
    prof = _parse_profile(arg)

    cand = df
    if prof["region"]:
        cand = cand[(cand["region"] == prof["region"]) | (cand["region"] == "전국")]
    if prof["age"] is not None:
        cand = cand[(cand["age_min"] <= prof["age"]) & (prof["age"] <= cand["age_max"])]

    if len(cand) == 0:
        return "후보 없음 (region/age 조건에 맞는 지원사업이 없습니다)."
    rows = [f"{r.id}|{r.name}" for r in cand.itertuples()]
    return f"후보 {len(rows)}건: " + "; ".join(rows)


def _eval_program(row, prof) -> tuple:
    """프로그램 1건을 프로필과 대조 → (충족여부, 사유목록). 전부 코드로 판정."""
    reasons, fails = [], []

    # region
    if prof["region"] and row.region not in (prof["region"], "전국"):
        fails.append(f"지역 불일치(대상 {row.region})")
    elif prof["region"]:
        reasons.append(f"지역 {row.region} 해당")

    # age
    if prof["age"] is not None:
        if row.age_min <= prof["age"] <= row.age_max:
            hi = "무관" if row.age_max >= _NO_LIMIT_AGE else row.age_max
            reasons.append(f"나이 {prof['age']}세가 {row.age_min}~{hi} 범위 내")
        else:
            fails.append(f"나이 범위({row.age_min}~{row.age_max}) 벗어남")

    # income (만원). 상한 이하이면 충족
    if prof["income"] is not None:
        if prof["income"] <= row.income_max:
            if row.income_max < _NO_LIMIT_INCOME:
                reasons.append(f"소득 {prof['income']}만원 ≤ 상한 {row.income_max}만원")
        else:
            fails.append(f"소득 상한 {row.income_max}만원 초과")

    # household: 프로그램이 특정 가구형태(무관 아님)를 요구하면 대조
    if row.household != "무관":
        if prof["household"] == row.household:
            reasons.append(f"{row.household} 가구 요건 충족")
        else:
            fails.append(f"{row.household} 가구 대상")

    return (len(fails) == 0, reasons, fails)


def check_eligibility(arg: str) -> str:
    """모든 지원사업을 자격요건과 대조해 '충족 항목 + 근거'를 돌려준다(코드가 판정).
    충족 항목이 없으면 가장 근접한 1건과 탈락 사유를 설명한다.
    사용법: check_eligibility[서울 27살 1인가구 월소득 250만원]
    """
    try:
        df = _load_programs()
    except Exception as e:
        return f"데이터 로드 오류: {e}"
    prof = _parse_profile(arg)

    eligible, near = [], []
    for row in df.itertuples():
        ok, reasons, fails = _eval_program(row, prof)
        if ok:
            eligible.append((row, reasons))
        else:
            near.append((row, reasons, fails))

    if eligible:
        lines = [f"자격 충족 {len(eligible)}건:"]
        for row, reasons in eligible:
            why = ", ".join(reasons) if reasons else "조건 무관"
            lines.append(f"- {row.name}: {row.benefit} (근거: {why}) [{row.apply_url}]")
        return "\n".join(lines)

    # 충족 0건 → 가장 근접한(탈락 사유 최소) 1건
    if near:
        near.sort(key=lambda x: len(x[2]))
        row, reasons, fails = near[0]
        return (f"자격 충족 0건. 가장 근접: {row.name} — 탈락 사유: {', '.join(fails)}"
                + (f" (충족: {', '.join(reasons)})" if reasons else ""))
    return "대조할 지원사업이 없습니다."


# --- 도구 레지스트리 ----------------------------------------------------------
# 이름 -> {함수, 설명}. 설명은 그대로 시스템 프롬프트에 들어가 모델에게 노출된다.
TOOLS = {
    "search":    {"fn": search,    "desc": "로컬 지식베이스에서 사실을 찾는다. 사용법: search[질의어]"},
    "calculate": {"fn": calculate, "desc": "사칙연산을 정확히 계산한다. 사용법: calculate[수식]"},
    "dataset_search":    {"fn": dataset_search,
                          "desc": "조건(지역·나이)에 맞는 지원사업 후보를 찾는다. 사용법: dataset_search[서울 27살 1인가구]"},
    "check_eligibility": {"fn": check_eligibility,
                          "desc": "후보의 자격요건(나이·소득·지역·가구)을 조건과 대조해 근거와 함께 추천한다. 사용법: check_eligibility[서울 27살 1인가구 월소득 250만원]"},
}

def render_tools() -> str:
    """프롬프트에 넣을 도구 목록 문자열."""
    return "\n".join(f"- {name}: {t['desc']}" for name, t in TOOLS.items())
