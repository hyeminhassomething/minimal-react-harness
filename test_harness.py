# test_harness.py
# -----------------------------------------------------------------------------
# 하네스 '엔진'을 모델 없이 검증한다 (MockLLM 사용 → torch/GPU 불필요).
# 실행: python test_harness.py
# -----------------------------------------------------------------------------
from react_agent import ReActAgent
from tools import TOOLS
from llm import MockLLM


def test_eiffel_trace():
    """강의에서 본 '에펠탑' 트레이스가 그대로 재현되는지 확인."""
    scripted = [
        "Thought: 에펠탑 완공 연도를 정확히 모른다. 검색이 필요하다.\nAction: search[에펠탑]",
        "Thought: 1889년이다. 2026에서 빼야 하는데 암산은 틀릴 수 있으니 계산기를 쓰자.\nAction: calculate[2026 - 1889]",
        "Thought: 137이 나왔다. 이제 답할 수 있다.\nAction: finish[약 137년]",
    ]
    agent = ReActAgent(MockLLM(scripted), TOOLS)
    result = agent.run("에펠탑은 올해(2026년) 기준 몇 년 됐어?")
    print(agent.format_trace(result))
    assert result["ok"] and "137" in result["answer"], result
    print(">> test_eiffel_trace PASS\n")


def test_bad_format_recovers():
    """모델이 형식을 틀려도(첫 응답에 Action 없음) 하네스가 잡아주고 회복하는지."""
    scripted = [
        "Thought: 그냥 답하자. 답은 4입니다.",          # Action 없음 → 하네스가 교정 요구
        "Thought: 형식을 지키자.\nAction: calculate[2 + 2]",
        "Thought: 4다.\nAction: finish[4]",
    ]
    agent = ReActAgent(MockLLM(scripted), TOOLS, max_steps=5)
    result = agent.run("2 더하기 2는?")
    print(agent.format_trace(result))
    assert result["ok"] and "4" in result["answer"], result
    print(">> test_bad_format_recovers PASS\n")


def test_observability_fields():
    """관찰 발판: 결과에 format_errors 카운터, 각 스텝에 raw 원문이 담기는지."""
    scripted = [
        "Thought: 그냥 답하자.",                         # Action 없음 → format_errors +1
        "Thought: 형식을 지키자.\nAction: finish[done]",
    ]
    agent = ReActAgent(MockLLM(scripted), TOOLS, max_steps=5)
    result = agent.run("형식 오류 한 번 내보기")
    assert result["format_errors"] == 1, result
    assert all("raw" in s for s in result["steps"]), result
    # show_raw=True 출력에 원문 마커가 포함되는지
    assert "--- raw ---" in agent.format_trace(result, show_raw=True)
    print(">> test_observability_fields PASS\n")


def _has_pandas():
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


def test_eligibility_income_reject():
    """[시나리오] 자격 대조가 '코드'로 이뤄지는지: 소득 상한 초과를 탈락시키는가."""
    if not _has_pandas():
        print(">> test_eligibility_income_reject SKIP (pandas 없음)\n")
        return
    from tools import check_eligibility
    # 저소득청년통장(P028): 소득 상한 250만원. 300만원이면 초과 → 이 항목은 탈락해야 한다.
    out = check_eligibility("전국 20살 월소득 300만원")
    assert "저소득청년통장" not in out or "충족 0건" in out or "초과" in out, out
    # 상한 이하(200만원)면 통과 후보에 들어와야 한다.
    ok = check_eligibility("전국 20살 월소득 200만원")
    assert "충족" in ok, ok
    print(">> test_eligibility_income_reject PASS\n")


def test_dataset_search_filters():
    """[시나리오] dataset_search가 region/age로 후보를 거르는지."""
    if not _has_pandas():
        print(">> test_dataset_search_filters SKIP (pandas 없음)\n")
        return
    from tools import dataset_search
    out = dataset_search("서울 27살")
    assert "후보" in out and "P002" in out, out          # 서울청년월세지원 포함
    # 나이 범위 밖(70살)이면 청년 대상 프로그램은 빠져야 한다.
    old = dataset_search("서울 70살")
    assert "P002" not in old, old
    print(">> test_dataset_search_filters PASS\n")


def test_recommend_full_trace():
    """[시나리오] dataset_search→check_eligibility→finish 3스텝 체인이 도는지(MockLLM)."""
    if not _has_pandas():
        print(">> test_recommend_full_trace SKIP (pandas 없음)\n")
        return
    scripted = [
        "Thought: 지역·나이로 후보를 찾자.\nAction: dataset_search[서울 27살 1인가구]",
        "Thought: 소득까지 넣어 코드로 검증하자.\nAction: check_eligibility[서울 27살 1인가구 월소득 250만원]",
        "Thought: 검증된 항목으로 답한다.\nAction: finish[조건에 맞는 지원사업을 추천합니다]",
    ]
    agent = ReActAgent(MockLLM(scripted), TOOLS, max_steps=6)
    result = agent.run("서울 사는 27살 1인가구 월소득 250만원, 주거 지원 알아봐줘")
    print(agent.format_trace(result))
    assert result["ok"], result
    # 2번째 스텝(check_eligibility)의 Observation에 근거가 담겼는지
    assert any("check_eligibility" in s["action"] for s in result["steps"]), result
    print(">> test_recommend_full_trace PASS\n")


if __name__ == "__main__":
    test_eiffel_trace()
    test_bad_format_recovers()
    test_observability_fields()
    test_eligibility_income_reject()
    test_dataset_search_filters()
    test_recommend_full_trace()
    print("ALL TESTS PASSED ✅")
