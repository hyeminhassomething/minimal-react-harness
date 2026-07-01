# minimal-react-harness

작은 언어 모델(sLM)을 **두뇌**로 삼아, 스스로 **생각 → 도구 사용 → 결과 관찰**을 반복해
답을 찾는 **ReAct 에이전트 하네스(harness)**를 처음부터(from scratch) 구현한 예제입니다.
외부 API에 의존하지 않고, 모델을 **로컬에서 직접 실행**합니다.

> 학습/포트폴리오 목적으로, 프레임워크(예: smolagents) 없이 ReAct 루프의 동작 원리를
> 직접 코드로 드러내는 데 초점을 맞췄습니다.

## 핵심 개념 — ReAct

ReAct = **Re**asoning + **Acting**. 모델이 머릿속으로만 추론하지 않고, 중간에 실제로
도구를 사용하게 하는 패턴입니다. 세 가지가 반복됩니다.

```
Thought      : 지금 무엇을 할지 판단 (자연어)
Action       : 도구 호출            예) search[에펠탑]  /  calculate[2026 - 1889]
Observation  : 도구가 돌려준 결과 (시스템이 채워줌)
   ⟲ 충분히 풀릴 때까지 반복 → Action: finish[최종답]
```

모델은 **"무슨 도구를 언제 쓸지" 판단만** 하고, 사실 조회·계산은 도구가 책임집니다.
덕분에 작은 모델도 환각이 줄고, 자신이 모르는 정보·계산을 외부로 위임할 수 있습니다.

## 동작 예시 (실제 트레이스)

```
Q: 에펠탑은 올해(2026년) 기준 몇 년 됐어?

[1] Thought: 에펠탑 완공 연도를 정확히 모른다. 검색이 필요하다.
    Action: search[에펠탑]
    Observation: 에펠탑은 1889년에 완공되었다.
[2] Thought: 1889년이다. 2026에서 빼자. 암산은 틀릴 수 있으니 계산기를 쓰자.
    Action: calculate[2026 - 1889]
    Observation: 137
[3] Thought: 137이 나왔다. 이제 답할 수 있다.
    Action: finish[약 137년]
==> 최종 답: 약 137년
```

## 아키텍처

| 파일 | 역할 |
|------|------|
| `react_agent.py` | **하네스 엔진**. Thought/Action 파싱 → 도구 실행 → Observation 주입 → 반복. 형식 오류·무한 루프·max_steps 가드 포함 |
| `tools.py` | 도구 정의 (`search`, `calculate`). 도구는 평범한 파이썬 함수 |
| `llm.py` | 모델 백엔드 2종 — `MockLLM`(테스트용), `HFLocalLLM`(실제 로컬 모델) |
| `main.py` | 실제 모델로 실행하는 진입점 |
| `test_harness.py` | 가짜 모델로 엔진을 검증 (GPU 불필요) |

설계 포인트: **모델을 생성자에 주입**(dependency injection)하기 때문에, 동일한 하네스를
가짜 모델로 테스트하고 진짜 모델로 실행할 수 있습니다. 모델은 `(prompt, stop) -> str`
규약만 지키면 무엇이든 끼울 수 있습니다.

## 실행 방법

엔진만 빠르게 검증 (의존성·GPU 불필요):

```bash
python test_harness.py
```

실제 작은 로컬 모델로 실행:

```bash
pip install -r requirements.txt
python main.py
```

다른 모델로 교체하려면 `main.py`의 `model_name`만 바꾸면 됩니다.

## 작은 모델에서의 핵심 난점

작은 모델은 지식보다 **도구 호출 형식**에서 더 자주 실패합니다(괄호 누락, 잘못된 도구,
Observation을 스스로 지어내기 등). 이 하네스는 그 실패를 다음으로 방어합니다.

- `Observation:`에서 생성을 멈춰 모델이 결과를 날조하지 못하게 함
- Action 파싱 실패 시 형식 교정을 요구하는 Observation을 되돌려줌
- 동일 행동 반복 / max_steps 가드로 무한 루프 차단

## 한계 & 다음 단계

- `search`는 재현성을 위한 로컬 사전 **스텁**입니다. 실제 검색/RAG 리트리버로 교체 가능합니다.
- 도구 입력 파싱은 단순합니다(입력 안의 `]` 미지원). 더 견고히 하려면 JSON 스키마 기반 호출로 확장.
- 프로덕션 프레임워크로는 Hugging Face **smolagents**, **LangGraph** 등을 참고.

## 라이선스

MIT. 기본 예시 모델의 라이선스는 각 모델 카드를 확인하세요.
