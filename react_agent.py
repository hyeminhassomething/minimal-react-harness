# react_agent.py
# -----------------------------------------------------------------------------
# ReAct 하네스(harness)의 '엔진'.
#   - 모델이 'Thought / Action'을 출력하면
#   - Action을 파싱해 실제 도구를 실행하고
#   - 결과를 'Observation'으로 다시 모델에게 넣어주고
#   - 끝(finish)나거나 한계(max_steps)에 도달할 때까지 반복한다.
# 모델은 생성자에 '주입'된다 (llm 인자). 그래서 가짜 모델로도, 진짜 작은 모델로도
# 똑같이 돌아간다. ← 테스트 가능성 + 포폴 어필 포인트.
# -----------------------------------------------------------------------------
import re
from tools import render_tools

FINISH = "finish"
# "Action: 도구이름[입력]" 을 잡는 정규식 (입력 안의 줄바꿈도 허용)
ACTION_RE = re.compile(r"Action:\s*([A-Za-z_]\w*)\s*\[(.*?)\]", re.DOTALL)
THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?:\nAction:|$)", re.DOTALL)

SYSTEM_TEMPLATE = """너는 도구를 사용할 수 있는 추론 에이전트다.
아래 형식을 반드시 지켜, 한 번에 Thought 하나와 Action 하나만 출력한다.

[형식]
Thought: (지금 무엇을 할지 한국어로 짧게 추론)
Action: 도구이름[입력]

[사용 가능한 도구]
{tools}

[규칙]
- 사실이 필요하면 search, 계산이 필요하면 calculate를 호출한다.
- 도구 결과(Observation)는 시스템이 채워주므로 절대 네가 지어내지 않는다.
- 최종 답을 낼 수 있으면 Action: finish[최종답] 으로 끝낸다.
- Action 에는 반드시 도구이름 뒤에 대괄호 [ ] 로 입력을 적는다. 입력 없는 'Action: search' 는 금지.

[예시] (형식만 참고. 실제 질문에는 아래 내용을 베끼지 말 것)
Question: 파이썬은 올해(2026년) 기준 몇 살이야?
Thought: 파이썬 공개 연도를 모르니 search로 찾는다.
Action: search[파이썬]
Observation: 파이썬(Python)은 1991년에 처음 공개되었다.
Thought: 1991년이다. 2026에서 빼야 하니 calculate를 쓴다.
Action: calculate[2026 - 1991]
Observation: 35
Thought: 35가 나왔으니 답할 수 있다.
Action: finish[약 35살]
"""


class ReActAgent:
    def __init__(self, llm, tools, max_steps: int = 6, system_template: str = None):
        # llm: (prompt:str, stop:list[str]) -> str  형태의 호출 가능 객체
        # system_template: 시나리오별 프롬프트 주입(None이면 기본=에펠탑용). {tools}만 채운다.
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.system = (system_template or SYSTEM_TEMPLATE).format(tools=render_tools())

    def run(self, question: str) -> dict:
        """질문 하나를 ReAct 루프로 풀고, 답 + 전체 트레이스를 반환."""
        scratch = f"Question: {question}\n"   # 누적되는 작업 기록(scratchpad)
        steps = []
        last_action = None
        format_errors = 0                     # 형식 오류 누적 카운터(관찰용)

        for _ in range(self.max_steps):
            # 1) 모델에게 다음 'Thought/Action' 블록을 생성하게 한다.
            #    'Observation:'에서 멈춰, 모델이 결과를 멋대로 지어내지 못하게 막는다.
            prompt = self.system + "\n" + scratch
            block = self.llm(prompt, stop=["Observation:"])
            scratch += block.strip() + "\n"

            tm = THOUGHT_RE.search(block)
            thought = tm.group(1).strip() if tm else ""

            # 2) Action 파싱 — 작은 모델은 형식을 자주 틀린다(=하네스가 잡아줘야 함)
            #    'raw'에 모델 원문을 그대로 보관 → 어디서 깨졌는지 관찰 가능.
            m = ACTION_RE.search(block)
            if not m:
                format_errors += 1
                obs = "오류: 'Action: 도구[입력]' 형식을 찾지 못했습니다. 형식을 지켜 다시 출력하세요."
                scratch += f"Observation: {obs}\n"
                steps.append({"thought": thought, "action": "(형식 오류)",
                              "observation": obs, "raw": block})
                continue

            tool, arg = m.group(1), m.group(2).strip()

            # 3) 종료 신호
            if tool == FINISH:
                steps.append({"thought": thought, "action": f"finish[{arg}]",
                              "observation": None, "raw": block})
                return {"ok": True, "answer": arg, "steps": steps,
                        "transcript": scratch, "format_errors": format_errors}

            # 4) 무한 루프 가드: 같은 행동을 연속 반복하면 중단
            if (tool, arg) == last_action:
                steps.append({"thought": thought, "action": f"{tool}[{arg}]",
                              "observation": "동일 행동 반복 감지 — 중단", "raw": block})
                return {"ok": False, "answer": None, "steps": steps,
                        "transcript": scratch, "format_errors": format_errors}
            last_action = (tool, arg)

            # 5) 도구 실행 → Observation
            if tool not in self.tools:
                obs = f"오류: 알 수 없는 도구 '{tool}'. 사용 가능: {', '.join(self.tools)}"
            else:
                obs = self.tools[tool]["fn"](arg)
            scratch += f"Observation: {obs}\n"
            steps.append({"thought": thought, "action": f"{tool}[{arg}]",
                          "observation": obs, "raw": block})

        # max_steps 도달 → 답 확정 실패
        return {"ok": False, "answer": None, "steps": steps,
                "transcript": scratch, "format_errors": format_errors}

    @staticmethod
    def format_trace(result: dict, show_raw: bool = False) -> str:
        """트레이스를 사람이 읽기 좋게(=README/포폴 캡처용) 출력.

        show_raw=True 면 각 스텝에 모델 '원문(raw)'을 함께 찍는다.
        → 파싱이 깨진 지점에서 모델이 실제로 무엇을 뱉었는지 관찰할 때 유용.
        """
        lines = []
        for i, s in enumerate(result["steps"], 1):
            lines.append(f"[{i}] Thought: {s['thought']}")
            lines.append(f"    Action: {s['action']}")
            if s.get("observation") is not None:
                lines.append(f"    Observation: {s['observation']}")
            if show_raw and s.get("raw") is not None:
                raw = s["raw"].strip().replace("\n", "\n      | ")
                lines.append(f"    --- raw ---\n      | {raw}")
        if result["ok"]:
            lines.append(f"==> 최종 답: {result['answer']}")
        else:
            lines.append("==> 최종 답 확정 실패 (형식 오류 반복 또는 max_steps 도달)")
        # 형식 오류 횟수 요약(관찰 지표) — 작은 모델일수록 이 값이 커진다.
        fe = result.get("format_errors", 0)
        lines.append(f"[형식 오류 {fe}회]")
        return "\n".join(lines)
