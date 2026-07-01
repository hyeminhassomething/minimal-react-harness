# recommend.py
# -----------------------------------------------------------------------------
# 시나리오: 온디바이스 "지원사업 추천" 에이전트.
#   사용자가 자기 상황(지역·나이·가구·소득)을 자연어로 말하면,
#   받을 수 있는 지원사업을 로컬 데이터셋에서 찾아 '코드'로 자격을 검증해 추천.
#   개인정보가 기기 밖으로 나가지 않도록 전부 로컬(외부 API 금지). 작은 모델 사용.
#
# 실행: python recommend.py -p "서울 사는 27살 1인가구 월소득 250만원, 주거 지원 알아봐줘"
# -----------------------------------------------------------------------------
import argparse
import datetime as _dt
import os

from react_agent import ReActAgent
from tools import TOOLS
from llm import HFLocalLLM

# 지원사업 시나리오 전용 프롬프트. {tools}는 엔진이 채운다.
# 자격 대조/계산은 반드시 도구가 하도록 지시(모델 암산 금지).
RECOMMEND_TEMPLATE = """너는 사용자의 상황을 듣고 받을 수 있는 정부/지자체 지원사업을 찾아주는 추천 에이전트다.
아래 형식을 반드시 지켜, 한 번에 Thought 하나와 Action 하나만 출력한다.

[형식]
Thought: (지금 무엇을 할지 한국어로 짧게 추론)
Action: 도구이름[입력]

[사용 가능한 도구]
{tools}

[규칙]
- 나이·소득·지역·가구 같은 자격 대조와 계산은 절대 네 머리로 하지 말고, 반드시 도구에 맡긴다.
- 먼저 dataset_search 로 후보를 찾고, 그 다음 check_eligibility 로 자격을 검증한다.
- 도구 결과(Observation)는 시스템이 채워주므로 절대 네가 지어내지 않는다.
- 검증 결과를 바탕으로 Action: finish[추천 결과] 로 끝낸다.
- Action 에는 반드시 도구이름 뒤에 대괄호 [ ] 로 입력을 적는다.

[예시] (형식만 참고. 실제 입력에는 아래 내용을 베끼지 말 것)
Question: 부산 사는 30살, 월소득 200만원인데 받을 수 있는 지원 있어?
Thought: 먼저 지역·나이로 후보를 찾자.
Action: dataset_search[부산 30살]
Observation: 후보 3건: P005|국민취업지원제도; P007|부산청년월세지원; ...
Thought: 후보의 자격요건을 소득까지 넣어 코드로 검증하자.
Action: check_eligibility[부산 30살 월소득 200만원]
Observation: 자격 충족 2건: - 부산청년월세지원: ...
Thought: 검증된 항목으로 답한다.
Action: finish[부산청년월세지원 등 2건을 추천합니다]
"""

DEFAULT_PROFILE = "서울 사는 27살 1인가구 월소득 250만원, 주거 지원 알아봐줘"


def _print_env():
    try:
        import torch
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[env] torch={torch.__version__} device={dev}")
    except Exception as e:
        print(f"[env] torch 확인 실패: {e}")


def main():
    p = argparse.ArgumentParser(description="온디바이스 지원사업 추천 에이전트")
    p.add_argument("-p", "--profile", default=DEFAULT_PROFILE,
                   help="사용자 상황(자연어). 예: '서울 27살 1인가구 월소득 250만원'")
    p.add_argument("-m", "--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--show-raw", action="store_true")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    _print_env()
    print(f"[model] {args.model} 로드 중...")
    llm = HFLocalLLM(model_name=args.model)
    agent = ReActAgent(llm, TOOLS, max_steps=args.max_steps,
                       system_template=RECOMMEND_TEMPLATE)

    print("=" * 60)
    print("입력:", args.profile)
    result = agent.run(args.profile)
    trace = agent.format_trace(result, show_raw=args.show_raw)
    print(trace)

    if args.save:
        os.makedirs("examples", exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("examples", f"recommend_{ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# model={args.model}\n입력: {args.profile}\n{trace}\n")
        print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
