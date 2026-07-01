# main.py
# -----------------------------------------------------------------------------
# 실제 작은 로컬 모델로 ReAct 하네스를 돌리는 진입점.
# 실행: python main.py                      (기본 질문 2개)
#       python main.py -q "에펠탑은 ..." --show-raw --save
# 최초 1회 모델 다운로드 발생. GPU(CUDA) 권장, Apple Silicon은 MPS로 동작.
# -----------------------------------------------------------------------------
import argparse
import datetime as _dt
import os

from react_agent import ReActAgent
from tools import TOOLS
from llm import HFLocalLLM

DEFAULT_QUESTIONS = [
    "에펠탑은 올해(2026년) 기준 몇 년 됐어?",
    "경복궁 창건 연도에 600을 더하면 몇 년이야?",
]


def _print_env():
    """추론 디바이스 진단 한 줄 — 어디서(CUDA/MPS/CPU) 도는지 관찰용."""
    try:
        import torch
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[env] torch={torch.__version__} device={dev} "
              f"(cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()})")
    except Exception as e:
        print(f"[env] torch 확인 실패: {e}")


def main():
    p = argparse.ArgumentParser(description="minimal-react-harness 실행")
    p.add_argument("-q", "--question", help="단일 질문(없으면 기본 질문 세트 실행)")
    p.add_argument("-m", "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace 모델 이름(작은 로컬 모델)")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--show-raw", action="store_true",
                   help="각 스텝에서 모델 원문(raw)을 함께 출력")
    p.add_argument("--save", action="store_true",
                   help="트레이스를 examples/trace_<시각>.txt 로 저장")
    args = p.parse_args()

    _print_env()
    print(f"[model] {args.model} 로드 중...")
    llm = HFLocalLLM(model_name=args.model)         # 모델 주입(DI)
    agent = ReActAgent(llm, TOOLS, max_steps=args.max_steps)

    questions = [args.question] if args.question else DEFAULT_QUESTIONS
    outputs = []
    for q in questions:
        print("=" * 60)
        print("Q:", q)
        result = agent.run(q)
        trace = agent.format_trace(result, show_raw=args.show_raw)
        print(trace)
        outputs.append(f"{'=' * 60}\nQ: {q}\n{trace}\n")

    if args.save:
        os.makedirs("examples", exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("examples", f"trace_{ts}.txt")
        header = f"# model={args.model}\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(outputs))
        print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
