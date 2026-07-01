# llm.py
# -----------------------------------------------------------------------------
# 하네스에 '주입'할 모델 백엔드.
#   - MockLLM     : 미리 짜둔 응답을 순서대로 반환. GPU/라이브러리 불필요.
#                   => 하네스 엔진(파싱/도구실행/루프)을 빠르게 테스트할 때 사용.
#   - HFLocalLLM  : Hugging Face의 작은 모델을 '로컬'에서 직접 실행 (외부 API 아님).
# 두 백엔드 모두 동일한 호출 규약: __call__(prompt, stop) -> str
# 그리고 둘 다 'Thought:'로 시작하는 블록을 반환한다.
# -----------------------------------------------------------------------------


class MockLLM:
    """미리 정의된 응답을 순서대로 돌려주는 가짜 모델 (테스트/데모용)."""
    def __init__(self, scripted_responses):
        self.scripted = list(scripted_responses)
        self.i = 0

    def __call__(self, prompt: str, stop=None) -> str:
        resp = self.scripted[self.i]
        self.i += 1
        if stop:
            for s in stop:
                resp = resp.split(s)[0]
        return resp


class HFLocalLLM:
    """작은 오픈 모델을 로컬에서 직접 실행하는 백엔드.

    transformers/torch는 무거우므로 여기서만 '지연 import' 한다.
    (덕분에 MockLLM 테스트는 이들 없이도 돌아간다.)

    ※ 대회 제출용이라면 규칙에 맞는(공개일/라이선스/파라미터) 모델로 교체할 것.
      여기 기본값은 학습/포폴 실습용 예시다.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str, stop=None) -> str:
        # 'Thought:'를 미리 붙여 모델이 곧장 생각을 이어쓰게 유도(작은 모델 안정화)
        text = prompt + "Thought:"
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False,                       # 그리디 = 재현 가능
                pad_token_id=self.tok.eos_token_id,
            )
        gen = self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        gen = "Thought:" + gen
        if stop:
            for s in stop:
                gen = gen.split(s)[0]
        return gen.strip()
