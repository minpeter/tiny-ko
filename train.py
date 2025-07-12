# RUN COMMAND: time uv run accelerate launch train.py

import os
import torch
import math
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk

# --- 설정 ---
# preprocess.py에서 저장한 데이터셋 경로
PROCESSED_DATA_PATH = "./processed_data"
TOKENIZER_PATH = "./tknz/tiny-ko-tokenizer"
CONTEXT_LENGTH = 2048
HF_MODEL_ID = "minpeter/tiny-ko-124m-base-muon"
LOCAL_MODEL_PATH = "model/tiny-ko-124m-base-muon"
# ----------------

# (추가) Muon 옵티마이저 관련 코드 시작 ##################################


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        # (수정) `params`를 직접 받도록 생성자 변경
        muon_params, adamw_params = params

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        all_params = list(muon_params) + list(adamw_params)
        super().__init__(all_params, defaults)

        for p in muon_params:
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon 파라미터 업데이트
            muon_params = [
                p for p in group["params"] if self.state[p].get("use_muon", False)
            ]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]

            for p in muon_params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                u = zeropower_via_newtonschulz5(g, steps=ns_steps)
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            # AdamW 파라미터 업데이트
            adamw_params = [
                p for p in group["params"] if not self.state[p].get("use_muon", False)
            ]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(p)
                    state["moment2"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g_updated = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # (수정) Trainer의 스케줄러와 호환되도록 lr 스케일링 방식 변경
                step_size = group["lr"] / bias_correction1

                p.data.add_(g_updated, alpha=-step_size * (bias_correction2**0.5))
                p.data.mul_(1 - group["lr"] * weight_decay)

        return loss


class MuonTrainer(Trainer):  # (추가) 사용자 정의 Trainer
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            # Muon과 AdamW에 사용할 파라미터를 분리합니다.
            # 2D 이상이고, 임베딩과 lm_head가 아닌 파라미터는 Muon으로 최적화합니다.
            muon_params = [
                p
                for n, p in self.model.named_parameters()
                if p.requires_grad
                and p.ndim >= 2
                and "embed_tokens" not in n
                and "lm_head" not in n
            ]
            adamw_params = [
                p
                for n, p in self.model.named_parameters()
                if p.requires_grad
                and not (p.ndim >= 2 and "embed_tokens" not in n and "lm_head" not in n)
            ]

            optimizer_grouped_parameters = (muon_params, adamw_params)

            self.optimizer = Muon(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                wd=self.args.weight_decay,
            )
        return self.optimizer


# (추가) Muon 옵티마이저 관련 코드 끝 ####################################


# 1. 전처리 완료된 데이터셋을 디스크에서 바로 로드
print(f"사전 처리된 데이터셋을 '{PROCESSED_DATA_PATH}'에서 로드합니다.")
tokenized_dataset = load_from_disk(PROCESSED_DATA_PATH)

print("\n로딩 완료된 데이터셋 구조:")
print(tokenized_dataset)
print(f"훈련 샘플 수: {len(tokenized_dataset['train'])}")
print(f"테스트 샘플 수: {len(tokenized_dataset['test'])}")
print(f"샘플 0의 토큰 수: {len(tokenized_dataset['train'][0]['input_ids'])}")

# 2. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
# tokenizer.model_max_length = CONTEXT_LENGTH

try:
    print(f"\n사용될 EOS 토큰: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
    print(f"사용될 PAD 토큰: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    print(f"사용될 BOS 토큰: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
except AttributeError as e:
    print(e)

# 3. 모델 구성 (Config)
config = LlamaConfig(
    hidden_size=480,
    num_hidden_layers=32,
    intermediate_size=1920,
    tie_word_embeddings=True,
    num_attention_heads=6,
    num_key_value_heads=2,
    vocab_size=len(tokenizer),
    max_position_embeddings=CONTEXT_LENGTH,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    rope_theta=10000.0,
    use_cache=False,
    attn_implementation="flash_attention_2",
)

# 4. 모델 초기화
config._attn_implementation = "flash_attention_2"
model = LlamaForCausalLM(config)
model = model.to(torch.bfloat16)

model_size = sum(t.numel() for t in model.parameters())
print(f"\n모델 크기: {model_size/1000**3:.2f}B parameters")

# 5. 데이터 콜레이터, 토크나이저 저장 및 푸시
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
tokenizer.save_pretrained(LOCAL_MODEL_PATH)
tokenizer.push_to_hub(HF_MODEL_ID)

# 6. 학습 인자 (TrainingArguments) 설정
max_cpu_count = int(os.cpu_count() / 3) or 1
args = TrainingArguments(
    output_dir=LOCAL_MODEL_PATH,
    push_to_hub=True,  # 필요시 주석 해제
    hub_model_id=HF_MODEL_ID,
    hub_strategy="every_save",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1_000,
    save_steps=1_000,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=39,
    per_device_eval_batch_size=39,
    logging_steps=25,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",  # warmup_stable_decay
    learning_rate=6e-4,
    # optim="adamw_torch_fused", # (수정) MuonTrainer가 옵티마이저를 생성하므로 이 인자는 제거합니다.
    dataloader_pin_memory=True,
    bf16=True,
    torch_compile=True,
    dataloader_num_workers=max_cpu_count,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# 7. 트레이너(Trainer) 초기화 및 학습 시작
trainer = MuonTrainer(  # (수정) Trainer 대신 MuonTrainer를 사용합니다.
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()
