# # train llama-like dense model with muon
# uv run moonlight-muon-train.py --model llama --optimizer muon --dataset openwebtext-100k --hidden_size 896 --lr 1e-3 --max_seq_length 4096

# # train qwen-like dense model with adamw
# uv run moonlight-muon-train.py --model qwen --optimizer adamw --dataset openwebtext-100k --hidden_size 896 --lr 1e-3

import os
import torch
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from muon_optimizer import Muon


class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        # CHANGED: Use max_length in the cache filename
        cache_file = f"{self.dataset_name}_{self.max_length}.bin"
        if os.path.exists(cache_file):
            logger.info(f"Loading tokenized data from {cache_file}")
            self.tokens = torch.load(cache_file)
        else:
            logger.info(f"Tokenizing texts and creating cache at {cache_file}")
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, cache_file)

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data


def get_model_and_dataloader(model_name, dataset_name, hidden_size, max_seq_length):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    train_dataset = load_dataset(name2path[dataset_name])
    tokenizer = AutoTokenizer.from_pretrained(
        "./tknz/tiny-ko-tokenizer"
    )
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer, max_length=max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    if model_name == "qwen":
        config = Qwen2Config(
            attn_implementation="flash_attention_2",
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=max_seq_length,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=16,
            num_hidden_layers=12,
            num_key_value_heads=16,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=len(tokenizer),
        )
        model = Qwen2ForCausalLM(config)
    elif model_name == "llama":
        config = LlamaConfig(
            hidden_size=480,
            num_hidden_layers=32,
            intermediate_size=1920,
            tie_word_embeddings=True,
            num_attention_heads=6,
            num_key_value_heads=2,
            vocab_size=len(tokenizer),
            max_position_embeddings=max_seq_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            rope_theta=10000.0,
            use_cache=False,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
        )
        model = LlamaForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"
    return model, train_loader, tokenizer


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the trained model")
    parser.add_argument("--max_seq_length", type=int, default=512, help="The maximum sequence length for tokenization and model.")
    args = parser.parse_args()
    
    log_file_name = f"train_{args.model}_{args.optimizer}_lr{args.lr}.log"
    logger.add(os.path.join("logs", log_file_name))

    output_path = os.path.join(args.output_dir, f"{args.model}_{args.optimizer}_lr{args.lr}")
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Model will be saved to: {output_path}")

    model, train_loader, tokenizer = get_model_and_dataloader(
        args.model, args.dataset, args.hidden_size, args.max_seq_length
    )
    optimizer = get_optimizer(
        args.optimizer, model, lr=args.lr, wd=args.wd
    )

    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(torch.bfloat16)
    model.to(device)
    model = torch.compile(model)

    model.train()
    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )
    for epoch in range(epoch):
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            batch = batch.to(device)
            input_ids = batch
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if step % 50 == 0:
                logger.info(
                    f"Epoch: {epoch} Step: {step} LR: {optimizer.param_groups[0]['lr']:.6f} Training loss: {loss.item():.4f}"
                )
    logger.info("Training finished. Saving model and tokenizer...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info(f"Model and tokenizer saved successfully to {output_path}")