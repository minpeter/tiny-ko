import os
import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from transformers import DataCollatorWithFlattening
from transformers import TrainingArguments, Trainer, get_scheduler
from datasets import load_dataset
from muon_optimizer import create_muon_optimizer

max_seq_length = 4096
tokenizer = AutoTokenizer.from_pretrained(
    "./tknz/tiny-ko-tokenizer"
)

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
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model = LlamaForCausalLM(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(torch.bfloat16)
model.to(device)

train_dataset = load_dataset("Elriggs/openwebtext-100k", split="train")

def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples['text'])

    if tokenizer.eos_token_id is not None:
        for i in range(len(tokenized_inputs["input_ids"])):
            tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
            tokenized_inputs["attention_mask"][i].append(1)
            if "token_type_ids" in tokenized_inputs:
                tokenized_inputs["token_type_ids"][i].append(0)

    return tokenized_inputs

# 데이터셋 로드 및 매핑
train_dataset = load_dataset("Elriggs/openwebtext-100k", split="train")
tokenized_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=train_dataset.column_names
)

data_collator = DataCollatorWithFlattening()

train_args = TrainingArguments(
    output_dir="./outputs/test",
    bf16=True,
    # per_device_train_batch_size=8,
    auto_find_batch_size=True,
    torch_compile=True,
    logging_steps=50,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    dataloader_pin_memory=True,
    dataloader_num_workers=8,
    dataloader_prefetch_factor=8,
    dataloader_persistent_workers=True,
)


learning_rate = 1e-3
weight_decay = 0.1
optimizer = create_muon_optimizer(model, lr=learning_rate, wd=weight_decay)
num_training_steps = len(tokenized_dataset) // (train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps)
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)


trainer = Trainer(
    args=train_args,
    model=model,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    optimizers=(optimizer, lr_scheduler)
)
trainer.train()