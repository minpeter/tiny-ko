import os
import evaluate
import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

ds = load_dataset('minpeter/pretrain-korean-dedup', split='train')
ds = ds.train_test_split(test_size=0.001, shuffle=True, seed=5768112)
print(ds)

# tokenize dataset
context_length = 512
tokenizer = AutoTokenizer.from_pretrained("kakaocorp/kanana-nano-2.1b-base")

def tokenize(element):
    """
    A text which length is over `context_length` is divided into multiple segments
    """
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    return outputs

tokenized_dataset = ds.map(
    tokenize,
    remove_columns=ds["train"].column_names,
    batched=True,
    batch_size=5_000,  # adjust batch size based on your memory capacity
    num_proc=64,      # depending on your CPU cores, you can adjust this number
)
print(tokenized_dataset)

# initialize model
config = AutoConfig.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    vocab_size=len(tokenizer),
    max_position_embeddings=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = LlamaForCausalLM(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"Model size: {model_size/1000**3:.1f}B parameters")



# prepare evaluation metric
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]

    return torch.argmax(logits, axis=-1)

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)

    return metric.compute(predictions=preds, references=labels)


tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="model/pretrained-tiny-ko",
    push_to_hub=True,
    hub_model_id="minpeter/pretrained-tiny-ko",
    hub_strategy="every_save",

    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1_000,
    save_steps=1_000,

    gradient_accumulation_steps=4,
    per_device_train_batch_size=56,
    per_device_eval_batch_size=56,

    logging_steps=10,
    num_train_epochs=2,
    weight_decay=0.1,
    warmup_steps=10_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    # fp16=True,
    bf16=True,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()