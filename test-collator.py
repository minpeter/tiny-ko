from transformers import DataCollatorForLanguageModeling, AutoTokenizer
import pprint

tokenizer = AutoTokenizer.from_pretrained(
    "./artifacts/tknz"
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
examples = [
    {"input_ids": tokenizer("Hello, my name is peter")["input_ids"]},
    {"input_ids": tokenizer("Hi.")["input_ids"]}
]
result = collator(examples)

pprint.pprint({k: v.tolist() for k, v in result.items()})