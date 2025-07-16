from transformers import AutoTokenizer
from transformers import DataCollatorWithFlattening, DataCollatorForLanguageModeling
from datasets import Dataset


max_seq_length = 4096
tokenizer = AutoTokenizer.from_pretrained(
    "./tknz/tiny-ko-tokenizer"
)

train_dataset = Dataset.from_dict({
    'text': ["This is a test sentence.", "Here is another one.", "And yet another test sentence."]
})

def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples['text'])

    if tokenizer.eos_token_id is not None:
        for i in range(len(tokenized_inputs["input_ids"])):
            tokenized_inputs["input_ids"][i].append(tokenizer.eos_token_id)
            tokenized_inputs["attention_mask"][i].append(1)
            if "token_type_ids" in tokenized_inputs:
                tokenized_inputs["token_type_ids"][i].append(0)

    return tokenized_inputs

tokenized_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

collated_batch = data_collator(tokenized_dataset)

print(collated_batch)  # Fallback to printing the entire batch if there's an error