# Run this after training.jsonl has been obtained
# this will create a true base model from bulk training data

import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # change this as needed

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None,
    )


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        lines = file.readlines()

    data = [json.loads(line) for line in lines]

    combined = [
        f"Prompt: {item['prompt']}\nResponse: {item['response']}" for item in data
    ]
    return Dataset.from_dict({"text": combined})


def tokenize_and_mask(examples):
    tokenized = tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length"
    )
    labels = []
    for i, text in enumerate(examples["text"]):
        prompt = text.split("Response:")[0] + "Response:"
        prompt_ids = tokenizer(
            prompt, truncation=True, max_length=512, padding="max_length"
        )["input_ids"]
        label_ids = tokenized["input_ids"][i].copy()
        label_ids[: len(prompt_ids)] = [-100] * len(prompt_ids)  # mask prompt tokens
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized


dataset = load_dataset("training.jsonl")

# Convert dataset to format the base model can parse
tokenized_dataset = dataset.map(
    tokenize_and_mask, batched=True, remove_columns=["text"]
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Tweak these args to your own use case
# e.g. depending your GPU, might need to lower train batch size
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=100,
    save_total_limit=2,
    logging_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained("./trained-model")
tokenizer.save_pretrained("./trained-model")
