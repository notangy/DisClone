import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

def tokenize_function(examples):
  return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512, return_tensors=None)


def load_dataset(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        lines = file.readlines()
    
    data = [json.loads(line) for line in lines]

    combined = [f"Prompt: {item['prompt']}\nResponse: {item['response']}" for item in data]
    return Dataset.from_dict({'text': combined})


def tokenize_and_mask(examples):
    tokenized = tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')
    labels = []
    for i, text in enumerate(examples['text']):
        prompt = text.split('Response:')[0] + 'Response:'
        prompt_ids = tokenizer(prompt, truncation=True, max_length=512, padding='max_length')['input_ids']
        label_ids = tokenized['input_ids'][i].copy()
        label_ids[:len(prompt_ids)] = [-100] * len(prompt_ids)  # mask prompt tokens
        labels.append(label_ids)
    tokenized['labels'] = labels
    return tokenized


dataset = load_dataset('training.jsonl') 

tokenized_dataset = dataset.map(tokenize_and_mask, batched=True, remove_columns=['text'])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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
    train_dataset=tokenized_dataset
)

trainer.train()

model.save_pretrained("./trained-model")
tokenizer.save_pretrained("./trained-model")