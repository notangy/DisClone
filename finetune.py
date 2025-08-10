from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
import torch

# use this if resuming from checkpoint
# orig_torch_load = torch.load
# torch.load = lambda *args, **kwargs: orig_torch_load(*args, **{**kwargs, "weights_only": False})

special_tokens_dict = {'additional_special_tokens': ['Prompt:', 'Response:']}

class TokenizerCheckpointCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        print(f"Saving tokenizer to {checkpoint_dir}")
        self.tokenizer.save_pretrained(checkpoint_dir)
        trainer.save_model(f"{args.output_dir}/checkpoint-{state.global_step}")
        return control


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model with LM head
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
tokenizer.add_special_tokens(special_tokens_dict)


model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device) 
model.config.pad_token_id = model.config.eos_token_id  # Important for training
model.resize_token_embeddings(len(tokenizer))

# Load your dataset from JSONL with 'text' field
dataset = load_dataset("json", data_files="training.jsonl")["train"]




# Tokenize with labels for causal LM
def tokenize_function(examples):
    merged_texts = [
        f"Prompt: {p}\nResponse: {r}"
        for p, r in zip(examples["prompt"], examples["response"])
    ]

    # Tokenize merged texts once
    tokens = tokenizer(
        merged_texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None  # avoid PyTorch tensors to save memory here
    )

    labels = []
    for i, p in enumerate(examples["prompt"]):
        # Tokenize prompt only once per batch item
        prompt_ids = tokenizer(
            f"Prompt: {p}\nResponse:",
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None
        )["input_ids"]

        mask_len = min(len(prompt_ids), len(tokens["input_ids"][i]))
        # Create label list from input_ids
        label = tokens["input_ids"][i].copy()
        label[:mask_len] = [-100] * mask_len  # Mask prompt tokens
        labels.append(label)

    tokens["labels"] = labels
    return tokens

# Apply tokenization and drop original columns
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["prompt", "response"]
)

# Set training args
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-6,
    weight_decay=0.01,
    eval_strategy="no",
    push_to_hub=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    callbacks=[TokenizerCheckpointCallback(tokenizer)],
)

print(f"Dataset size: {len(tokenized_dataset)}")
# Start training
trainer.train()

tokenizer.save_pretrained('./results')
# Resume from last checkpoint
# trainer.train(resume_from_checkpoint="./results/checkpoint-1000")