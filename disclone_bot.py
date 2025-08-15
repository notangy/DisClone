import discord
import os
import torch
import json
import random
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from apscheduler.schedulers.background import BackgroundScheduler


load_dotenv()
BOT_TOKEN = os.getenv('discord_token')
MODEL_PATH = os.path.abspath("./trained-model")  # Location of trained base model in directory

INTERACTIONS_FILE = "user_interactions.jsonl" # Location of discord user interaction logs
ADAPTER_DIR = "./lora-adapter" # Location of LoRA training output

MAX_INTERACTIONS = 5000  # Total dataset size after pruning
RECENT_PERCENT = 0.8     # Fraction of dataset that will be recent
OLD_PERCENT = 0.2        # Fraction that will be old random samples to prevent catastrophic forgetting

TRAINING_INTERVAL = 6    # How often to train the LoRA adaptation in hours

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Tokenizer & Base Model
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
base_model.to(device)

model = base_model  # will be replaced with LoRA adapter when loaded

# 2. LoRA Config
# ------------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],  # may need changed depending on base model
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Interaction Logging
# ------------------------------
def log_interaction(prompt, response):
    with open(INTERACTIONS_FILE, "a", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "response": response}, f)
        f.write("\n")

# 4. Smart Pruning
# ------------------------------
def prune_interactions():
    if not os.path.exists(INTERACTIONS_FILE):
        return

    with open(INTERACTIONS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) > MAX_INTERACTIONS:
        recent_count = int(MAX_INTERACTIONS * RECENT_PERCENT)
        old_count = MAX_INTERACTIONS - recent_count
        recent_entries = lines[-recent_count:]
        old_entries_pool = lines[:-recent_count]
        old_entries = random.sample(old_entries_pool, min(old_count, len(old_entries_pool)))
        new_dataset = old_entries + recent_entries
        random.shuffle(new_dataset)
        with open(INTERACTIONS_FILE, "w", encoding="utf-8") as f:
            f.writelines(new_dataset)
        print(f"[Prune] Trimmed to {MAX_INTERACTIONS} entries.")

# 5. LoRA Training
# ------------------------------
def train_lora():
    prune_interactions()
    if not os.path.exists(INTERACTIONS_FILE):
        print("No interactions logged yet.")
        return

    with open(INTERACTIONS_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if not data:
        print("No training data available.")
        return

    combined = [f"Prompt: {d['prompt']}\nResponse: {d['response']}" for d in data]
    dataset = Dataset.from_dict({"text": combined})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    model_for_training = get_peft_model(base_model, peft_config)

    training_args = TrainingArguments(
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch",
        overwrite_output_dir=True
    )

    trainer = Trainer(
        model=model_for_training,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    print("[Trainer] Starting LoRA fine-tuning...")
    trainer.train()
    model_for_training.save_pretrained(ADAPTER_DIR)
    print("[Trainer] LoRA adapter saved.")
    load_lora_adapter()


# 6. Load LoRA Adapter
# ------------------------------
def load_lora_adapter(adapter_path=ADAPTER_DIR):
    global model
    if not os.path.exists(adapter_path):
        print("No adapter found. Using base model only.")
        model = base_model
    else:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)
        print("[Model] LoRA adapter loaded.")


# 7. Text Generation
# ------------------------------
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        top_p=0.95,
        top_k=90,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 8. Discord Bot
# ------------------------------
class DisCloneClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        if message.author == self.user:
            return
        if self.user.mentioned_in(message):
            prompt_text = f"prompt: {message.content}\nresponse:"
            bot_reply = generate_response(prompt_text).split("response:")[-1].strip()
            log_interaction(message.content, bot_reply)
            await message.channel.send(bot_reply)


# 9. LoRA Scheduler
# ------------------------------

scheduler = BackgroundScheduler()
scheduler.add_job(train_lora, "interval", hours=TRAINING_INTERVAL)
scheduler.start()


# 10. Main Bot Loop
# ------------------------------
if __name__ == "__main__":
    load_lora_adapter()
    intents = discord.Intents.default()
    intents.message_content = True
    intents.dm_messages = True
    client = DisCloneClient(intents=intents)
    client.run(BOT_TOKEN)
