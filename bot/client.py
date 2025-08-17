import discord
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

from bot.lora_training import (
    log_interaction,
    load_lora_adapter,
    set_up_scheduler,
)

load_dotenv()

BOT_TOKEN = os.getenv("discord_token")
MODEL_PATH = os.path.abspath("./trained-model")

LORA_ENABLED = bool(os.getenv("use_lora", 0))  # false by default
DISCORD_USERNAME = os.getenv("discord_username")  # for optional LoRA training


# 1. Tokenizer & Base Model
# ------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)

# 2. Response Generation
# ------------------------------


def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate output (tweak parameters for creativity or length)
    outputs = base_model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        top_p=0.95,
        top_k=90,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 3. Discord Bot
# ------------------------------


class DisCloneClient(discord.Client):
    async def on_ready(self):
        print(f"Logged on as {self.user}!")

    async def on_message(self, message):
        if message.author == self.user:
            return
        if self.user.mentioned_in(message):
            prompt_text = f"prompt: {message.content}\nresponse:"

            bot_reply = generate_response(prompt_text)

            if LORA_ENABLED:
                log_interaction(message.content, bot_reply)

            await message.channel.send(bot_reply)


# 4. Main Bot Loop
# ------------------------------


if __name__ == "__main__":

    if LORA_ENABLED:
        load_lora_adapter()
        set_up_scheduler()

    intents = discord.Intents.default()

    intents.message_content = True  # read message content
    intents.dm_messages = True  # enable DM events

    client = DisCloneClient(intents=intents)
    client.run(BOT_TOKEN)
