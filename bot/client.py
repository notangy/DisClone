import discord
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("discord_token")
MODEL_PATH = os.path.abspath("./trained-model")

# For optional LoRA training.
LORA_ENABLED = bool(os.getenv("USE_LORA", 0))  # false by default
ALLOW_SELF_TRAINING = bool(os.getenv("ALLOW_SELF_TRAINING", 0))

DISCORD_ID = os.getenv("DISCORD_ID")  # Note this is NOT your username


# 1. Tokenizer & Base Model
# ------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)


# Import LoRA dependencies after model & tokenizer init
from lora_training import (
    log_interaction,
    load_lora_adapter,
    set_up_scheduler,
)


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
    generated_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_sequence.split("response:")[-1].strip()


# 3. Discord Bot
# ------------------------------


class DisCloneClient(discord.Client):
    async def on_ready(self):
        print(f"[Discord] Logged on as {self.user}!")

    async def on_message(self, message):
        if message.author == self.user:  # prevent feedback loop
            return

        # Remove ping from the message
        message_cleaned = message.content.sub(r"<@!?\d+>", "", s)

        # Continue to gather our own messages for further training
        if message.author.id == DISCORD_ID and LORA_ENABLED:

            # is this message a direct reply to someone?
            if message.reference and message.reference.resolved:

                replied_message = message.reference.resolved
                log_interaction(replied_message.content, message_cleaned)
                return
            else:
                # if not a reply, fetch the last few messages before ours for prompt context
                history = await message.channel.history(
                    limit=2, before=message
                ).flatten()

                # reverse so older ones come first
                history = list(reversed(history))

                for msg in history:
                    prompt += f"{msg.content} \n"

                log_interaction(prompt, message_cleaned)

        # if bot has been pinged
        if self.user.mentioned_in(message):
            prompt_text = f"prompt: {message_cleaned}\nresponse:"

            bot_reply = generate_response(prompt_text)

            if LORA_ENABLED and ALLOW_SELF_TRAINING:
                log_interaction(message.content, bot_reply)

            await message.channel.send(bot_reply)


# 4. Main Bot Loop
# ------------------------------


if __name__ == "__main__":

    if LORA_ENABLED:
        print("[LoRA] LoRA enabled. Attempting to load adapter...")
        load_lora_adapter()
        set_up_scheduler()

    intents = discord.Intents.default()

    intents.message_content = True  # read message content
    intents.dm_messages = True  # enable DM events

    client = DisCloneClient(intents=intents)
    client.run(BOT_TOKEN)
