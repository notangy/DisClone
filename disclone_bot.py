import discord
import os
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('token') # put your bot token in a .env file


checkpoint_path = os.path.abspath("./trained-model") 

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token 

model = GPT2LMHeadModel.from_pretrained(checkpoint_path, local_files_only=True)
model.eval()  # set model to evaluation mode


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class DisCloneClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):

        if message.author == self.user:
            return
        
        if client.user.mentioned_in(message):
            # The bot has been pinged; run our AI model to generate a response
            formatted_input = f"prompt: {message.content}\nresponse:"
            inputs = tokenizer.encode(formatted_input, return_tensors="pt")

            inputs = inputs.to(device)
        
            # Generate output (tweak parameters for creativity or length)
            outputs = model.generate(
                inputs, 
                max_new_tokens=60,
                do_sample=True, 
                top_p=0.95, 
                top_k=90,
                temperature=0.8,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
         )
            generated_sequence = tokenizer.decode(outputs[0])
            generated_text = generated_sequence.split("response:")[-1].strip() 

            await message.channel.send(generated_text)

intents = discord.Intents.default()
intents.message_content = True

client = DisCloneClient(intents=intents)
client.run(BOT_TOKEN)
