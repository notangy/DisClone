**DISCLAIMER**
As this is a pet project for the purpose of learning and AI tools can rapidly change, I give no guarantee that the provided files will work out of the box. Feel free to play around with them however you wish!

# (Dis)clone yourself!

Wouldn't it be great if you could be online all the time, able to entertain your friends whenever they needed you?
Well, you can't. But you can do the next best thing; introduce a custom Discord bot that's trained on your own message history!

# How does it work?

Make sure you install all the necessary requirements in `requirements.txt` and have python installed on your machine (below version 3.12, pytorch is very picky about the version you have!)

Once that's done you'll need .html exports from your most active server channels and private messages (certain tools already exist for these purposes...). Once you have these, run `chatlog_cleaner.py` with your discord username:

e.g. python3 chatlog_cleaner.py --username notangy

This will produce a `training.jsonl` file, which will contain your own messages paired up to prompts from other users in the following format:

{
prompt: other users messages
response: your own messages
prompt: other users messages
response: your own messages....
}

This mimics conversational flow, assuming that each message not from your own username is a prompt.
Due to the way the training data is structured, this works best for data pulled from DMs or small servers.

# Fine-tuning the model

Once you have your .jsonl file, you're ready to train!

Use the `finetune_base_model.py` script, which will read your training data and produce a fine-tuned model from gpt
(feel free to replace the hardcoded model with any other that you have access to).

**Note**: Depending on the size of your training data, the fitting process may take a long time with high resource usage. For reference, my data contained over 15000 key/value pairs and took an hour to train with an CUDA enabled RTX 4080. I only recommend using CPU with torch if your data set is very small (<1000)

With your trained model ready, you can test it in the `./testing` folder with the flask app `app.py`. It has a simple text box you can put prompts into to see what your model responds with.

# Creating the DisClone

The last step will mostly be coming from https://discordpy.readthedocs.io/en/stable/ (assuming your bot code will be written in python).

There's many different use cases for a Discord bot, but for now we'll just have ours respond with generated text whenever someone in a server directly mentions it.
You'll need to have a bot set up already on the Discord developer portal, and a token for it. Place the token into a .env file as token={token_here}, and run `./bot/client.py` when your bot has been added to a server.

# Optional feature: LoRA training

If you want your bot to be continually learning from its and/or your own messages, set the following env vars:

```
USE_LORA=1
ALLOW_SELF_TRAINING= 1 or 0
DISCORD_ID={your Discord ID here}
```

(If ALLOW_SELF_TRAINING is true, the bots own output will also be used for further training)

This will enable **LoRA** (Low-rank adapation) training by continuing to gather your own messages wherever the bot is present.
By default, LoRA training takes place every six hours and automatically prunes older interactions to stop the file becoming massive.

# That's it!

Your bot will respond with text generated from the trained model whenever it detects a message mentioning it! You can expand the script to do more actions, this is just the bare minimum. Note that you will need to have the script running on a server or remote machine in order for the bot to stay live, but that's outside the scope of this project.

Overall, this took three days of trial-and-error, reinstalling venvs, and praying that my RTX didn't explode. It was fun!
