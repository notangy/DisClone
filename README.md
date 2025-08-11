# 🚨 THIS IS AN EXPERIMENTAL WORK IN PROGRESS 🚨
I don't mind people looking at or borrowing my work but I give no guarantee that the provided files will work out of the box. This is purely for my own learning until I verify that the training outputs work as intended.

# (Dis)clone yourself!

Wouldn't it be great if you could be online all the time, able to entertain your friends whenever they needed you?
Well, you can't. But you can do the next best thing; introduce a custom Discord bot that's trained on your own message history!

# How does it work?

Make sure you install all the necessary requirements in `requirements.txt` and have python installed on your machine (below version 3.12, pytorch is very picky about the version you have!)
Once that's sorted you'll need .html exports from your most active server channels and private messages (certain tools already exist for these purposes...). Once you have these, run `chatlog_cleaner.py` with your discord username:

e.g. python3 chatlog_cleaner.py --username notangy

This will produce a `training.jsonl` file, which will contain your own messages paired up to prompts from other users. Due to the way the training data is structured, this works best for data pulled from DMs or small servers.

Now here comes the interesting part... 

TODO put finetune.py instructions once script is trained and working
