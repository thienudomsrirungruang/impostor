from typing import *

import discord

import yaml
import os

import logging

import re

from generate import generate_from_history, load_model_and_tokenizer

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))

logger = logging.getLogger("discord")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename="log/bot-log.txt", encoding="utf-8", mode="w")
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)


likely_command_regex = r"^[\;\?\!\^\~\>][a-zA-Z].*|[a-zA-Z\;\?\!\^\~\>][\;\?\!\^\~\>][a-zA-Z].*$"


class Bot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model, self.tokenizer = load_model_and_tokenizer(config["eval"]["model_path"])

    async def on_ready(self):
        print("Ready, logged in as {}".format(self.user))

    async def on_message(self, message: discord.Message):
        print("Received message: {}".format(message.content))
        channel = message.channel
        if message.author.id == self.user.id or message.author.bot or re.match(likely_command_regex, message.content):
            print("Skipping")
            return
        history = await channel.history(limit=config["bot"]["history_limit"]).flatten()
        history.reverse()
        # filter bots except itself, and likely commands
        history = filter(lambda x: (not x.author.bot or x.author.id == self.user.id) and
                        (not re.match(likely_command_regex, x.content)), history)
        history = list(map(lambda x: (x.author.id == self.user.id, x.content), history))
        print(history)
        reply = generate_from_history(history, self.tokenizer, self.model)
        for x in reply:
            await channel.send(x)


if __name__ == "__main__":
    intents = discord.Intents.default()
    client = Bot(intents=intents)
    client.run(config["bot"]["bot_token"])
