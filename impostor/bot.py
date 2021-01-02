from typing import *

import discord
from discord.ext import tasks

import numpy as np
import torch

import yaml
import os

import logging

import re

import random


from generate import generate_from_history, load_model_and_tokenizer, chance_reply

from special_tokens import photo, call, video, voice, sticker

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))

logger = logging.getLogger("discord")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename="log/bot-log.txt", encoding="utf-8", mode="w")
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)


likely_command_regex = r"^[\;\?\!\^\~\>\.\,\-\$\=][a-zA-Z].*|[a-zA-Z\;\?\!\^\~\>\.\,\-\$\=][\;\?\!\^\~\>\.\,\-\$\=][a-zA-Z].*$"

status_messages = ["hi!", "hello!", "hey", "how are you?"]


class Bot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model, self.tokenizer = load_model_and_tokenizer(config["eval"]["model_path"])
        # TODO: customisable prefix
        self.prefix = ","
        # eagerness is how often the bot will reply to messages
        self.eagerness = 2.3
        self.interactivity = 0.4
        self.since_last_reply = 0

    async def on_ready(self):
        print("Ready, logged in as {}".format(self.user))
        self.update_status.start()

    @tasks.loop(minutes=config["bot"]["update_status_interval_mins"])
    async def update_status(self):
        print("update")
        await self.change_presence(activity=discord.Game(name=",help | {}".format(random.choice(status_messages))))

    async def on_message(self, message: discord.Message):
        print("Received message from {}: {}".format(message.author.name, message.content))
        channel = message.channel
        force_reply = False
        if message.content.startswith(self.prefix):
            command = message.content[len(self.prefix):]
            if command == "forcereply":
                force_reply = True
            elif command == "help":
                embed = discord.Embed(title="Help",
                                      description="{0}forcereply: Forces the bot to reply.".format(self.prefix))
                await channel.send(embed=embed)
                return
            else:
                await channel.send("^^Command not recognized.")
                return
        else:
            if message.author.id == self.user.id or message.author.bot or re.match(likely_command_regex, message.content):
                print("Skipping")
                return
        history = await channel.history(limit=config["bot"]["history_limit"]).flatten()
        history.reverse()
        # filter bots except itself, and likely commands
        history = filter(lambda x: (not x.author.bot or x.author.id == self.user.id) and
                        (not re.match(likely_command_regex, x.content)), history)
        history = list(map(lambda x: (x.author.id == self.user.id, x.content), history))
        reply_chance = chance_reply(history, self.tokenizer, self.model, torch.device(config["bot"]["device"]))
        probability = 1 - np.exp(-self.interactivity * self.since_last_reply) * ((1 - reply_chance) ** self.eagerness)
        print("Chance: {:.03f} Probability: {:.03f}".format(reply_chance, probability))
        print(history)
        if force_reply or np.random.binomial(1, probability):
            print("Replying")
            reply = generate_from_history(history, self.tokenizer, self.model, torch.device(config["bot"]["device"]),
                                          token_blacklist=[photo, call, video, voice, sticker])
            for x in reply:
                await channel.send(x)
            self.since_last_reply = 0
        else:
            print("Not replying")
            self.since_last_reply += 1


if __name__ == "__main__":
    intents = discord.Intents.default()
    client = Bot(intents=intents)
    client.run(config["bot"]["bot_token"])
