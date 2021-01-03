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

from database import DatabaseAccessor

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))

logger = logging.getLogger("discord")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename="log/bot-log.txt", encoding="utf-8", mode="w")
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)


likely_command_regex = r"^[\;\?\!\^\~\>\.\,\-\$\=][a-zA-Z].*|[a-zA-Z\;\?\!\^\~\>\.\,\-\$\=][\;\?\!\^\~\>\.\,\-\$\=][a-zA-Z].*$"

status_messages = ["hi!", "hello!", "hey", "uwu", "bruh", "owo", "lmao", "ye", "yeet", "i like trains", "hmmmmmmmm",
                   "well...", "how", "what"]


class Bot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model, self.tokenizer = load_model_and_tokenizer(config["eval"]["model_path"])
        self.database_accessor = DatabaseAccessor()

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
        # check for prefix/eagerness/interactivity/since_last_reply
        if channel.type == discord.ChannelType.text:
            # create guild if doesn't exist
            if not self.database_accessor.guild_exists(channel.guild.id):
                print("no guild, creating")
                self.database_accessor.add_guild(channel.guild.id, channel.guild.name)
            # create chat if doesn't exist
            if not self.database_accessor.chat_exists(channel.id):
                print("no chat, creating")
                self.database_accessor.add_chat(channel.id, True, "text", channel.guild.id, channel.name)
            guild_obj = self.database_accessor.get_guild_by_id(channel.guild.id)
            chat_obj = self.database_accessor.get_chat_by_id(channel.id)
            if chat_obj.override_chat_settings:
                eagerness, interactivity = chat_obj.eagerness, chat_obj.interactivity
            else:
                eagerness, interactivity = guild_obj.eagerness, guild_obj.interactivity
            prefix = guild_obj.prefix
        elif channel.type in (discord.ChannelType.private, discord.ChannelType.group):
            # create chat if doesn't exist
            if not self.database_accessor.chat_exists(channel.id):
                self.database_accessor.add_chat(channel.id, False,
                                                "private" if channel.type == discord.ChannelType.private else "group",
                                                None,
                                                channel.recipient.name if channel.type == discord.ChannelType.private
                                                else channel.name)
            chat_obj = self.database_accessor.get_chat_by_id(channel.id)
            eagerness, interactivity = chat_obj.eagerness, chat_obj.interactivity
            prefix = chat_obj.prefix
        else:
            raise NotImplementedError("Channel type not implemented: {}".format(str(channel.type)))
        since_last_reply = chat_obj.since_last_reply
        print("prefix: {} eagerness: {} interactivity: {} since_last_reply: {}"
              .format(prefix, eagerness, interactivity, since_last_reply))
        force_reply = False
        if message.content.startswith(prefix):
            command = message.content[len(prefix):]
            if command == "forcereply":
                force_reply = True
            elif command == "help":
                embed = discord.Embed(title="Help",
                                      description="{0}forcereply: Forces the bot to reply.".format(prefix))
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
        probability = 1 - np.exp(-interactivity * since_last_reply) * ((1 - reply_chance) ** eagerness)
        print("Chance: {:.03f} Probability: {:.03f}".format(reply_chance, probability))
        if force_reply or np.random.binomial(1, probability):
            async with channel.typing():
                print("Replying")
                reply = generate_from_history(history, self.tokenizer, self.model, torch.device(config["bot"]["device"]),
                                              token_blacklist=[photo, call, video, voice, sticker])
                for x in reply:
                    await channel.send(x)
                self.database_accessor.set_since_last_reply(channel.id, 0)
        else:
            print("Not replying")
            self.database_accessor.increment_last_reply(channel.id)


if __name__ == "__main__":
    intents = discord.Intents.default()
    client = Bot(intents=intents)
    client.run(config["bot"]["bot_token"])
