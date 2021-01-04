import string
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

conversation_presets = {"never": (0.0, 0.0), "rarely": (0.03, 0.0), "sometimes": (0.1, 0.0), "often": (1.0, 0.1),
                        "conversational": (3.0, 1.4), "always": (1000.0, 1000.0)}


help_text = """`{0}help`: Shows this message.
`{0}forcereply`: Forces the bot to reply.
`{0}forget`: Makes the chatbot forget everything and start fresh.
`{0}options prefix`: Changes the prefix.
`{0}options mode`: Sets how often the bot should reply.
> `{0}options mode get` - Gets the current mode.
> `{0}options mode default` (servers only) - uses the server default (see `options smode`).
> `{0}options mode [rarely|sometimes|often|conversational|always]` - presets (default `often` for servers, `help` for dms)
> `{0}options mode <eagerness> <interactivity>`:
> - `eagerness` (0 to 1000) modifies the base value of probability.
> - `interactivity` (0 to 1000) modifies the increased chance of replying after a message is ignored.
> Preset values:
> - `never`: 0.0 / 0.0
> - `rarely`: 0.03 / 0.0
> - `sometimes`: 0.1 / 0.0
> - `often`: 1.0 / 0.1
> - `conversational`: 3.0 / 1.4
> - `always`: 1000.0 / 1000.0
`{0}options smode`: Same as `mode` but sets server default values. Only works on servers."""


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
        # check for prefix/eagerness/interactivity/since_last_reply/last_forget
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
        last_forget = chat_obj.last_forget
        print("prefix: {} eagerness: {} interactivity: {} since_last_reply: {} last_forget: {}"
              .format(prefix, eagerness, interactivity, since_last_reply, last_forget))
        # Commands
        force_reply = False
        if message.content.startswith(prefix):
            command = message.content[len(prefix):].strip()
            split_command = command.split(" ")
            keyword = split_command[0]
            clean_prefix = prefix + " " if prefix[-1] in string.ascii_letters else prefix
            if keyword == "forcereply":
                force_reply = True
            elif keyword == "help":
                embed = discord.Embed(title="Help",
                                      description=help_text.format(clean_prefix))
                await channel.send(embed=embed)
                return
            elif keyword == "forget":
                await channel.send("^^I have forgotten everything before this message!")
                self.database_accessor.reset_last_forget_chat(channel.id)
                return
            elif keyword == "options":
                if len(split_command) <= 1:
                    await channel.send("^^Command not recognized. Please type `{0}help` for more info.".format(clean_prefix))
                if split_command[1] == "prefix":
                    new_prefix = " ".join(split_command[2:])
                    if len(new_prefix) > 8:
                        await channel.send("^^Prefix is too long. Max length is 8 characters.")
                    elif len(new_prefix) == 0:
                        await channel.send("^^Prefix cannot be empty!")
                    else:
                        if channel.type == discord.ChannelType.text:
                            self.database_accessor.set_guild_prefix(channel.guild.id, new_prefix)
                        else:
                            self.database_accessor.set_chat_prefix(channel.id, new_prefix)
                        await channel.send("^^Success! Prefix changed to `{0}`".format(new_prefix))
                elif split_command[1] in ("mode", "smode"):
                    if split_command[1] == "smode" and channel.type != discord.ChannelType.text:
                        await channel.send("^^This command is only available on servers.")
                        return
                    if len(split_command) == 2:
                        await channel.send("^^Command options not recognized. Please type `{0}help` for more info.".format(clean_prefix))
                    elif len(split_command) == 3:
                        if split_command[2] == "default" and split_command[1] == "mode":
                            if channel.type == discord.ChannelType.text:
                                self.database_accessor.set_chat_override(channel.id, False)
                                await channel.send("^^Success! Options reset to server default.")
                            else:
                                await channel.send("^^This command is only available on servers.")
                        elif split_command[2] == "get":
                            if split_command[1] == "mode":
                                e, i = eagerness, interactivity
                            else:
                                e, i = guild_obj.eagerness, guild_obj.interactivity
                            for preset, values in conversation_presets.items():
                                if (e, i) == values:
                                    await channel.send("^^This {}'s mode is currently: {} (eagerness {}, interactivity {})"
                                                       .format("channel" if split_command[1] == "mode" else "server",
                                                               preset, e, i))
                                    break
                            else:
                                await channel.send("^^This {}'s mode is currently: eagerness {}, interactivity {}"
                                                   .format("channel" if split_command[1] == "mode" else "server",
                                                           e, i))
                        else:
                            for preset, values in conversation_presets.items():
                                if split_command[2] == preset:
                                    if split_command[1] == "mode":
                                        self.database_accessor.set_chat_eagerness_interactivity(channel.id, *values)
                                        if channel.type == discord.ChannelType.text:
                                            self.database_accessor.set_chat_override(channel.id, True)
                                        await channel.send("^^Success! Channel options set to {}.".format(preset))
                                    else:
                                        self.database_accessor.set_guild_eagerness_interactivity(channel.guild.id, *values)
                                        await channel.send("^^Success! Server options set to {}.".format(preset))
                                    break
                            else:
                                await channel.send("^^Command options not recognized. Please type `{0}help` for more info.".format(clean_prefix))
                    else:
                        try:
                            eagerness = float(split_command[2])
                            interactivity = float(split_command[3])
                            if not (0 <= eagerness <= 1000 and 0 <= interactivity <= 1000):
                                await channel.send("^^Eagerness and interactivity values must be between 0 and 1000 inclusive.")
                            else:
                                if split_command[1] == "mode":
                                    self.database_accessor.set_chat_eagerness_interactivity(channel.id, eagerness, interactivity)
                                    if channel.type == discord.ChannelType.text:
                                        self.database_accessor.set_chat_override(channel.id, True)
                                    await channel.send("^^Success! Channel eagerness set to {} and interactivity set to {}.".format(eagerness, interactivity))
                                else:
                                    self.database_accessor.set_guild_eagerness_interactivity(channel.guild.id, eagerness, interactivity)
                                    await channel.send("^^Success! Server eagerness set to {} and interactivity set to {}.".format(eagerness, interactivity))
                        except ValueError:
                            await channel.send("^^Command options not recognized. Please type `{0}help` for more info.".format(clean_prefix))
                else:
                    await channel.send("^^Command options not recognized. Please type `{0}help` for more info.".format(clean_prefix))
                return
            else:
                await channel.send("^^Command options not recognized. Please type `{0}help` for more info.".format(clean_prefix))
                return
        else:
            if message.author.id == self.user.id or message.author.bot or re.match(likely_command_regex, message.content):
                print("Skipping")
                return
        history = await channel.history(limit=config["bot"]["history_limit"]).flatten()
        history.reverse()
        # filter bots except itself, and likely commands
        history = filter(lambda x: (not x.author.bot or x.author.id == self.user.id) and
                         not re.match(likely_command_regex, x.content) and
                         (last_forget is None or x.created_at > last_forget),
                         history)
        history = list(map(lambda x: (x.author.id == self.user.id, x.content), history))
        print("History length: {}".format(len(history)))
        reply_chance = chance_reply(history, self.tokenizer, self.model, torch.device(config["bot"]["device"]))
        # probability = 1 - np.exp(-interactivity * since_last_reply) * ((1 - reply_chance) ** eagerness)
        probability = 1 - (1 - reply_chance) ** (eagerness + since_last_reply * interactivity)
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
