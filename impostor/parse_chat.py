import os
from typing import *

import yaml

import re

from special_tokens import photo, video, sticker, voice, call

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))

user_name = config["user-name"]

new_chat_regex = r"(\d{2}:\d{2})\t([^\t]*)\t(.*)"

date_regex = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{2}\/\d{2}\/\d{4}"


def preprocess_message(msg: str) -> str:
    """Tokenizes photos/videos/stickers/voice messages/calls"""
    if msg == "[Photo]":
        return photo
    if msg == "[Video]":
        return video
    if msg == "[Sticker]":
        return sticker
    if msg == "[Voice message]":
        return voice
    if re.match(r"^Missed|Cancelled call|Call time (\d{1,2}:)?\d{1,2}:\d{2}$", msg):
        return call
    return msg



def parse_chat_logs(file_path: str) -> List[List[Tuple[bool, str]]]:
    """Parses a chat log in the file specified, into lists of chats with annotations.

    Takes in a file_path, which is a file exported from LINE.
    Splits chats into conversations by date, and returns a list of conversations.
    Each conversation is a list of tuples (user, message) where user is a boolean corresponding to whether the sender is the
    user or not, and message is a string corresponding to the message.
    Also converts newlines into <lsep>.
    """
    with open(file_path, "r", encoding="ascii", errors="ignore") as f:
        chats = f.read()
    chats = chats.split("\n")

    # merge chats so that the same message stays together
    merged_chats = []
    current_msg = ""
    sender = ""
    current_convo = []
    for line in chats:
        if line == "":
            continue
        m = re.fullmatch(new_chat_regex, line)
        if re.fullmatch(date_regex, line):
            if len(current_convo) > 0:
                merged_chats.append(current_convo)
            current_convo = []
        elif m:
            if len(current_msg) > 0:
                current_convo.append((sender == user_name, preprocess_message(current_msg)))
            current_msg = m.groups()[2]
            sender = m.groups()[1]
        else:
            current_msg += " <lsep> " + line

    if len(current_convo) > 0:
        current_convo.append((sender == user_name, preprocess_message(current_msg)))
    if len(current_msg) > 0:
        merged_chats.append(current_convo)

    return merged_chats


if __name__ == "__main__":
    print(parse_chat_logs("../data/raw-history-test/test.txt"))
