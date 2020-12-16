from typing import *

import yaml

import re

config = yaml.safe_load(open("../config.yaml"))

user_name = config["user-name"]

new_chat_regex = r"(\d{2}:\d{2})\t([^\t]*)\t(.*)"

date_regex = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{2}\/\d{2}\/\d{4}"


"""Parses a chat log in the file specified, into lists of chats with annotations.

Takes in a file_path, which is a file exported from LINE.
Splits chats into conversations by date, and returns a list of conversations.
Each conversation is a list of tuples (user, message) where user is a boolean corresponding to whether the sender is the
user or not, and message is a string corresponding to the message.
Also converts newlines into <lsep>.
"""


def parse_chat_logs(file_path: str) -> List[List[Tuple[bool, str]]]:
    words = []
    with open(file_path, "r") as f:
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
        print(repr(line))
        if re.fullmatch(date_regex, line):
            print("date")
            if len(current_convo) > 0:
                merged_chats.append(current_convo)
            current_convo = []
        elif m := re.fullmatch(new_chat_regex, line):
            print("match")
            if len(current_msg) > 0:
                current_convo.append((sender == user_name, current_msg))
            current_msg = m.groups()[2]
            sender = m.groups()[1]
            print("current_msg = {} sender = {}".format(current_msg, sender))
        else:
            current_msg += " <lsep> " + line

    if len(current_convo) > 0:
        current_convo.append(current_msg)
    if len(current_msg) > 0:
        merged_chats.append(current_convo)

    return merged_chats


if __name__ == "__main__":
    print(parse_chat_logs("../data/test.txt"))
