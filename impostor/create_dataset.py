from typing import *

import torch
import numpy as np

from collections import defaultdict

import yaml

import os
import shutil

from parse_chat import parse_chat_logs

from transformers import OpenAIGPTTokenizer

from special_tokens import SPECIAL_TOKENS

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))


def create_dataset(input_dir: str, output_file: str, num_candidates: int, max_history: int):
    files_to_parse = os.listdir(input_dir)

    dataset = defaultdict(list)

    parsed_logs = []
    for file in files_to_parse:
        parsed = parse_chat_logs(os.path.join(input_dir, file))
        parsed_logs.append(parsed)

    # init tokenizer for checking
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    utterances_set = set()
    for parsed in parsed_logs:
        for dialog in parsed:
            for utterance in dialog:
                length = len(tokenizer.tokenize(utterance[1]))
                if length > config["dataset"]["max_message_length"]:
                    print("Skipping following message:\n{}".format(utterance[1] if len(utterance[1]) < 512 else utterance[1][:510] + "..."))
                else:
                    utterances_set.add(utterance)
    utterances = list(utterances_set)
    # print(utterances)

    for parsed in parsed_logs:
        for dialog in parsed:
            # remove invalid dialogs
            clean_dialog = []
            for utterance in dialog:
                if utterance in utterances_set:
                    clean_dialog.append(utterance)
            dialog = clean_dialog
            for i, utterance in enumerate(dialog):
                history = dialog[max(0, i - max_history):i]
                # replies = utterance + np.random.choice(utterances, NUM_CANDIDATES - 1)
                candidates = [utterance]
                for _ in range(num_candidates - 1):
                    x = utterances[np.random.randint(0, len(utterances))]
                    while x in candidates:
                        x = utterances[np.random.randint(0, len(utterances))]
                    candidates.append(x)
                correct = np.random.randint(0, num_candidates)
                candidates[correct], candidates[0] = candidates[0], candidates[correct]
                dataset["history"].append(history)
                dataset["candidates"].append(candidates)
                dataset["correct"].append(correct)

    torch.save(dataset, output_file)


if __name__ == "__main__":
    input_dir = config["chat_log_directory"]
    output_file = config["dataset"]["file"]
    max_candidates = config["dataset"]["num_candidates"]
    max_history = config["dataset"]["max_history"]
    print(input_dir, output_file, max_candidates)
    create_dataset(input_dir, output_file, max_candidates, max_history)
