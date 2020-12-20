from typing import *

import torch
import numpy as np

from collections import defaultdict

import yaml

import os
import shutil

from impostor.parse_chat import parse_chat_logs

config = yaml.safe_load(open("config.yaml"))

INPUT_DIR = config["chat_log_directory"]
OUTPUT_FILE = config["dataset"]["file"]
NUM_CANDIDATES = config["dataset"]["num_candidates"]
MAX_HISTORY = config["dataset"]["max_history"]

if __name__ == "__main__":
    print(INPUT_DIR, OUTPUT_FILE, NUM_CANDIDATES)

    files_to_parse = os.listdir(INPUT_DIR)

    dataset = defaultdict(list)

    parsed_logs = []
    for file in files_to_parse:
        parsed = parse_chat_logs(os.path.join(INPUT_DIR, file))
        parsed_logs.append(parsed)

    utterances = set()
    for parsed in parsed_logs:
        for dialog in parsed:
            for utterance in dialog:
                utterances.add(utterance)
    utterances = list(utterances)
    # print(utterances)

    for parsed in parsed_logs:
        for dialog in parsed:
            for i, utterance in enumerate(dialog):
                history = dialog[max(0, i-MAX_HISTORY):i]
                # replies = utterance + np.random.choice(utterances, NUM_CANDIDATES - 1)
                candidates = [utterance]
                for _ in range(NUM_CANDIDATES - 1):
                    while (x := utterances[np.random.randint(0, len(utterances))]) in candidates:
                        pass
                    candidates.append(x)
                correct = np.random.randint(0, NUM_CANDIDATES)
                candidates[correct], candidates[0] = candidates[0], candidates[correct]
                dataset["history"].append(history)
                dataset["candidates"].append(candidates)
                dataset["correct"].append(correct)

    torch.save(dataset, OUTPUT_FILE)
