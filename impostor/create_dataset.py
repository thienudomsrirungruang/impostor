from typing import *

import torch
import numpy as np

from collections import defaultdict

import yaml

import os
import shutil

from impostor.parse_chat import parse_chat_logs

config = yaml.safe_load(open("config.yaml"))


def create_dataset(input_dir: str, output_file: str, num_candidates: int, max_history: int):
    files_to_parse = os.listdir(input_dir)

    dataset = defaultdict(list)

    parsed_logs = []
    for file in files_to_parse:
        parsed = parse_chat_logs(os.path.join(input_dir, file))
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
                history = dialog[max(0, i - max_history):i]
                # replies = utterance + np.random.choice(utterances, NUM_CANDIDATES - 1)
                candidates = [utterance]
                for _ in range(num_candidates - 1):
                    while (x := utterances[np.random.randint(0, len(utterances))]) in candidates:
                        pass
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
