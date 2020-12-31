from typing import *

from generate import load_model_and_tokenizer, generate_from_history

import yaml
import os

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(config["eval"]["model_path"])
    history: List[Tuple[bool, str]] = []
    while True:
        prompt = input(">>> ")
        prompt.replace("\n", " <lsep> ")
        history.append((False, prompt))
        ai_output = generate_from_history(history, tokenizer, model)
        for msg in ai_output:
            print("Bot: {}".format(msg.replace("<lsep>", "\n     ")))
            history.append((True, msg))
