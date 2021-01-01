from typing import *

from dataset import build_inputs

from transformers import OpenAIGPTTokenizer, OpenAIGPTDoubleHeadsModel
import torch
import torch.nn.functional as F

import yaml
import os

from special_tokens import bos, eos, speaker_self, speaker_other, lsep, pad, SPECIAL_TOKENS

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))


def load_model_and_tokenizer(file_path: str) -> Tuple[OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer]:
    model = OpenAIGPTDoubleHeadsModel.from_pretrained("openai-gpt")
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

    model.load_state_dict(torch.load(file_path))

    return model, tokenizer


def top_p_sample(logits: torch.Tensor, top_p=0.9) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probabilities > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Back to unsorted indices and set them to -infinity
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float("inf")

    return logits


def filter_logits(logits: torch.Tensor, tokenizer: OpenAIGPTTokenizer,
                  use_whitelist: bool, whitelist: Optional[List[str]] = None,
                  blacklist: Optional[List[str]] = None) -> torch.Tensor:
    # mask: 1 if -inf, 0 if keep
    if use_whitelist:
        whitelist = tokenizer.convert_tokens_to_ids(whitelist)
        mask = torch.ones(logits.size())
        mask[whitelist] = 0
    else:
        blacklist = tokenizer.convert_tokens_to_ids(blacklist)
        mask = torch.zeros(logits.size())
        mask[blacklist] = 1

    indices = mask == 1
    logits[indices] = float("-inf")
    return logits


def generate_from_history(history: List[Tuple[bool, str]], tokenizer: OpenAIGPTTokenizer,
                          model: OpenAIGPTDoubleHeadsModel) -> List[str]:
    """Generates an utterance given a set of messages preceding it.

    :argument history: a list of tuples (user, message)
                            user is a boolean on whether sender is user.
                            message is string.
    :argument tokenizer: the tokenizer
    :argument model: the model"""

    # build the network inputs
    output = []
    inputs = [bos]
    token_types = [speaker_other if len(history) > 0 and not history[0][0] else speaker_self]
    for user, text in history:
        inputs.append(speaker_self if user else speaker_other)
        token_types.append(speaker_self if user else speaker_other)
        for token in tokenizer.tokenize(text):
            inputs.append(token)
            token_types.append(speaker_self if user else speaker_other)
    inputs.append(speaker_self)
    token_types.append(speaker_self)

    input_ids = tokenizer.convert_tokens_to_ids(inputs)
    token_type_ids = tokenizer.convert_tokens_to_ids(token_types)

    model.eval()

    eos_token = tokenizer.convert_tokens_to_ids(eos)
    speaker_self_token = tokenizer.convert_tokens_to_ids(speaker_self)

    cutoff = 500
    for i in range(config["bot"]["token_limit"]):
        model_out = model(torch.tensor([input_ids], dtype=torch.long)[-cutoff:],
                          token_type_ids=torch.tensor([token_type_ids], dtype=torch.long)[-cutoff:])
        logits = model_out.logits[0, -1, :] / config["eval"]["temperature"]
        blacklist = [bos, speaker_other, pad]
        logits = filter_logits(logits, tokenizer, False, blacklist=blacklist)
        logits = top_p_sample(logits, config["eval"]["top_p"])
        # print("{} -> {}".format(tokenizer.convert_ids_to_tokens(output[-5:]), tokenizer.convert_ids_to_tokens(torch.topk(logits, 5)[1])))
        probs = F.softmax(logits, dim=-1)
        prev = torch.multinomial(probs, 1).item()
        input_ids.append(prev)
        token_type_ids.append(speaker_self_token)
        output.append(prev)
        if prev == eos_token:
            break

    output = tokenizer.convert_ids_to_tokens(output)
    current_msg = []
    messages = []
    for i in output:
        if i in (speaker_self, eos):
            messages.append(tokenizer.convert_tokens_to_string(current_msg))
            current_msg = []
        else:
            current_msg.append(i)
    if len(current_msg) > 0:
        messages.append(tokenizer.convert_tokens_to_string(current_msg))
    return messages


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(config["eval"]["model_path"])
    print(model)
    history = [(False, "hey"), (False, "how u doin?"), (True, "pretty good, u?"), (False, "feeling good after playing some games")]
    print(generate_from_history(history, tokenizer, model))
