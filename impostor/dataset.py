from typing import *

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import OpenAIGPTTokenizer

from itertools import chain
from collections import defaultdict

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device: {}".format(device))

bos, eos, speaker_self, speaker_other = "<bos>", "<eos>", "<speaker_self>", "<speaker_other>"


def build_inputs(history: List[Tuple[bool, List[str]]], reply: Tuple[bool, List[str]],
                 tokenizer: transformers.OpenAIGPTTokenizer, populate_lm_labels=False, with_eos=True):
    history = history + [reply]
    sequence = list(map(lambda x: [speaker_self if x[0] else speaker_other] + x[1], history))
    # print(sequence)
    sequence[0] = [bos] + sequence[0]
    sequence[-1] = sequence[-1] + [eos]
    words = list(chain(*sequence))
    segments = list(chain(*[[speaker_self if s[0] else speaker_other] * len(sequence[i]) for i, s in enumerate(history)]))
    input_ids = tokenizer.convert_tokens_to_ids(words)
    mc_token_ids = len(input_ids) - 1
    token_type_ids = tokenizer.convert_tokens_to_ids(segments)
    lm_labels = [-100] * len(input_ids)
    if populate_lm_labels:
        lm_labels = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return input_ids, mc_token_ids, token_type_ids, lm_labels


class ChatDataset(Dataset):
    def __init__(self, dataset_object: DefaultDict, tokenizer: transformers.OpenAIGPTTokenizer):
        self._dataset_object = dataset_object
        self._length = len(dataset_object["history"])
        self._tokenizer = tokenizer

    def __getitem__(self, idx: int) -> DefaultDict:
        out = defaultdict(list)
        history = list(map(lambda x: (x[0], tokenizer.tokenize(x[1])), self._dataset_object["history"][idx]))
        for candidate in self._dataset_object["candidates"][idx]:
            candidate = (candidate[0], tokenizer.tokenize(candidate[1]))
            input_ids, mc_token_ids, token_type_ids, lm_labels = build_inputs(history,
                                                                              candidate, tokenizer)
            out["input_ids"].append(input_ids)
            out["mc_token_ids"].append(mc_token_ids)
            out["token_type_ids"].append(token_type_ids)
            out["lm_labels"].append(lm_labels)
        out["correct"] = self._dataset_object["correct"][idx]
        return out

    def __len__(self):
        return self._length


def pad_list(x: List[int], padding: int, padding_length: int):
    return x + [padding] * (padding_length - len(x))


def make_batch(dialogs: List[DefaultDict], pad_token_id: int):
    out = {}
    max_length = max(max(len(y) for y in x["input_ids"]) for x in dialogs)
    for k, v in dialogs[0].items():
        if k == "correct":
            out[k] = torch.tensor([x["correct"] for x in dialogs], dtype=torch.long)
        elif k == "mc_token_ids":
            out[k] = torch.tensor([pad_list(x[k], pad_token_id, max_length) for x in dialogs], dtype=torch.long)
        else:
            out[k] = torch.tensor([[pad_list(y, -100 if k == "lm_labels" else pad_token_id, max_length) for y in x[k]] for x in dialogs],
                                  dtype=torch.long)
    return out


def get_data_loader(dataset_path: str, tokenizer: transformers.OpenAIGPTTokenizer,
                    batch_size: int = 4, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    dataset_object = torch.load(dataset_path)
    chat_dataset = ChatDataset(dataset_object, tokenizer)
    pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    loader = DataLoader(chat_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        collate_fn=lambda x: make_batch(x, pad_token_id))
    return loader


if __name__ == "__main__":
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

    bos, eos, speaker_self, speaker_other, lsep, pad = "<bos>", "<eos>", "<speaker_self>", "<speaker_other>", "<lsep>", "<pad>"

    SPECIAL_TOKENS = {"bos_token": bos, "eos_token": eos,
                      "additional_special_tokens": [speaker_self, speaker_other, lsep], "pad_token": pad}
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS)

    loader = get_data_loader("../dataset/testing-set.pt", tokenizer)
    for x in loader:
        print(x)
