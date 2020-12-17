from typing import *

from transformers import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer

from itertools import chain

from impostor.data_collection.parse_chat import parse_chat_logs

model = OpenAIGPTDoubleHeadsModel.from_pretrained("openai-gpt")
tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

bos, eos, speaker_self, speaker_other, lsep, pad = "<bos>", "<eos>", "<speaker_self>", "<speaker_other>", "<lsep>", "<pad>"

# history = [[(True, "hello"), (True, "how"), (True, "are"), (True, "you"), (True, "?")],
#            [(False, "i"), (False, "am"), (False, "fine"), (False, "thanks"), (False, ".")]]

history = [(True, ["hello", "how", "are", "you", "?"]),
           (False, ["i", "am", "fine", "thanks", "."])]

reply = (True, ["good", "to", "hear", "."])

SPECIAL_TOKENS = {"bos_token": bos, "eos_token": eos, "additional_special_tokens": [speaker_self, speaker_other, lsep], "pad_token": pad}

orig_num_tokens = len(tokenizer.encoder)
print(orig_num_tokens)
num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS)
model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def build_inputs(history: List[Tuple[bool, List[str]]], reply: Tuple[bool, List[str]]):
    sequence = list(map(lambda x: [speaker_self if x[0] else speaker_other] + x[1], history))
    # print(sequence)
    sequence[0] = [bos] + sequence[0]
    sequence[-1] = sequence[-1] + [eos]
    # print(sequence)
    words = list(chain(*sequence))
    segments = list(chain(*[[speaker_self if s[0] else speaker_other] * len(sequence[i]) for i, s in enumerate(history)]))
    position = list(range(len(words)))
    return words, segments, position, sequence


words, segments, position, sequence = build_inputs(history, reply)

print(words, segments, position, sequence, sep="\n")

words = tokenizer.convert_tokens_to_ids(words)
segments = tokenizer.convert_tokens_to_ids(segments)

print(words, segments, sep="\n")
