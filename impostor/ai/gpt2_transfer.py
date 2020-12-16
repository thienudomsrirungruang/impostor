from transformers import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer

from itertools import chain

model = OpenAIGPTDoubleHeadsModel.from_pretrained("openai-gpt")
tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

history = [["hello", "how", "are", "you", "?"],
           ["i", "am", "fine", "thanks", "."]]

reply = ["great", "to", "hear", "."]

bos, eos, speaker_self, speaker_other, lsep, pad = "<bos>", "<eos>", "<speaker_self>", "<speaker_other>", "<lsep>", "<pad>"

SPECIAL_TOKENS = [bos, eos, speaker_self, speaker_other, lsep, pad]

model.set_special_tokens(SPECIAL_TOKENS)
tokenizer.set_special_tokens(SPECIAL_TOKENS)


