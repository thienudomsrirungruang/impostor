from transformers import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer

from itertools import chain

model = OpenAIGPTDoubleHeadsModel.from_pretrained("openai-gpt")
tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

history = [["hello", "how", "are", "you", "?"],
           ["i", "am", "fine", "thanks", "."]]

reply = ["great", "to", "hear", "."]

bos, eos, self, other, lsep = "<bos>", "<eos>", "<self>", "<other>", "<lsep>"

