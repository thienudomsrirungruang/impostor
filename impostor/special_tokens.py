bos, eos, speaker_self, speaker_other, lsep, pad = "<bos>", "<eos>", "<speaker_self>", "<speaker_other>", "<lsep>", "<pad>"

SPECIAL_TOKENS = {"bos_token": bos, "eos_token": eos,
                  "additional_special_tokens": [speaker_self, speaker_other, lsep], "pad_token": pad}
