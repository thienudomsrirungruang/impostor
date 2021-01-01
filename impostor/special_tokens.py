bos, eos, speaker_self, speaker_other, lsep, pad = "<bos>", "<eos>", "<speaker_self>", "<speaker_other>", "<lsep>", "<pad>"

photo, call, video, voice, sticker = "<photo>", "<call>", "<video>", "<voice>", "<sticker>"

SPECIAL_TOKENS = {"bos_token": bos, "eos_token": eos,
                  "additional_special_tokens": [speaker_self, speaker_other, lsep, photo, call, video, voice, sticker],
                  "pad_token": pad}
