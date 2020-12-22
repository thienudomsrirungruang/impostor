import os
from typing import *

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import random_split
from transformers import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer

import yaml

import datetime

from dataset import get_dataset, get_data_loader

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))


def train(dataset_path: str):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")  # gpu not enough memory :(

    model = OpenAIGPTDoubleHeadsModel.from_pretrained("openai-gpt")
    model.to(device)
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

    bos, eos, speaker_self, speaker_other, lsep, pad = "<bos>", "<eos>", "<speaker_self>", "<speaker_other>", "<lsep>", "<pad>"

    special_tokens = {"bos_token": bos, "eos_token": eos,
                      "additional_special_tokens": [speaker_self, speaker_other, lsep], "pad_token": pad}

    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

    # dataloader = get_data_loader(dataset_path, tokenizer, batch_size=4, shuffle=False, num_workers=1)
    full_dataset = get_dataset(dataset_path, tokenizer)
    assert len(full_dataset) > 0
    train_size = int(len(full_dataset) * config["train"]["train_dataset_proportion"] + 1)
    test_size = len(full_dataset) - train_size
    print("Full dataset has {} dialogs. Splitting into train: {} and test: {}"
          .format(len(full_dataset), train_size, test_size))
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], torch.Generator().manual_seed(42))
    print(len(train_dataset), len(test_dataset))

    train_loader = get_data_loader(train_dataset, tokenizer, config["train"]["batch_size"], True, 0)
    test_loader = get_data_loader(test_dataset, tokenizer, 1, False, 0)

    lr = config["train"]["learning_rate"]
    print("lr: {}".format(lr))
    optimizer = AdamW(model.parameters(), lr=lr)

    # init logging
    start_time = datetime.datetime.now()
    last_model_save = start_time
    log_file = open(os.path.join(os.path.dirname(__file__), "log/log-{}.txt").format(start_time.strftime("%y-%m-%d-%H-%M-%S")), "w+")

    epochs = config["train"]["num_epochs"]
    iteration = 0
    for epoch in range(epochs):
        print("Starting epoch {}/{}".format(epoch, epochs))
        for batch in train_loader:
            model.train()
            input_ids = batch["input_ids"].to(device)
            mc_token_ids = batch["mc_token_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            lm_labels = batch["lm_labels"].to(device)
            mc_labels = batch["correct"].to(device)

            model_output = model(input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
                                 mc_labels=mc_labels, labels=lm_labels)

            # print("input_ids: {}\ntoken_type_ids: {}\nmc_token_ids: {}\nlm_labels: {}\nmc_labels: {}"
            #       .format(input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels))

            # print(model_output.loss.item(), model_output.mc_loss.item())
            lm_loss = model_output.loss
            mc_loss = model_output.mc_loss

            loss = lm_loss * config["train"]["lm_coeff"] + mc_loss * config["train"]["mc_coeff"]

            # logging
            log_file.write("{},{},{},{},{}\n".format(iteration, epoch, loss, lm_loss, mc_loss))
            log_file.flush()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["train"]["max_norm"])
            optimizer.step()
            optimizer.zero_grad()

            # TODO: evaluation

            print("Time: {} Epoch: {}/{} Iteration: {}/{} Loss: {} ({} {})"
                  .format(datetime.datetime.now() - start_time,
                          epoch, epochs, iteration, epochs * (len(train_dataset) // config["train"]["batch_size"]),
                          loss.item(), lm_loss.item(), mc_loss.item()))

            if datetime.datetime.now() - last_model_save > datetime.timedelta(hours=1):
                print("Saving model...")
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "checkpoints/model-{}-iter{}.pt")
                           .format(start_time.strftime("%y-%m-%d-%H-%M-%S"), iteration))

            iteration += 1


if __name__ == "__main__":
    train(config["dataset"]["file"])
