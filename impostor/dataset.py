from typing import *

import torch

from torch.utils.data import Dataset


class ChatDataset(Dataset):
    def __init__(self, dataset_object: DefaultDict):
        self._dataset_object = dataset_object
        self._length = len(dataset_object["history"])

    def __getitem__(self, idx: int):
        output = {}
        for k, v in self._dataset_object.items():
            output[k] = v[idx]
        return output

    def __len__(self):
        return self._length


def get_data_loader(dataset_path: str):
    dataset_object = torch.load(dataset_path)
    chat_dataset = ChatDataset(dataset_object)


if __name__ == "__main__":
    get_data_loader("../dataset/testing-set.pt")
