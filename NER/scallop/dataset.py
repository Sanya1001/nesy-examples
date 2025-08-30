import itertools
import json
import random
from pathlib import Path
from typing import Callable, List, Iterable, Tuple
import torch
from torch.utils.data import Dataset as TorchDataset

class JSONDataset(TorchDataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.raw_data = json.load(f)

        self.data = []

        for item in self.raw_data:
            self.data.append(self._vec(item, 'person1'))
            self.data.append(self._vec(item, 'person2'))
            self.data.append(self._vec(item, 'person3'))
            self.data.append(self._vec(item, 'location1'))
            self.data.append(self._vec(item, 'location2'))
            self.data.append(self._vec(item, 'location3'))

        self.dataset = "train" if "train" in json_path else "test"
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def _vec(self, item, key):
        return torch.tensor(item[key][0], dtype=torch.float32)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for idx in range(len(self.data)):
            yield self.data[idx]