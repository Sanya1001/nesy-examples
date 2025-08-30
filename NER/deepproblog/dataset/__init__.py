import itertools
import json
import random
from pathlib import Path
from typing import Callable, List, Iterable, Tuple

import torchvision
import torchvision.transforms as transforms
from problog.logic import Term, list2term, Constant
from torch.utils.data import Dataset as TorchDataset

from deepproblog.dataset import Dataset
from deepproblog.query import Query

import torch
from torch.utils.data import DataLoader

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
        return self.data[int(idx[0])]

    def __iter__(self):
        for idx in range(len(self.data)):
            yield self.data[idx]

datasets = {
    "train": JSONDataset("nesy-examples/NER/dataset/train.json"),
    "test": JSONDataset("nesy-examples/NER/dataset/test.json"),
}

class JSONDataOperator(Dataset, TorchDataset):
    def __init__(self, 
                 dataset_name: str,                      
                 transform=None):
        super(JSONDataOperator, self).__init__()
        
        self.raw = datasets[dataset_name].raw_data
        self.dataset = datasets[dataset_name]
        self.dataset_name = dataset_name
        self.transform = transform

    def __len__(self):
        return len(self.raw)
    
    def __getitem__(self, idx):
        return self.to_query(idx)

    def to_query(self, idx: int) -> Query:
        """
        Turn the idx-th example into a Problog Query:
        
          check(P1, P2, P3, L1, L2, L3, Z1, Z2, F)
        
        with subs mapping each Pi/Li to a term
          tensor(dataset_name, idx, arg_index)
        so that DeepProbLog's TensorSource will route it to our __getitem__.
        """
        subs = {}
        args = []

        print("Index:", idx)

        # first the 3 persons
        for i in range(3):
            # p = self.raw[idx][f"person{i+1}"][0]
            var = Term(f"P{i+1}")
            # arg_index = i  → tells DataLoader which of the 6 embeddings to use
            args.append(Term("tensor", Term(self.dataset_name, Constant(int(idx)*6+i))))
            # args.append(var)

        # then the 3 locations
        for i in range(3):
            # l = self.raw[idx][f"location{i+1}"][0]
            var = Term(f"L{i+1}")
            args.append(Term("tensor", Term(self.dataset_name, Constant(int(idx)*6+i+3))))
            # args.append(var)

        # final boolean/int label
        label_value1 = int(self.raw[idx]["condition_1"][0])
        label_value2 = int(self.raw[idx]["condition_2"][0])

        # subs[Term('Z1')] = Constant(label_value1)
        # subs[Term('Z2')] = Constant(label_value2)
        # args.append(Term("Z1"))
        # args.append(Term("Z2"))

        F = Constant(label_value1 * label_value2)

        return Query(
            Term("check", *args, F)
        )