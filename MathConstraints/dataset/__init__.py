import json

import torch
from typing import Mapping, Iterator, Callable, List, Iterable, Tuple
from problog.logic import Term, Constant
from deepproblog.dataset import Dataset
import itertools
from torch.utils.data import Dataset as TorchDataset
from deepproblog.query import Query


class MathConstraintsEMB(Mapping[Term, torch.Tensor]):
    """
    Getting the embedding tensor of each object
    """

    def __init__(self, json_filename: str):
        super(MathConstraintsEMB, self).__init__()

        with open(json_filename, 'r') as json_file:
            raw_data = json.load(json_file)

        self.data = []
        for idx in range(len(raw_data)):
            for emb in raw_data[idx]['obj_emb']:
                self.data.append(torch.Tensor(emb))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[int(idx[0])]

    def __iter__(self) -> Iterator:
        for emb in self.data:
            yield emb


class MathConstraintsDataset(Dataset, TorchDataset):

    def __init__(self, subset, data_file):
        self.subset = subset

        with open(data_file, 'r') as json_file:
            self.data = json.load(json_file)

    def to_query(self, i: int) -> Query:
        # Output to query from this one
        # Getting two object (this is pre-define)
        cur_data = self.data[i]
        obj1 = Term("tensor", Term(self.subset, Constant(i * 2)))
        obj2 = Term("tensor", Term(self.subset, Constant(i * 2 + 1)))
        label = Constant(int(cur_data["condition_label"][0]))

        property_object1_idx = cur_data["logic_order"][0][-1]
        relation_condition_idx = cur_data["logic_order"][1][-1]
        property_object2_cond_idx = cur_data["logic_order"][2][-1]

        inference_name = f"inference_{property_object1_idx}_{relation_condition_idx}_{property_object2_cond_idx}"

        # inference_name = "inference_1_1_1"

        query_term = Term(f"{inference_name}", obj1, obj2, label)
        return Query(query_term)

    def __len__(self):
        return len(self.data)
