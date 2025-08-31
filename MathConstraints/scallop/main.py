import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm
import json
import scallopy


class MathConstraintsDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ):
        # Contains a MNIST dataset
        data_path = os.path.join(root, "train.json" if train else "test.json")
        with open(data_path, 'rb') as f:
            self.data = json.load(f)

        self.train = True

    def __len__(self):
        return int(len(self.data))

    def __getitem__(self, idx):
        # Get two data points
        obj1 = self.data[idx]['obj_emb'][0]
        obj2 = self.data[idx]['obj_emb'][1]
        label = self.data[idx]['condition_label']
        logic_order = [int(self.data[idx]['logic_order'][0][-1]),
                       int(self.data[idx]['logic_order'][1][-1]),
                       int(self.data[idx]['logic_order'][2][-1])]

        # Each data has two images and the GT is the sum of two digits
        return (obj1, obj2, label, logic_order)

    @staticmethod
    def collate_fn(batch):
        obj1s = torch.stack([torch.tensor(item[0]) for item in batch])
        obj2s = torch.stack([torch.tensor(item[1]) for item in batch])
        labels = torch.stack([torch.tensor(item[2]).long() for item in batch])
        logic_orders = [item[3] for item in batch]
        return ((obj1s, obj2s), labels, logic_orders)


def MathConstraintsLoader(data_dir, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        MathConstraintsDataset(
            data_dir,
            train=True
        ),
        collate_fn=MathConstraintsDataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        MathConstraintsDataset(
            data_dir,
            train=False
        ),
        collate_fn=MathConstraintsDataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, test_loader


# Normal MNIST Sum
class PropertyNN(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer = torch.nn.Sequential(torch.nn.Linear(self.size, 256),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(256, 2))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, p):
        output = self.layer(p)
        return self.softmax(output)


class RelationNN(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(self.size * 2, 512),
            torch.nn.Sigmoid(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, l, r):
        pairs = torch.cat((l, r), dim=-1)
        output = self.layer(pairs)
        return self.softmax(output)


class MathConstraintsNet(nn.Module):
    def __init__(self, provenance, k, EMD_DIM=8):
        super(MathConstraintsNet, self).__init__()

        # Define NN
        self.property1_nn = PropertyNN(EMD_DIM)
        self.property2_nn = PropertyNN(EMD_DIM)
        self.relation1_nn = RelationNN(EMD_DIM)
        self.relation2_nn = RelationNN(EMD_DIM)

        # Scallop Context
        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self.scl_ctx.add_relation("property_obj1", int, input_mapping=list(range(2)))
        self.scl_ctx.add_relation("relation_obj1_obj2", int, input_mapping=list(range(2)))
        self.scl_ctx.add_relation("property_obj2", int, input_mapping=list(range(2)))

        self.scl_ctx.add_rule("inference(a * b * c) :- property_obj1(a), relation_obj1_obj2(b), property_obj2(c)")

        # Define reasoning module for each inference
        self.inference = self.scl_ctx.forward_function("inference", output_mapping=[(i,) for i in range(2)],
                                                             jit=args.jit, dispatch=args.dispatch)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], logic_order):
        (obj1, obj2) = x

        # Getting all relation in order
        logic_order = logic_order[0]
        property1 = self.property1_nn(obj1) if logic_order[0] == 1 else self.property2_nn(obj1)
        relation1 = self.relation1_nn(obj1, obj2) if logic_order[1] == 1 else self.relation2_nn(obj1, obj2)
        property2 = self.property1_nn(obj2) if logic_order[2] == 1 else self.property2_nn(obj2)
        # print(property1, relation1)
        # Then execute the reasoning module;
        return self.inference(property_obj1=property1, relation_obj1_obj2=relation1, property_obj2=property2)  # Tensor 64 x 19


def bce_loss(output, ground_truth):
    (_, dim) = output.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
    return F.nll_loss(output, ground_truth)


class Trainer():
    def __init__(self, train_loader, test_loader, model_dir, learning_rate, loss, k, provenance):
        self.model_dir = model_dir
        self.network = MathConstraintsNet(provenance, k)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_loss = 10000000000
        if loss == "nll":
            self.loss = nll_loss
        elif loss == "bce":
            self.loss = bce_loss
        else:
            raise Exception(f"Unknown loss function `{loss}`")

    def train_epoch(self, epoch):
        self.network.train()
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        for (data, target, logic_order) in iter:
            self.optimizer.zero_grad()
            output = self.network(data, logic_order)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")

    def test_epoch(self, epoch):
        self.network.eval()
        num_items = len(self.test_loader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            iter = tqdm(self.test_loader, total=len(self.test_loader))
            for (data, target, logic_order) in iter:
                output = self.network(data, logic_order)
                test_loss += self.loss(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                perc = 100. * correct / num_items
                iter.set_description(
                    f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                torch.save(self.network, os.path.join(model_dir, "sum_2_best.pt"))

    def train(self, n_epochs):
        self.test_epoch(0)
        for epoch in range(1, n_epochs + 1):
            self.train_epoch(epoch)
            self.test_epoch(epoch)


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("mnist_sum_2")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size-train", type=int, default=1)
    parser.add_argument("--batch-size-test", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--loss-fn", type=str, default="bce")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--provenance", type=str, default="difftopkproofs")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--dispatch", type=str, default="parallel")
    args = parser.parse_args()

    # Parameters
    n_epochs = args.n_epochs
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test
    learning_rate = args.learning_rate
    loss_fn = args.loss_fn
    k = args.top_k
    provenance = args.provenance
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Data
    data_dir = "../dataset"
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    # Dataloaders
    train_loader, test_loader = MathConstraintsLoader(data_dir, batch_size_train, batch_size_test)

    # Create trainer and train
    trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, loss_fn, k, provenance)
    trainer.train(n_epochs)
