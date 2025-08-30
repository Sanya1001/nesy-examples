import json
import os
import torch
import scallopy
from typing import *

import torch.optim as optim
from tqdm import tqdm
from argparse import ArgumentParser
import random

from torch.utils.data import Dataset
from dataset import JSONDataset

EMB_DIM = 2

class NERDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = JSONDataset("nesy-examples/NER/dataset/train.json")
        self.index_map = list(range(len(self.data)))
        random.shuffle(self.index_map)

    def __len__(self):
        return len(self.data) // 6

    def __getitem__(self, idx):
        p1 = self.data[idx*6]
        p2 = self.data[idx*6+1]
        p3 = self.data[idx*6+2]
        l1 = self.data[idx*6+3]
        l2 = self.data[idx*6+4]
        l3 = self.data[idx*6+5]

        output = int(self.data.raw_data[idx]["condition_1"][0]) * int(self.data.raw_data[idx]["condition_2"][0])
        output = int(self.data.raw_data[idx]["condition_1"][0])

        return ((p1, p2, p3, l1, l2, l3), output)

def ner_loader():
   train_loader = torch.utils.data.DataLoader(NERDataset(), batch_size=1, shuffle=True)
   test_loader = torch.utils.data.DataLoader(NERDataset(), batch_size=1, shuffle=False)

   return train_loader, test_loader

class PersonNet(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.lin = torch.nn.Linear(emb_dim, 2)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        output = self.lin(x)
        return output

class WorkNet(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.lin = torch.nn.Linear(2 * emb_dim, 2)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, px, lx):
        output = self.lin(torch.cat([px, lx], dim=-1))
        return output
    
class NER2Net(torch.nn.Module):
    def __init__(self, provenance, k):
        super(NER2Net, self).__init__()

        self.person_net = PersonNet(emb_dim=EMB_DIM)
        self.work_net = WorkNet(emb_dim=EMB_DIM)

        # Scallop Context
        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self.scl_ctx.add_relation("is_real_person1", int, input_mapping=list(range(2)))
        self.scl_ctx.add_relation("is_real_person2", int, input_mapping=list(range(2)))
        self.scl_ctx.add_relation("is_real_person3", int, input_mapping=list(range(2)))

        self.scl_ctx.add_relation("work_in1", int, input_mapping=list(range(2)))
        self.scl_ctx.add_relation("work_in2", int, input_mapping=list(range(2)))
        self.scl_ctx.add_relation("work_in3", int, input_mapping=list(range(2)))

        self.scl_ctx.add_rule("check(P1* P2 * W1 * W2 + P3 * W3) :- is_real_person1(P1), work_in1(W1), is_real_person2(P2), work_in2(W2), is_real_person3(P3), work_in3(W3)")

        # print("printing...", list(self.scl_ctx.relations()))

        self.check = self.scl_ctx.forward_function("check", output_mapping=[(i,) for i in range(2)], jit=False)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        print("here")
        (p1, p2, p3, l1, l2, l3) = x

        # Get person predictions
        p1_pred = self.person_net(p1)
        p2_pred = self.person_net(p2)
        p3_pred = self.person_net(p3)

        # Get work predictions
        w1_pred = self.work_net(p1, l1)
        w2_pred = self.work_net(p2, l2)
        w3_pred = self.work_net(p3, l3)

        # Get Scallop outputs
        scl_output = self.check(
            is_real_person1 = p1_pred,
            is_real_person2 = p2_pred,
            is_real_person3 = p3_pred,
            work_in1 = w1_pred,
            work_in2 = w2_pred,
            work_in3 = w3_pred
        )

        print(scl_output)

        return scl_output
    
def bce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    print(predictions.shape, targets.shape)
    targets = targets.long()
    bce = torch.nn.CrossEntropyLoss()
    return bce(predictions, targets)
    

class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate, loss, k, provenance):
    self.model_dir = model_dir
    self.network = NER2Net(provenance, k)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.loss = bce_loss

  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (data, target) in iter:
      self.optimizer.zero_grad()
      output = self.network(data)
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
      for (data, target) in iter:
        output = self.network(data)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
      if test_loss < self.best_loss:
        self.best_loss = test_loss
        torch.save(self.network, os.path.join(self.model_dir, "ner_scl.pt"))

  def train(self, n_epochs):
    self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("ner_scl")
  parser.add_argument("--n-epochs", type=int, default=1)
  parser.add_argument("--batch-size-train", type=int, default=1)
  parser.add_argument("--batch-size-test", type=int, default=1)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
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
#   data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../dataset"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../model/ner_scl"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  # train_dataset = JSONDataOperator("train")
  # test_dataset = JSONDataOperator("test")
  train_loader, test_loader = ner_loader()


  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, loss_fn, k, provenance)
  trainer.train(n_epochs)