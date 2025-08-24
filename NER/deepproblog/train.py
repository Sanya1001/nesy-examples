import json
import torch
import torch
from torch.utils.data import TensorDataset
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.dataset import DataLoader, Dataset
from deepproblog.engines import ExactEngine
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.query import Query
from problog.logic import Term, Constant
from dataset import JSONDataOperator, JSONDataset

# --------------------
# 1) Define your PyTorch nets
# --------------------

class PersonNet(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.lin = torch.nn.Linear(emb_dim, 2)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        print('x:', x)
        output = self.lin(x)
        print("PersonNet Output:", self.softmax(output))
        return self.softmax(output)

class WorkNet(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.lin = torch.nn.Linear(2 * emb_dim, 2)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, px, lx):
        output = self.lin(torch.cat([px, lx], dim=-1))
        return self.softmax(output)

# --------------------
# 2) Wrap them as DeepProbLog networks
# --------------------

EMB_DIM = 2

pn1 = Network(PersonNet(EMB_DIM),  "person_net1", batching=True)
pn2 = Network(PersonNet(EMB_DIM),  "person_net2", batching=True)
pn3 = Network(PersonNet(EMB_DIM),  "person_net3", batching=True)

wn1 = Network(WorkNet(EMB_DIM),    "work_net1",       batching=True)
wn2 = Network(WorkNet(EMB_DIM),    "work_net2",       batching=True)
wn3 = Network(WorkNet(EMB_DIM),    "work_net3",       batching=True)

for net in (pn1, pn2, pn3, wn1, wn2, wn3):
    net.optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

# --------------------
# 3) Build the Problog Model
# --------------------

model = Model("ner.pl", [pn1, pn2, pn3, wn1, wn2, wn3])
model.set_engine(ExactEngine(model), cache=True)

# --------------------
# 4) Load your datasets
# --------------------

model.add_tensor_source("train", JSONDataset("dataset/train.json"))
model.add_tensor_source("test", JSONDataset("dataset/test.json"))

train_dataset = JSONDataOperator("train")
test_dataset = JSONDataOperator("test")

print(len(train_dataset), len(test_dataset))

loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
train = train_model(model, loader, 1, log_iter = 100, profile = 0)
model.save_state("model/model_state.pth")

train.logger.comment(json.dumps(model.get_hyperparameters()))
train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_dataset, verbose=1).accuracy())
)
train.logger.write_to_file("log.txt")