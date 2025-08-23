import os
os.environ["SWI_HOME_DIR"] = "/opt/homebrew/opt/swi-prolog/libexec"
from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from dataset import MathConstraintsEMB, MathConstraintsDataset


# 1. Define Pytorch Network

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
            torch.nn.Linear(512, 512))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, l, r):
        pairs = torch.cat((l, r), dim=-1)
        output = self.layer(pairs)
        return self.softmax(output)


# 2. NN in DeepProbLog NN and set optimizer

EMD_DIM = 8

property_nn1 = Network(PropertyNN(size=EMD_DIM), "property_net1", batching=True)
property_nn2 = Network(PropertyNN(size=EMD_DIM), "property_net2", batching=True)
relation_nn1 = Network(RelationNN(size=EMD_DIM), "relation_net1", batching=True)
relation_nn2 = Network(RelationNN(size=EMD_DIM), "relation_net2", batching=True)

for net in (property_nn1, property_nn2, relation_nn1, relation_nn2):
    net.optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)


# 3. Setting up Model
model = Model("math.pl", [property_nn1, property_nn2, relation_nn1, relation_nn2])
model.set_engine(ExactEngine(model), cache=True)
model.add_tensor_source("train", MathConstraintsEMB("dataset/train.json"))
model.add_tensor_source("test", MathConstraintsEMB("dataset/test.json"))

train_dataset = MathConstraintsDataset("train", "dataset/train.json")
test_dataset = MathConstraintsDataset("test", "dataset/test.json")
print(len(train_dataset))
# Train the model
loader = DataLoader(train_dataset, 2, False)

print("Accuracy Before Training {}".format(get_confusion_matrix(model, test_dataset, verbose=1).accuracy()))

train = train_model(model, loader, 1, log_iter=100, profile=0)

train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_dataset, verbose=1).accuracy())
)
#
# model.add_tensor_source("train", MNIST_train)
# model.add_tensor_source("test", MNIST_test)
#
# loader = DataLoader(train_set, 2, False)
# train = train_model(model, loader, 1, log_iter=100, profile=0)
# model.save_state("snapshot/" + name + ".pth")
# train.logger.comment(dumps(model.get_hyperparameters()))
# train.logger.comment(
#     "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
# )
# train.logger.write_to_file("log/" + name)
