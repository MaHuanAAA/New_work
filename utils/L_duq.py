import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1024, 512)
        self.linear2 = torch.nn.Linear(512, 128)
        self.linear3 = torch.nn.Linear(300, 128)

    def compute_features(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x

    def compute_features2(self, x):
        x = F.relu(self.linear3(x))
        return x


class L_DUQ(Model):
    def __init__(
        self,
        num_classes,
        embedding_size,
        length_scale,
        gamma,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, 128), 0.05)
        )

        self.register_buffer("N", torch.ones(num_classes) * 12)
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)
#2
        self.W2 = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, 128), 0.05)
        )

        self.register_buffer("N2", torch.ones(num_classes) * 12)
        self.register_buffer(
            "m2", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m2 = self.m2 * self.N2.unsqueeze(0)

#2
        self.sigma = length_scale

    def update_embeddings(self, x, y):
        z = self.last_layer(self.compute_features(x))

        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum
#2
    def update_embeddings2(self, x, y):
        z = self.last_layer(self.compute_features2(x))

        # normalizing value per class, assumes y is one_hot encoded
        self.N2 = self.gamma * self.N2 + (1 - self.gamma) * y.sum(0)

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m2 = self.gamma * self.m2 + (1 - self.gamma) * features_sum
#2
    def last_layer(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)
        return z

    def last_layer2(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W2)
        return z

    def output_layer(self, z):
        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        distances = (-(diff ** 2)).mean(1).div(2 * self.sigma ** 2).exp()

        return distances

    def output_layer2(self, z):
        embeddings = self.m2 / self.N2.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        distances = (-(diff ** 2)).mean(1).div(2 * self.sigma ** 2).exp()

        return distances

    def forward(self, x1, x2):
        z1 = self.last_layer(self.compute_features(x1))
        y_pred1 = self.output_layer(z1)
        z2 = self.last_layer(self.compute_features2(x2))
        y_pred2 = self.output_layer(z2)
        return z1, y_pred1, z2, y_pred2


class SoftmaxModel(Model):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.last_layer = nn.Linear(1024, num_classes)
        self.output_layer = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.last_layer(self.compute_features(x))
        y_pred = F.log_softmax(z, dim=1)

        return y_pred
