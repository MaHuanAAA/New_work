import torch
from torch import nn, optim
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def set_temperature(logits, label):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    nll_criterion = nn.CrossEntropyLoss().to(device)

    class TemperatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        def forward(self, logits):
            temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
            logits = logits / temperature
            return logits, self.temperature

    model_T = TemperatureModel()
    optimizer = optim.Adam(
        model_T.parameters(), lr=0.01, weight_decay=1e-4
    )
    for i in range(40):
        TC_pred, _ = model_T(logits)
        optimizer.zero_grad()
        loss = nll_criterion(TC_pred, label)
        loss.backward()
        optimizer.step()

    _, T = model_T(logits)
    return T

def set_temperature2(logits, label):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """

    nll_criterion = nn.CrossEntropyLoss().to(device)

    class TemperatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        def forward(self, logits):
            temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
            logits = logits / temperature
            return logits, self.temperature

    model_T = TemperatureModel()
    optimizer = optim.Adam(
        model_T.parameters(), lr=0.01, weight_decay=1e-4
    )
    for i in range(40):
        TC_pred, _ = model_T(logits)
        optimizer.zero_grad()
        loss = nll_criterion(TC_pred, label)
        loss.backward()
        optimizer.step()

    _, T = model_T(logits)
    return T