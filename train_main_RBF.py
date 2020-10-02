from torch.utils.data import Dataset
from utils.datasets import get_dataset
import torch
from utils.calibration import calibrate
from torch.nn import functional as F
from ignite.engine import Engine
from utils.model import Model, Model2
from utils.draw_picture import draw_loss, draw_two
from utils.ECE import ece_score
from utils.Temperature import set_temperature, set_temperature2
import xlsxwriter

workbook = xlsxwriter.Workbook('data.xlsx')
worksheet = workbook.add_worksheet()

train_dataset, test_dataset, val_dataset = get_dataset()   # get dataset
num_classes = 10
epoch = 300


def train_model(learning_rate, scale, bins, la):
    model = Model()
    model2 = Model2()
    sf = torch.nn.Softmax(dim=1)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model2.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    optimizer2 = torch.optim.Adam(
        model2.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    def get_dataloader():
        dl_train = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True
        )
        dl_val = torch.utils.data.DataLoader(
            val_dataset, batch_size=400, shuffle=False, num_workers=0
        )
        dl_test = torch.utils.data.DataLoader(
            test_dataset, batch_size=400, shuffle=False, num_workers=0
        )

        return dl_train, dl_test, dl_val

    # get the pred from multi-views
    def get_pred_max(y_pred, y_pred2):
        pred_max = torch.max(y_pred, y_pred2)
        return pred_max

    def get_acc(y_pred, y):
        acc_1 = 0
        for i in range(len(y)):
            if torch.argmax(y[i]) == torch.argmax(y_pred[i]):
                acc_1 += 1

        return acc_1/len(y)

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, x2, y = batch
        y = F.one_hot(y, num_classes=10).float()

        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        x, x2, y = x.to(device), x2.to(device), y.to(device)

        x.requires_grad_(True)

        y_pred = sf(model(x))

        loss = F.binary_cross_entropy(y_pred, y)
        x.requires_grad_(False)

        loss.backward()
        optimizer.step()

        return loss.item()

    def step2(engine, batch):
        model2.train()
        optimizer2.zero_grad()

        x, x2, y = batch
        y = F.one_hot(y, num_classes=10).float()

        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        x, x2, y = x.to(device), x2.to(device), y.to(device)

        x2.requires_grad_(True)

        y_pred = sf(model2(x2))

        loss2 = F.binary_cross_entropy(y_pred, y)
        x2.requires_grad_(False)

        loss2.backward()
        optimizer2.step()

        return loss2.item()

    def val_step():
        with torch.no_grad():
            for batch in dl_val:
                x, x2, y_list = batch
                y = F.one_hot(y_list, num_classes=10).float()

                device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
                x, x2, y = x.to(device), x2.to(device), y.to(device)

                x.requires_grad_(True)
                x2.requires_grad_(True)
                y_pred_ = model(x)
                y_pred2_ = model2(x2)
                y_pred = sf(y_pred_)
                y_pred2 = sf(y_pred2_)

        return y_pred, y_pred2, y, y_pred_, y_pred2_, y_list

    def eval_step():
        acc1, acc2, acc_m, acc_m_c = 0, 0, 0, 0
        with torch.no_grad():
            for batch in dl_test:
                x, x2, y = batch
                y = F.one_hot(y, num_classes=10).float()

                device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
                x, x2, y = x.to(device), x2.to(device), y.to(device)
                global v1count
                x.requires_grad_(True)
                x2.requires_grad_(True)
                y_pred_ = model(x)
                y_pred2_ = model2(x2)
                y_pred = sf(y_pred_)
                y_pred2 = sf(y_pred2_)
                acc1 = get_acc(y_pred, y)
                acc2 = get_acc(y_pred2, y)
                y_pred_max = get_pred_max(y_pred, y_pred2)
                acc_m = get_acc(y_pred_max, y)
                y_pred_ts = sf(y_pred_/temp)
                y_pred2_c = sf(y_pred2_/temp2)
                y_pred_max_ts = get_pred_max(y_pred_ts, y_pred2_c)
                acc_m_ts = get_acc(y_pred_max_ts, y)
                y_pred_c = calibrate(y_pred, y_pred2, y, val_pred1, val_pred2, y_val, scale, bins, la)
                y_pred_max_c = get_pred_max(y_pred_c, y_pred2)
                acc_m_c = get_acc(y_pred_max_c, y)
        return acc1, acc2, acc_m, acc_m_ts, acc_m_c

    trainer = Engine(step)
    trainer2 = Engine(step2)

    dl_train, dl_test, dl_val = get_dataloader()

    trainer.run(dl_train, max_epochs=epoch)
    trainer2.run(dl_train, max_epochs=epoch)
    val_pred1, val_pred2, y_val, y_pred_v, y_pred2_v, y_list = val_step()
    temp = set_temperature(y_pred_v, y_list)
    temp2 = set_temperature2(y_pred2_v, y_list)
    acc1, acc2, acc_m, acc_m_ts, acc_m_c = eval_step()

    return model, acc1, acc2, acc_m, acc_m_ts, acc_m_c


if __name__ == "__main__":

    bas = [0.1]
    bins = [4]
    repetition = 1  # Increase for multiple repetitions
    final_model = False  # set true for final model to train on full train set

    results = {}
    row = 1
    for _ in range(repetition):
        for bin in bins:
            for ba in bas:

                print(" ### NEW MODEL ### ", bin, ba)
                model, acc1, acc2, acc_m, acc_m_ts, acc_m_c = train_model(
                    0.01, 1, bin, ba
                )
                print("acc1: {}    acc2: {}    acc_m: {}   acc_m_ts: {}   acc_m_c: {}\n\n".format(acc1, acc2, acc_m, acc_m_ts, acc_m_c))

workbook.close()