from torch.utils.data import Dataset
from utils.datasets import get_dataset
import torch
from torch.nn import functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from utils.L_duq import L_DUQ

train_dataset, test_dataset, val_dataset = get_dataset()   # get dataset

num_classes = 10
embedding_size = 128
gamma = 0.999
epoch = 200


def train_model(l_gradient_penalty, length_scale):

    model = L_DUQ(
        num_classes,
        embedding_size,
        length_scale,
        gamma,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.03, weight_decay=1e-4
    )

    def output_transform_acc(output):
        y_pred, y, _, _, _, _ = output
        return y_pred, torch.argmax(y, dim=1)

    def output_transform_acc2(output):
        _, y, _, _, y_pred2, _ = output
        return y_pred2, torch.argmax(y, dim=1)

    def output_transform_acc_multi(output):
        _, y, _, _, _, y_pred_max = output
        return y_pred_max, torch.argmax(y, dim=1)

    def calc_gradient_penalty(x, y_pred_sum):
        gradients = torch.autograd.grad(
            outputs=y_pred_sum,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred_sum),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def get_dataloader():
        dl_train = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True
        )
        dl_val = torch.utils.data.DataLoader(
            val_dataset, batch_size=100, shuffle=False, num_workers=0
        )
        dl_test = torch.utils.data.DataLoader(
            test_dataset, batch_size=100, shuffle=False, num_workers=0
        )

        return dl_train, dl_test, dl_val

    def train_dataset_acc_ratio(y_pred, y_pred2, y):
        conf1 = []
        conf2 = []
        con_max1 = []
        con_max2 = []
        for i in y_pred:
            conf1.append(i.detach().numpy())
        for i in y_pred2:
            conf2.append(i.detach().numpy())
        for i in range(len(conf1)):
            con_max1.append(max(conf1[i]))
            con_max2.append(max(conf2[i]))
        acc_1 = 0
        acc_2 = 0
        for i in range(len(y)):
            if torch.argmax(y[i]) == torch.argmax(y_pred[i]):
                acc_1 += 1
            if torch.argmax(y[i]) == torch.argmax(y_pred2[i]):
                acc_2 += 1
        acc_1 = acc_1 / len(y) + 1e-6
        acc_2 = acc_2 / len(y) + 1e-6

        return acc_1/acc_2

    def get_val_dataset_acc_ratio():
        evaluator.run(dl_val)
        acc_1 = evaluator.state.metrics["accuracy"] + 1e-6
        acc_2 = evaluator.state.metrics["accuracy2"] + 1e-6

        return acc_1/acc_2

    def get_conf_ratio(y_pred, y_pred2):
        con_sum = torch.sum(y_pred) + 1e-6
        con_sum2 = torch.sum(y_pred2) + 1e-6

        return con_sum/con_sum2

    # get the pred from multi-views
    def get_pred_max(y_pred, y_pred2):
        conf_t1 = []
        conf_t2 = []
        for i in y_pred:
            conf_t1.append(i.detach().numpy())
        for i in y_pred2:
            conf_t2.append(i.detach().numpy())
        pred_max = [[0 for j in range(10)] for i in range(100)]
        for i in range(len(conf_t1)):  # 获得两个模型置信度较大的值
            for j in range(len(conf_t1[0])):
                pred_max[i][j] = max(conf_t1[i][j], conf_t2[i][j])
        pred_max = torch.tensor(pred_max)

        return pred_max

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, x2, y = batch
        y = F.one_hot(y, num_classes=10).float()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, x2, y = x.to(device), x2.to(device), y.to(device)

        x.requires_grad_(True)
        x2.requires_grad_(True)

        z, y_pred, z2, y_pred2 = model(x, x2)

        # get acc ratio on tarin_dataset
        acc_ratio = train_dataset_acc_ratio(y_pred, y_pred2, y)

        # get acc ratio on val_dataset
        # acc_ratio = get_val_dataset_acc_ratio()
        conf_ratio = get_conf_ratio(y_pred, y_pred2)

        loss1 = F.binary_cross_entropy(y_pred, y)
        loss2 = F.binary_cross_entropy(y_pred2, y)  #RBF Loss
        loss3 = l_gradient_penalty * (calc_gradient_penalty(x, y_pred.sum(1)) + calc_gradient_penalty(x2, y_pred2.sum(1)))       #gradient panelty
        loss = loss1 + loss2 + loss3
        loss += 0.1*(1-acc_ratio/(conf_ratio))**2                         #acc panelty

        x.requires_grad_(False)
        x2.requires_grad_(False)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.update_embeddings(x, y)
            model.update_embeddings2(x2, y)

        return loss.item()

    def eval_step(engine, batch):
        model.eval()

        x, x2, y = batch
        y = F.one_hot(y, num_classes=10).float()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, x2, y = x.to(device), x2.to(device), y.to(device)

        x.requires_grad_(True)
        x2.requires_grad_(True)

        z, y_pred, z2, y_pred2 = model(x, x2)
        pred_max = get_pred_max(y_pred, y_pred2)

        return y_pred, y, x, y_pred.sum(1), y_pred2, pred_max

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Accuracy(output_transform=output_transform_acc)
    metric.attach(evaluator, "accuracy")
    metric = Accuracy(output_transform=output_transform_acc2)
    metric.attach(evaluator, "accuracy2")
    metric = Accuracy(output_transform=output_transform_acc_multi)
    metric.attach(evaluator, "accuracy_multi")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.2
    )

    dl_train, dl_test, dl_val = get_dataloader()

    pbar = ProgressBar()
    pbar.attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        scheduler.step()

    trainer.run(dl_train, max_epochs=epoch)
    evaluator.run(dl_test)
    test_accuracy = evaluator.state.metrics["accuracy"]
    test_accuracy2 = evaluator.state.metrics["accuracy2"]
    test_accuracy_multi = evaluator.state.metrics["accuracy_multi"]

    return model, test_accuracy, test_accuracy2, test_accuracy_multi


if __name__ == "__main__":

    # Finding length scale - decided based on validation accuracy
    l_gradient_penalties = [0.1, 0.2, 0.3, 0.4, 0.5]
    length_scales = [0.1]  #best  0.1

    repetition = 1 # Increase for multiple repetitions
    final_model = False  # set true for final model to train on full train set

    results = {}

    for l_gradient_penalty in l_gradient_penalties:
        for length_scale in length_scales:

            for _ in range(repetition):
                print(" ### NEW MODEL ### ", length_scale, l_gradient_penalty)
                model, test_accuracy, test_accuracy2, acc_multi= train_model(
                    l_gradient_penalty, length_scale
                )

                print("C1  ", test_accuracy)
                print("C2  ", test_accuracy2)
                print("multi ", acc_multi)

