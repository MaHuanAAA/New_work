import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

def get_dataset():
    data_t = loadmat("data\cub.mat")
    x1 = data_t['X'][0][0].T
    x2 = data_t['X'][0][1].T
    gt = data_t['gt']

    # 打乱索引,固定np种子是为了用相同的样本训练不同损失函数比较
    index = [i for i in range(len(x1))] # test_data为测试数据
    np.random.seed(0)
    np.random.shuffle(index) # 打乱索引
    test_x1 = []
    test_x2 = []
    test_label = []
    for i in index:
        test_x1.append(x1[i])
        test_x2.append(x2[i])
        test_label.append(gt[i])
    test_x1 = torch.tensor(test_x1)  # 转化为张量
    test_x2 = torch.tensor(test_x2)
    test_label = torch.tensor(test_label)
    test_label = test_label.squeeze()
    test_label = test_label-1
    test_label = test_label.to(torch.int64)
    test_x1 = test_x1.to(torch.float32)
    test_x2 = test_x2.to(torch.float32)

    #定义Mydataset继承自Dataset,并重写__getitem__和__len__

    class Mydataset(Dataset):
        def __init__(self, x, x2, y, num):
            super(Mydataset, self).__init__()
            self.num = num

            self.x_train = x
            self.x2_train = x2
            self.y_train = y
        # indexing

        def __getitem__(self, index):
            return self.x_train[index], self.x2_train[index], self.y_train[index]

        def __len__(self):
            return self.num

    train_dataset = Mydataset(test_x1[0:400], test_x2[0:400], test_label[0:400], 400)
    test_dataset = Mydataset(test_x1[500:600], test_x2[500:600], test_label[500:600], 100)
    val_dataset = Mydataset(test_x1[400:500], test_x2[400:500], test_label[400:500], 100)
    return train_dataset, test_dataset, val_dataset

