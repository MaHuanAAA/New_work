import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
index = [i for i in range(200)]
#np.random.seed(0)
np.random.shuffle(index)
def get_dataset():
    data_t = loadmat("data/handwritten0.mat")
    x1 = data_t['X'][0][4]
    x2 = data_t['X'][0][2]
    gt = data_t['gt']
    for i in range(21):
        x1[3*i+1] = x1[3*i+1] * 0.0
        x1[3*i] = x1[3*i] * 0.0
    for i in range(68):
        x2[3*i+1] = x2[3*i+1] * 0.0
        x2[3*i] = x2[3*i] * 0.0
    x1 = x1.T
    x2 = x2.T
    test_x1 = []
    test_x2 = []
    test_label = []
    for i in index:
        for j in range(10):
            test_x1.append(x1[i+200*j]/12.0)
            test_x2.append(x2[i+200*j]/600.0) # normalization
            test_label.append(gt[i+200*j])
    test_x1 = torch.tensor(test_x1)  # 转化为张量
    test_x2 = torch.tensor(test_x2)
    test_label = torch.tensor(test_label)
    test_label = test_label.squeeze()
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

    train_dataset = Mydataset(test_x1[0:1200], test_x2[0:1200], test_label[0:1200], 1200)
    test_dataset = Mydataset(test_x1[1600:2000], test_x2[1600:2000], test_label[1600:2000], 400)
    val_dataset = Mydataset(test_x1[1200:1600], test_x2[1200:1600], test_label[1200:1600], 400)
    return train_dataset, test_dataset, val_dataset