import torch
import numpy as np
from torch.utils.data import Dataset

class data_set3D(Dataset):
    def __init__(self, data, label):
        super(data_set3D, self).__init__()
        # 为了适应 pytorch 结构，数据要做 transpose
        dataExpend = np.expand_dims(data, axis=4)
        dataTrans = dataExpend.transpose(0, 4, 3, 1, 2)
        self.data = torch.FloatTensor(dataTrans)
        self.label = torch.LongTensor(label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class data_set2D(Dataset):
    def __init__(self, data, label):
        super(data_set2D, self).__init__()
        # 为了适应 pytorch 结构，数据要做 transpose
        dataTrans = data.transpose(0, 3, 1, 2)
        self.data = torch.FloatTensor(dataTrans)
        self.label = torch.LongTensor(label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

