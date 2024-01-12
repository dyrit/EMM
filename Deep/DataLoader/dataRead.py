import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def get_data_delicious(bs = 4000):

    x_train = "/home/dp7972/Desktop/DAYOU/Dayou/Data/Delicious_x_train.csv"
    y_train = "/home/dp7972/Desktop/DAYOU/Dayou/Data/Delicious_y_train.csv"
    x_test = "/home/dp7972/Desktop/DAYOU/Dayou/Data/Delicious_x_test.csv"
    y_test = "/home/dp7972/Desktop/DAYOU/Dayou/Data/Delicious_y_test.csv"
    trainDL = ReadDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(trainDL, batch_size = bs)
    testDL = ReadDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(testDL, batch_size = bs)
    return train_loader, test_loader

def get_data(bs = 800):
    x_train = "/home/dp7972/Desktop/DAYOU/Dayou/Data/x_tr.csv"
    y_train = "/home/dp7972/Desktop/DAYOU/Dayou/Data/y_tr.csv"
    x_test = "/home/dp7972/Desktop/DAYOU/Dayou/Data/x_test.csv"
    y_test = "/home/dp7972/Desktop/DAYOU/Dayou/Data/y_test.csv"
    trainDL = ReadDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(trainDL, batch_size = bs)
    testDL = ReadDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(testDL, batch_size = bs)

    return train_loader, test_loader



class ReadDataset(Dataset):
    def __init__(self, x, y):
        data_x = pd.read_csv(x).to_numpy()
        self.data_x = torch.tensor(data_x, dtype = torch.float32)
        data_y = pd.read_csv(y).to_numpy()
        self.data_y = torch.tensor(data_y, dtype = torch.float32)
    
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        return (self.data_x[idx], self.data_y[idx])



def get_data_AE(bs = 800):
    x_train = "/home/dp7972/Desktop/DAYOU/Dayou/Data/x_tr.csv"
    y_train = "/home/dp7972/Desktop/DAYOU/Dayou/Data/y_tr.csv"
    x_test = "/home/dp7972/Desktop/DAYOU/Dayou/Data/x_test.csv"
    y_test = "/home/dp7972/Desktop/DAYOU/Dayou/Data/y_test.csv"
    trainDL = ReadDatasetAE(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(trainDL, batch_size = bs)
    testDL = ReadDatasetAE(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(testDL, batch_size = bs)

    return train_loader, test_loader



class ReadDatasetAE(Dataset):
    def __init__(self, x, y):
        data_x = pd.read_csv(x).to_numpy()
        self.data_x = torch.tensor(data_x, dtype = torch.float32)
        data_y = pd.read_csv(y).to_numpy()
        self.data_y = torch.tensor(data_y, dtype = torch.float32)
    
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        return (self.data_x[idx], self.data_x[idx], self.data_y[idx])
