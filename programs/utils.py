import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
from tqdm import tqdm

def data_info(dataset):
    if dataset == 'UCI_HAR':
        input_nc = 6
        segment_size = 128
        class_num = 6 
    elif dataset == 'OPPORTUNITY':
        input_nc = 113
        segment_size = 90
        class_num = 17
    elif dataset == 'PAMAP2':
        input_nc = 31
        segment_size = 512
        class_num = 18
    elif dataset == "UniMiB-SHAR":
        input_nc = 3
        segment_size = 151
        class_num = 17
    elif dataset == "DSADS":
        input_nc = 45
        segment_size = 125
        class_num = 19
    elif dataset == "REALDISP":
        input_nc = 117
        segment_size = 250
        class_num = 33
    elif dataset == "KU-HAR":
        input_nc = 6
        segment_size = 300
        class_num = 18
    else:
        raise ValueError("The dataset does not exist")
    return input_nc, segment_size, class_num

def Read_Data(dataset):
    data_path = os.path.join('Data', 'preprocessed', dataset)
    train_X = np.load(data_path+'/train_x.npy')
    train_Y = np.load(data_path+'/train_y.npy')
    test_X = np.load(data_path+'/test_x.npy')
    test_Y = np.load(data_path+'/test_y.npy')

    return To_DataSet(train_X, train_Y), To_DataSet(test_X, test_Y), test_Y

class To_DataSet(Dataset):
    def __init__(self, X, Y):
        self.data_num = Y.shape[0]
        self.x = torch.as_tensor(X)
        self.y = torch.as_tensor(Y)#torch.max(torch.as_tensor(Y), 1)[1]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.data_num



def input_pipeline(dataset, input_nc, bs):
    train_data, eval_data, y_test_unary  = Read_Data(dataset)
    train_queue = DataLoader(
        train_data, batch_size=bs,shuffle=True,
        pin_memory=True, num_workers=0)
    eval_queue = DataLoader(
        eval_data, batch_size=bs,shuffle=False,
        pin_memory=True, num_workers=0)
    return train_queue, eval_queue, y_test_unary

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    total = target.size(0)
    correct = (predicted == target).sum()

    return float(correct) / total

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
