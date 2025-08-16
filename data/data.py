import numpy as np
import os
import torch
from torch.utils.data import Dataset


def log_transform(data, k=1, c=0):
    return (np.log1p(np.abs(k * data) + c)) * np.sign(data)


def log_detransform(data, k=1, c=0):
    return np.sign(data) * ((np.exp(data)-1-c)/k)


def minmax_normalize(vid, vmin, vmax, scale=2):
    vid -= vmin
    vid /= (vmax - vmin)
    return (vid - 0.5) * 2 if scale == 2 else vid


def minmax_denormalize(vid, vmin, vmax, scale=2):
    if scale == 2:
        vid = vid / 2 + 0.5
    return vid * (vmax - vmin) + vmin


class SeismicData(Dataset):
    def __init__(self, dataset, task, data_path):
        
        if dataset == 'fvb':
            X_train, y_train, X_test, y_test = data_fvb_train(task=task, path=data_path)
        elif dataset == 'cva':
            X_train, y_train, X_test, y_test = data_cva_train(task=task, path=data_path)
        elif dataset == 'cfa':
            X_train, y_train, X_test, y_test = data_cfa_train(task=task, path=data_path)
        elif dataset == 'sta':
            X_train, y_train, X_test, y_test = data_sta_train(task=task, path=data_path)
        else:
            raise NotImplementedError(f"dataset name should be 'fvb', 'cva', 'cfa', or 'sta'")

        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        self.data_size = self.train_x[0].shape[0]


    def __getitem__(self, index):

        seism_data = self.train_x[0][index]
        param_data = self.train_x[1][index]
        label_data = self.train_y[index]

        seism = torch.from_numpy(seism_data).float()
        param = torch.from_numpy(param_data).float()
        label = torch.from_numpy(label_data).float()

        return seism, param, label


    def __len__(self):
        if not self.data_size:
            return 1000
        return self.data_size
    

    def get_testdata(self):
        seism_data = self.test_x[0]
        param_data = self.test_x[1]
        label_data = self.test_y

        seism = torch.from_numpy(seism_data).float()
        param = torch.from_numpy(param_data).float()
        label = torch.from_numpy(label_data).float()

        return seism, param, label


def data_fvb_train(task, path, start_file=1, end_file=None):
    if end_file is None:
        end_file = 10 

    X_train_branch = []
    X_train_trunk = []
    y_train = []

    for i in range(start_file, end_file + 1):
        y_train.append(np.load(os.path.join(path, f"velocity/model{i}.npy")))
        X_train_branch.append(np.load(os.path.join(path, f"seismic/{task}/data{i}.npy")))

        if task == 'loc' or task == 'f':
            X_train_trunk.append(np.load(os.path.join(path, f"{task}/{task}{i}.npy")))
        elif task == 'loc_f':
            X_train_trunk1 = np.load(os.path.join(path, f"loc/loc{i}.npy"))
            X_train_trunk2 = np.load(os.path.join(path, f"f/f{i}.npy"))
            X_train_trunk.append(np.concatenate([X_train_trunk1, X_train_trunk2], axis=1))

    y_train = np.concatenate(y_train).astype(np.float16)
    y_train = minmax_normalize(y_train, 1500, 4500)

    X_train_branch = np.concatenate(X_train_branch).astype(np.float16)
    X_train_branch = log_transform(X_train_branch)
    X_train_branch = minmax_normalize(X_train_branch, log_transform(-30), log_transform(60))

    if task == 'loc':
        X_train_trunk = np.concatenate(X_train_trunk).astype(np.float16)
        X_train_trunk = log_transform(X_train_trunk)
        X_train_trunk = minmax_normalize(X_train_trunk, log_transform(0), log_transform(690))
    elif task == 'f':
        X_train_trunk = np.concatenate(X_train_trunk).astype(np.float16)
        X_train_trunk = log_transform(X_train_trunk)
        X_train_trunk = minmax_normalize(X_train_trunk, log_transform(5), log_transform(25))
    elif task == 'loc_f':
        X_train_trunk = np.concatenate(X_train_trunk, axis=0).astype(np.float16)
        X_train_trunk1 = X_train_trunk[:, :X_train_trunk1.shape[1]]
        X_train_trunk2 = X_train_trunk[:, X_train_trunk1.shape[1]:]
        X_train_trunk1 = log_transform(X_train_trunk1)
        X_train_trunk2 = log_transform(X_train_trunk2)
        X_train_trunk1 = minmax_normalize(X_train_trunk1, log_transform(0), log_transform(690))
        X_train_trunk2 = minmax_normalize(X_train_trunk2, log_transform(5), log_transform(25))
        X_train_trunk = np.concatenate([X_train_trunk1, X_train_trunk2], axis=1)

    return (X_train_branch, X_train_trunk), y_train



def data_cva_train(task, path):

    num_dataset = 48
    y_train = []
    for i in range(1, num_dataset + 1, 1):
        y_train.append(np.load(os.path.join(path, f"velocity/model{i}.npy")))
    y_train = np.concatenate(y_train).astype(np.float16)
    y_train = minmax_normalize(y_train, 1500, 4500)

    y_test = np.load(os.path.join(path, f"velocity/model60.npy"))[:50].astype(np.float16)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_train_branch = []
    for i in range(1, num_dataset + 1, 1):
        X_train_branch.append(np.load(os.path.join(path, f"seismic/{task}/data{i}.npy")))
    X_train_branch = np.concatenate(X_train_branch).astype(np.float16)
    X_train_branch = log_transform(X_train_branch)
    X_train_branch = minmax_normalize(X_train_branch, log_transform(-30), log_transform(60))

    X_test_branch = np.load(os.path.join(path, f"seismic/{task}/data60.npy"))[:50].astype(np.float16)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_train_trunk = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk.append(np.load(os.path.join(path, f"{task}/{task}{i}.npy")))
        X_train_trunk = np.concatenate(X_train_trunk).astype(np.float16)
        X_train_trunk = log_transform(X_train_trunk)
        if task == 'loc':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(5), log_transform(25))
        X_test_trunk = np.load(os.path.join(path, f"{task}/{task}60.npy"))[:50].astype(np.float16)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))
        
    elif task == 'loc_f':
        X_train_trunk1 = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk1.append(np.load(os.path.join(path, f"loc/loc{i}.npy")))
        X_train_trunk1 = np.concatenate(X_train_trunk1).astype(np.float16)
        X_train_trunk1 = log_transform(X_train_trunk1)
        X_train_trunk1 = minmax_normalize(X_train_trunk1, log_transform(0), log_transform(690))
        
        X_train_trunk2 = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk2.append(np.load(os.path.join(path, f"f/f{i}.npy")))
        X_train_trunk2 = np.concatenate(X_train_trunk2).astype(np.float16)
        X_train_trunk2 = log_transform(X_train_trunk2)
        X_train_trunk2 = minmax_normalize(X_train_trunk2, log_transform(5), log_transform(25))

        X_train_trunk = np.concatenate([X_train_trunk1, X_train_trunk2], axis=1)

        X_test_trunk1 = np.load(os.path.join(path, f"loc/loc60.npy"))[:50].astype(np.float16)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))
        
        X_test_trunk2 = np.load(os.path.join(path, f"f/f60.npy"))[:50].astype(np.float16)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_train = (X_train_branch, X_train_trunk)
    X_test = (X_test_branch, X_test_trunk)

    return X_train, y_train, X_test, y_test



def data_cfa_train(task, path):
    num_dataset = 96
    y_train = []
    for i in [2, 3, 4]:
        for j in range(num_dataset//3):
            y_train.append(os.path.join(path, np.load(os.path.join(path, f"velocity/vel{i}_1_{j}.npy"))))
    y_train = np.concatenate(y_train).astype(np.float16)
    y_train = minmax_normalize(y_train, 1500, 4500)

    y_test = np.load(os.path.join(path, f"velocity/vel3_1_35.npy"))[:50].astype(np.float16)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_train_branch = []
    for i in [2, 3, 4]:
        for j in range(num_dataset//3):
            X_train_branch.append(np.load(os.path.join(path, f"seismic/{task}/seis{i}_1_{j}.npy")))
    X_train_branch = np.concatenate(X_train_branch).astype(np.float16)
    X_train_branch = log_transform(X_train_branch)
    X_train_branch = minmax_normalize(X_train_branch, log_transform(-30), log_transform(60))

    X_test_branch = np.load(os.path.join(path, f"seismic/{task}/seis3_1_35.npy"))[:50].astype(np.float16)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_train_trunk = []
        for i in [2, 3, 4]:
            for j in range(num_dataset // 3):
                X_train_trunk.append(np.load(os.path.join(path, f"{task}/{task}{i}_1_{j}.npy")))
        X_train_trunk = np.concatenate(X_train_trunk).astype(np.float16)
        X_train_trunk = log_transform(X_train_trunk)
        if task == 'loc':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(5), log_transform(25))
        X_test_trunk = np.load(os.path.join(path, f"{task}/{task}3_1_35.npy"))[:50].astype(np.float16)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_train_trunk1 = []
        for i in [2, 3, 4]:
            for j in range(num_dataset // 3):
                X_train_trunk1.append(np.load(os.path.join(path, f"loc/loc{i}_1_{j}.npy")))
        X_train_trunk1 = np.concatenate(X_train_trunk1).astype(np.float16)
        X_train_trunk1 = log_transform(X_train_trunk1)
        X_train_trunk1 = minmax_normalize(X_train_trunk1, log_transform(0), log_transform(690))

        X_train_trunk2 = []
        for i in [2, 3, 4]:
            for j in range(num_dataset // 3):
                X_train_trunk2.append(np.load(os.path.join(path, f"f/f{i}_1_{j}.npy")))
        X_train_trunk2 = np.concatenate(X_train_trunk2).astype(np.float16)
        X_train_trunk2 = log_transform(X_train_trunk2)
        X_train_trunk2 = minmax_normalize(X_train_trunk2, log_transform(5), log_transform(25))

        X_train_trunk = np.concatenate([X_train_trunk1, X_train_trunk2], axis=1)

        X_test_trunk1 = np.load(os.path.join(path, f"loc/loc3_1_35.npy"))[:50].astype(np.float16)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = np.load(os.path.join(path, f"f/f3_1_35.npy"))[:50].astype(np.float16)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_train = (X_train_branch, X_train_trunk)
    X_test = (X_test_branch, X_test_trunk)

    return X_train, y_train, X_test, y_test


def data_sta_train(task, path):
    num_dataset = 120
    y_train = []
    for i in range(1, num_dataset + 1, 1):
        y_train.append(np.load(os.path.join(path, f"velocity/model{i}.npy")))
    y_train = np.concatenate(y_train).astype(np.float16)
    y_train = minmax_normalize(y_train, 1500, 4500)

    y_test = np.load(os.path.join(path, f"velocity/model134.npy"))[:50].astype(np.float16)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_train_branch = []
    for i in range(1, num_dataset + 1, 1):
        X_train_branch.append(np.load(os.path.join(path, f"seismic/{task}/data{i}.npy")))
    X_train_branch = np.concatenate(X_train_branch).astype(np.float16)
    X_train_branch = log_transform(X_train_branch)
    X_train_branch = minmax_normalize(X_train_branch, log_transform(-30), log_transform(60))

    X_test_branch = np.load(os.path.join(path, f"seismic/{task}/data134.npy"))[:50].astype(np.float16)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_train_trunk = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk.append(np.load(os.path.join(path, f"{task}/{task}{i}.npy")))
        X_train_trunk = np.concatenate(X_train_trunk).astype(np.float16)
        X_train_trunk = log_transform(X_train_trunk)
        if task == 'loc':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(5), log_transform(25))
        X_test_trunk = np.load(os.path.join(path, f"{task}/{task}134.npy"))[:50].astype(np.float16)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_train_trunk1 = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk1.append(np.load(os.path.join(path, f"loc/loc{i}.npy")))
        X_train_trunk1 = np.concatenate(X_train_trunk1).astype(np.float16)
        X_train_trunk1 = log_transform(X_train_trunk1)
        X_train_trunk1 = minmax_normalize(X_train_trunk1, log_transform(0), log_transform(690))

        X_train_trunk2 = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk2.append(np.load(os.path.join(path, f"f/f{i}.npy")))
        X_train_trunk2 = np.concatenate(X_train_trunk2).astype(np.float16)
        X_train_trunk2 = log_transform(X_train_trunk2)
        X_train_trunk2 = minmax_normalize(X_train_trunk2, log_transform(5), log_transform(25))

        X_train_trunk = np.concatenate([X_train_trunk1, X_train_trunk2], axis=1)

        X_test_trunk1 = np.load(os.path.join(path, f"loc/loc134.npy"))[:50].astype(np.float16)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = np.load(os.path.join(path, f"f/f134.npy"))[:50].astype(np.float16)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_train = (X_train_branch, X_train_trunk)
    X_test = (X_test_branch, X_test_trunk)

    return X_train, y_train, X_test, y_test



def data_fvb_test(task, path):
    num_dataset = 12
    y_test = []
    for i in range(49, num_dataset + 49, 1):
        y_test.append(np.load(os.path.join(path, f"velocity/model{i}.npy")))
    y_test = np.concatenate(y_test).astype(np.float16)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_test_branch = []
    for i in range(49, num_dataset + 49, 1):
        X_test_branch.append(np.load(os.path.join(path, f"seismic/{task}/data{i}.npy")))
    X_test_branch = np.concatenate(X_test_branch).astype(np.float16)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_test_trunk = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk.append(np.load(os.path.join(path, f"{task}/{task}{i}.npy")))
        X_test_trunk = np.concatenate(X_test_trunk).astype(np.float16)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_test_trunk1 = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk1.append(np.load(os.path.join(path, f"loc/loc{i}.npy")))
        X_test_trunk1 = np.concatenate(X_test_trunk1).astype(np.float16)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk2.append(np.load(os.path.join(path, f"f/f{i}.npy")))
        X_test_trunk2 = np.concatenate(X_test_trunk2).astype(np.float16)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_test = (X_test_branch, X_test_trunk)

    return X_test, y_test


def data_cva_test(task, path):
    num_dataset = 12
    y_test = []
    for i in range(49, num_dataset + 49, 1):
        y_test.append(np.load(os.path.join(path, f"velocity/model{i}.npy")))
    y_test = np.concatenate(y_test).astype(np.float16)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_test_branch = []
    for i in range(49, num_dataset + 49, 1):
        X_test_branch.append(np.load(os.path.join(path, f"seismic/{task}/data{i}.npy")))
    X_test_branch = np.concatenate(X_test_branch).astype(np.float16)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_test_trunk = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk.append(np.load(os.path.join(path, f"{task}/{task}{i}.npy")))
        X_test_trunk = np.concatenate(X_test_trunk).astype(np.float16)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_test_trunk1 = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk1.append(np.load(os.path.join(path, f"loc/loc{i}.npy")))
        X_test_trunk1 = np.concatenate(X_test_trunk1).astype(np.float16)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk2.append(np.load(os.path.join(path, f"f/f{i}.npy")))
        X_test_trunk2 = np.concatenate(X_test_trunk2).astype(np.float16)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_test = (X_test_branch, X_test_trunk)

    return X_test, y_test


def data_cfa_test(task, path):
    num_dataset = 12
    y_test = []
    for i in [2, 3, 4]:
        for j in range(32, num_dataset//3 + 32, 1):
            y_test.append(np.load(os.path.join(path, f"velocity/vel{i}_1_{j}.npy")))
    y_test = np.concatenate(y_test).astype(np.float16)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_test_branch = []
    for i in [2, 3, 4]:
        for j in range(32, num_dataset//3 + 32, 1):
            X_test_branch.append(np.load(os.path.join(path, f"seismic/{task}/seis{i}_1_{j}.npy")))
    X_test_branch = np.concatenate(X_test_branch).astype(np.float16)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_test_trunk = []
        for i in [2, 3, 4]:
            for j in range(32, num_dataset // 3 + 32, 1):
                X_test_trunk.append(np.load(os.path.join(path, f"{task}/{task}{i}_1_{j}.npy")))
        X_test_trunk = np.concatenate(X_test_trunk).astype(np.float16)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_test_trunk1 = []
        for i in [2, 3, 4]:
            for j in range(32, num_dataset // 3 + 32, 1):
                X_test_trunk1.append(np.load(os.path.join(path, f"loc/loc{i}_1_{j}.npy")))
        X_test_trunk1 = np.concatenate(X_test_trunk1).astype(np.float16)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = []
        for i in [2, 3, 4]:
            for j in range(32, num_dataset // 3 + 32, 1):
                X_test_trunk2.append(np.load(os.path.join(path, f"f/f{i}_1_{j}.npy")))
        X_test_trunk2 = np.concatenate(X_test_trunk2).astype(np.float16)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_test = (X_test_branch, X_test_trunk)

    return X_test, y_test


def data_sta_test(task, path):
    num_dataset = 14
    y_test = []
    for i in range(121, num_dataset + 121, 1):
        y_test.append(np.load(os.path.join(path, f"velocity/model{i}.npy")))
    y_test = np.concatenate(y_test).astype(np.float16)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_test_branch = []
    for i in range(121, num_dataset + 121, 1):
        X_test_branch.append(np.load(os.path.join(path, f"seismic/{task}/data{i}.npy")))
    X_test_branch = np.concatenate(X_test_branch).astype(np.float16)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_test_trunk = []
        for i in range(121, num_dataset + 121, 1):
            X_test_trunk.append(np.load(os.path.join(path, f"{task}/{task}{i}.npy")))
        X_test_trunk = np.concatenate(X_test_trunk).astype(np.float16)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_test_trunk1 = []
        for i in range(121, num_dataset + 121, 1):
            X_test_trunk1.append(np.load(os.path.join(path, f"loc/loc{i}.npy")))
        X_test_trunk1 = np.concatenate(X_test_trunk1).astype(np.float16)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = []
        for i in range(121, num_dataset + 121, 1):
            X_test_trunk2.append(np.load(os.path.join(path, f"f/f{i}.npy")))
        X_test_trunk2 = np.concatenate(X_test_trunk2).astype(np.float16)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_test = (X_test_branch, X_test_trunk)

    return X_test, y_test