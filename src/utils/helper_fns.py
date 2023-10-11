import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

import src.settings as settings
from src.data.wcs_data import WCSDataset


def get_complexity(w_m, ib_model=None):
    if ib_model is not None:
        prior = ib_model.pM[:, 0]
    else:
        prior = np.ones(len(w_m)) / len(w_m)
    marginal = np.average(w_m, axis=0, weights=prior)
    complexities = []
    for likelihood in w_m:
        summed = 0
        for l, p in zip(likelihood, marginal):
            if l == 0:
                continue
            summed += l * (np.log(l) - np.log(p))
        complexities.append(summed)
    return np.average(complexities, weights=prior)  # Note that this is in nats (not bits)


def nat_to_bit(n):
    return n / np.log(2)


def get_color_data(ibm, bs, train_dim=0, eval_dim=0):
    raw_data = WCSDataset()
    speaker_obs = []
    train_f = []  # The feature we train on.
    task_f = []  # The actual task feature. Used for task distortion stuff
    probs = []

    use_rew = train_dim >= 3
    if use_rew:
        train_dim = train_dim % 3
    use_rew_eval = eval_dim <= 2
    for c1 in range(1, 331):
        features1 = raw_data.get_features(c1)
        s_obs = features1  # Speaker only sees the target
        speaker_obs.append(s_obs)
        t_f = features1[train_dim]
        if use_rew:
            t_f = _color_rew_fn(t_f)
        train_f.append(t_f)  # Train on red, for example. But we could change this to be blue!
        eval_val = _color_rew_fn(features1[eval_dim]) if use_rew_eval else features1[eval_dim % 3]
        task_f.append(eval_val)  # We always care about red as the actual task
        # Also track the prior probability of the color to specify the sampling.
        probs.append(ibm.pM[c1 - 1, 0])

    np_s = np.array(speaker_obs)
    np_train = np.array(train_f)
    np_train = np.resize(np_train, (330, 1))
    np_task = np.array(task_f)
    np_task = np.resize(np_task, (330, 1))
    dataset = TensorDataset(torch.Tensor(np_s).to(settings.device), torch.Tensor(np_train).to(settings.device), torch.Tensor(np_task).to(settings.device))
    sampler = WeightedRandomSampler(probs, num_samples=bs)
    dataloader = DataLoader(dataset, batch_size=bs, sampler=sampler)
    return raw_data, dataloader


def _color_rew_fn(val):
    if val < 0.125:
        return 0.5
    if val < 0.25:
        return -0.5
    if val < 0.375:
        return 0
    if val < 0.5:
        return 0.75
    if val < 0.625:
        return 1.0
    if val < 0.75:
        return -1.0
    return -0.75


def get_grid_data(bs, train_dim):
    grid = np.array([[-0.5, 0.0, 0.2, 0.0, 0.5],
                     [-0.5, 0.0, 0.0, 0.0, -0.5],
                     [-0.5, 0.0, 0.0, 0.0, 0.0],
                     [-0.5, 1.0, 0.0, 0.0, 0.0]])
    speaker_obs = []
    train_f = []  # The feature we train on.
    task_f = []  # The actual task feature. Used for task distortion stuff

    for y, row in enumerate(grid):
        for x, rew in enumerate(row):
            speaker_obs.append([x, y])
            if train_dim == 0:
                t_f = x / grid.shape[1]
            elif train_dim == 1:
                t_f = y / grid.shape[0]
            else:
                assert train_dim == 2, "Bad train dim"
                t_f = rew
            train_f.append(t_f)
            task_f.append(rew)
    np_s = np.array(speaker_obs)
    np_train = np.array(train_f)
    np_train = np.resize(np_train, (len(train_f), 1))
    np_task = np.array(task_f)
    np_task = np.resize(np_task, (len(task_f), 1))
    dataset = TensorDataset(torch.Tensor(np_s).to(settings.device), torch.Tensor(np_train).to(settings.device),
                            torch.Tensor(np_task).to(settings.device))
    dataloader = DataLoader(dataset, batch_size=bs)
    return grid, dataloader
