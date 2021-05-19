# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Mengqi.Ye on 2020/12/5
"""
import torch
from torch.utils.data import Dataset, DataLoader


class FakeLinearDataset(Dataset):
    def __init__(self, in_features=100, dataset_size=10000, device='cpu'):
        self.data = torch.randn((dataset_size, in_features,), device=device)
        self.target = torch.mean(self.data, dim=1)
        # print(type(self.target.shape), self.target.shape)
        # assert self.data.shape == torch.Size((dataset_size, in_features,)), f"Real data shape : {self.data.shape}"
        # assert self.target.shape == torch.Size((dataset_size,)), f"Real target shape : {self.target.shape}"
        # assert self.data.device == torch.device('cuda', 0), f"Real data device : {self.data.device}"
        # assert self.target.device == torch.device('cuda', 0), f"Real target device : {self.target.device}"

    def __getitem__(self, item):
        if isinstance(item, slice):
            print(type(item), item)
            return self.data[item], self.target[item]

        else:
            idx = item % self.__len__()
            return self.data[idx], self.target[idx]

    def __len__(self):
        return len(self.data)


class FakeCustomBatch:
    def __init__(self, data):
        # FIXME : This is at one iteration cost.
        self.inp = torch.vstack([d[0] for d in data])
        self.tgt = torch.vstack([d[1] for d in data])

    def pin_memory(self):
        cpu_device = torch.device('cpu')
        if self.inp.device == cpu_device:
            self.inp = self.inp.pin_memory()
        if self.tgt.device == cpu_device:
            self.tgt = self.tgt.pin_memory()
        return self

