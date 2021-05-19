# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Mengqi.Ye on 2021/5/16
"""

import torch
from torch.nn import Linear, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from fake_datasets.fake_dataset import FakeCustomBatch, FakeLinearDataset
from fake_datasets.tests.test_fake_pipeline import print_array_as_csv
from net.hook import gradient_hook


class LinearNet(torch.nn.Module):
    def __init__(self):
        self.linear = torch.nn.Linear(in_features=100, out_features=1, bias=False)

    def forward(self, x):
        return self.linear.forward(x)


class Pipeline:
    def __init__(self, dataset: Dataset, net: torch.nn.Module):

        batch_size = 100
        dataset_size = 10000

        fake_dataset = dataset(in_features=100, dataset_size=dataset_size, device='cpu')
        one_tenth = fake_dataset.__len__() // 10

        tests_dataset, valid_dataset, train_dataset = torch.utils.data.dataset.random_split(
            fake_dataset, [one_tenth, one_tenth, 8 * one_tenth],
            # A fixed generator seed to reproduce training
            generator=torch.Generator().manual_seed(42)
        )

        collate_wrapper = lambda data: FakeCustomBatch(data)

        self.tests_loader = DataLoader(
            tests_dataset, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True
        )

        self.linear: Linear = net(in_features=100, out_features=1)
        self.loss = MSELoss()
        self.optimizer = Adam(self.linear.parameters(), lr=1e-3, eps=1e-3)

    def register_hook(self):
        self.linear.register_backward_hook(gradient_hook)

    def train(self):
        for epoch in range(1):
            print(f"epoch : {epoch}")
            print_array_as_csv(self.linear.weight.detach().cpu().numpy().reshape(10, 10))
            for batch_idx, sample_batch in enumerate(self.train_loader):
                sample_results = self.linear(sample_batch.inp)
                loss_v = self.loss(sample_results, sample_batch.tgt)
                self.optimizer.zero_grad()
                loss_v.backward()
                self.optimizer.step()

            for batch_idx, sample_batch in enumerate(self.valid_loader):
                sample_results = self.linear(sample_batch.inp)
                loss_v = self.loss(sample_results, sample_batch.tgt)
                # print(loss_v)

        print("Result")
        print_array_as_csv(self.linear.weight.detach().cpu().numpy().reshape(10, 10))


pipe = Pipeline(FakeLinearDataset, Linear)
pipe.register_hook()
pipe.train()
