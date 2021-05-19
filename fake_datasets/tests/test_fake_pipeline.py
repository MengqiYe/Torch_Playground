# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Mengqi.Ye on 2020/12/5
"""
from typing import List

import torch
import numpy as np
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from fake_datasets.fake_dataset import FakeLinearDataset, FakeCustomBatch


def print_array_as_csv(array: np.ndarray, used_columns: List = None):
    assert type(array) == np.ndarray, f"Wrong type!"
    for row in array:
        print(','.join(['%8s' % ('%1.3f' % f) for f in row]))


class TestFake:
    def setup(self):
        batch_size = 100
        dataset_size = 10000

        self.fake_dataset = FakeLinearDataset(in_features=100, dataset_size=dataset_size, device='cuda')
        one_tenth = self.fake_dataset.__len__() // 10

        tests_dataset, valid_dataset, train_dataset = torch.utils.data.dataset.random_split(
            self.fake_dataset, [one_tenth, one_tenth, 8 * one_tenth],
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

    def test_basic_iteration(self):
        for batch_idx, sample_batch in enumerate(self.train_loader):
            print(
                f"batch_idx : {batch_idx}, \n"
                f"sample_batch type : {type(sample_batch)}, \n"
                f"input shape : {sample_batch.inp.shape}, device : {sample_batch.inp.device}, \n"
                f"target shape : {sample_batch.tgt.shape}, device : {sample_batch.tgt.device}, \n"
            )

    def test_basic_linear_model(self):
        linear: Linear = Linear(in_features=100, out_features=1).cuda('cuda')
        loss = MSELoss()
        optimizer = Adam(linear.parameters(), lr=1e-3, eps=1e-3)

        for epoch in range(10):
            print(f"epoch : {epoch}")
            print_array_as_csv(linear.weight.detach().cpu().numpy().reshape(10, 10))
            for batch_idx, sample_batch in enumerate(self.train_loader):
                sample_results = linear(sample_batch.inp)
                loss_v = loss(sample_results, sample_batch.tgt)
                optimizer.zero_grad()
                loss_v.backward()
                optimizer.step()

            for batch_idx, sample_batch in enumerate(self.valid_loader):
                sample_results = linear(sample_batch.inp)
                loss_v = loss(sample_results, sample_batch.tgt)
                # print(loss_v)

        print("Result")
        print_array_as_csv(linear.weight.detach().cpu().numpy().reshape(10, 10))


    def test_a9c(self):
        pass


def test_fake():
    batch_size = 100
    dataset_size = 10000

    fake_dataset = FakeLinearDataset(in_features=100, dataset_size=10000, device='cuda')
    one_tenth = fake_dataset.__len__() // 10

    tests_dataset, valid_dataset, train_dataset = torch.utils.data.dataset.random_split(
        fake_dataset, [one_tenth, one_tenth, fake_dataset.__len__() - 2 * one_tenth],
        # A fixed generator seed to reproduce training
        generator=torch.Generator().manual_seed(42)
    )

    collate_wrapper = lambda data: FakeCustomBatch(data)

    tests_loader = DataLoader(tests_dataset, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_wrapper, pin_memory=True)

    for batch_idx, sample_batch in enumerate(train_loader):
        print(
            f"batch_idx : {batch_idx}, \n"
            f"sample_batch type : {type(sample_batch)}, \n"
            f"input shape : {sample_batch.inp.shape}, device : {sample_batch.inp.device}, \n"
            f"target shape : {sample_batch.tgt.shape}, device : {sample_batch.tgt.device}, \n"
        )
