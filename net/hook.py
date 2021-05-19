# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Mengqi.Ye on 2021/5/16
"""
from typing import Tuple
from torch.nn import Module
from fake_datasets.tests.test_fake_pipeline import print_array_as_csv


def gradient_hook(module: Module, grad_input: Tuple, grad_output: Tuple):
    print(module)

    print("grad_input", grad_input[0], grad_input[1])

    print("grad_output", grad_output[0].shape)
    print_array_as_csv(grad_output[0].reshape(10, 10).numpy())

