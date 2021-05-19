# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Mengqi.Ye on 2021/5/16
"""
from typing import List

import numpy as np


def print_array_as_csv(array: np.ndarray, used_columns: List = None):
    for row in array:
        print(','.join(['%16s' % str(f) for f in row]))

