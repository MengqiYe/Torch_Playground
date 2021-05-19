# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Mengqi.Ye on 2020/12/21
"""


class WhateverName:
    r"""
    Let this be a thread - safe object.
    """

    def __init__(self):
        self.shared_env = None

        self.asynchronous_env_list = []

    def init_hyper_parameters(self):
        pass

    def pick_action(self):
        pass

    def observe(self):
        pass

    def apply_action(self):
        pass

    def estimate_reward(self):
        r"""
        .. math::
        Q^{\pi}(a_t, s_t)
        """
        pass

    def estimate_value(self):
        r"""
        .. math::
        V^{\pi}(s_t)
        """
        pass
