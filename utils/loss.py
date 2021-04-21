# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-3-8 下午2:27

"""
各种loss函数，包括自定义loss函数
"""
import torch

def mse_loss():
    return torch.nn.MSELoss()
