# -*- coding: utf-8 -*-
# Author：sen
# Date：2021/4/21 12:30

"""
通过tensorboard记录训练模型和日志
"""
from torch.utils.tensorboard import SummaryWriter

def writer(logs_dir):
    return SummaryWriter(log_dir=logs_dir, max_queue=5, flush_secs=30)
