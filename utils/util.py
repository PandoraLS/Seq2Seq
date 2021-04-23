# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-3-7 下午4:44
"""
工具代码
"""
import time
import datetime
import importlib

class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()

        <Something...>

        print(f"Finished in {timer.duration()} seconds.")
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return time.time() - self.start_time

class ExecutionTime2:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print("Execution time: ", timer.duratioin())
    """

    def __init__(self):
        self.start_time = datetime.datetime.now()

    def duration(self):
        end_time = datetime.datetime.now()
        return str(datetime.timedelta(seconds=(end_time - self.start_time).seconds))


def initialize_config(module_cfg):
    """
    根据配置项，动态加载对应的模块， 并将参数传入模块内部的制定函数
    eg. 配置文件如下:
        module_cfg = {
            "module": "models.unet", 
            "main": "UNet",
            "args": {...}
        }
    1. 加载 type 参数对应的模块
    2. 调用(实例化)模块内部对应 main 参数的函数(类)
    3. 再调用(实例化)时将 args 参数输入函数(类)
    :param module_cfg: 配置信息， 参见json文件
    :return: 实例化后的函数(类)
    """
    module = importlib.import_module(module_cfg["module"])
    return getattr(module, module_cfg["main"])(**module_cfg["args"])


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.
    :param dirs (list): directors list 
    :param resume (bool):  是否继续试验，默认是False
    :return: 
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


def set_requires_grad(nets, requires_grad=False):
    """
    :param nets: list of networks
    :param requires_grad: 
    :return: 
    """
    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


if __name__ == '__main__':
    pass
    # timer = ExecutionTime()
    # time.sleep(1)
    # print("Execution time: ", timer.duration())

    cfg = {
        "module": "models.unet",
        "main": "UNet",
        "args": {...}
    }
    
    