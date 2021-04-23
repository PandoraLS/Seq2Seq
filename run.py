# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 2:00 下午
# @Author  : sen
import util.global_config
import logging


import argparse
import os
import json5
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.util import initialize_config
from trainer.trainer import PLCGeneralTrainerRNN2
from trainer.tester import PLC_pitchL_Predicter
from util.pitchL_data_prep import pitchL_to_predicted_20200630, pitchL_to_array

def config():
    parser = argparse.ArgumentParser(description="Packet Loss Concealment")
    parser.add_argument("-C", "--configuration", default='config/plc/plc_pitch_0712.json5', type=str, help="Configuration (*.json).")
    parser.add_argument("-R", "--resume", default=False, type=bool, help="Resume experiment from latest checkpoint.")
    parser.add_argument("-T", "--test", default=False, type=bool, help="Test from the best breakpoint.")
    parser.add_argument("--visual", default=True, type=bool, help="Whether to visualize with tensorboard, it is best to set false when running on GPU")
    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration
    return configuration, args.resume, args.test, args.visual

def main(config, resume, test_flag, visual_flag):
    """
    获取config,resume,test_flag参数
    Args:
        config: 配置文件json格式
        resume(bool): 是否继续训练
        test_flag(bool): 是否仅测试
        visual_flag(bool): 是否开启tensorboard相关代码
    Returns:
    """
    torch.manual_seed(int(config["seed"]))  # both CPU and CUDA
    np.random.seed(config["seed"])

    train_dataloader = DataLoader(
        dataset=initialize_config(config["train_dataset"]),
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"],
        pin_memory=config["train_dataloader"]["pin_memory"]  # Very small data set False
    )

    validation_dataloader = DataLoader(
        dataset=initialize_config(config["validation_dataset"]),
        batch_size=1,
        num_workers=1
    )

    test_dataloader = DataLoader(
        dataset=initialize_config(config["test_dataset"]),
        batch_size=1,
        num_workers=1
    )

    model = initialize_config(config["model"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = initialize_config(config["loss_function"])

    trainer = PLCGeneralTrainerRNN2(
        config=config,
        resume=resume,
        model=model,
        optim=optimizer,
        loss_fucntion=loss_function,
        train_dl=train_dataloader,
        validation_dl=validation_dataloader,
        visual=visual_flag
    )
    if test_flag:
        tester = PLC_pitchL_Predicter(
            config=config,
            resume=resume,
            test=True,
            model=model,
            optim=optimizer,
            loss_function=loss_function,
            test_dl=test_dataloader,
            visual=visual_flag
        )
        tester.test()
        pitchL_to_predicted_20200630()
        pitchL_to_array()
    else:
        trainer.train()



if __name__ == '__main__':
    util.global_config.config_log()

    logging.info("plc 实验程序 ----------------------------------------------------------")
    configuration, resume_flag, test_flag, visual_flag = config()
    main(configuration, resume=resume_flag, test_flag=test_flag, visual_flag=visual_flag)

