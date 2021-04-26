# -*- coding: utf-8 -*-
# @Time    : 2021/4/26 2:00 下午
# @Author  : sen

import os
import json5
import torch
import argparse
import numpy as np
from utils.util import initialize_config
from trainer.trainer import Trainer
from model.seq2seq import EncoderGRU, DecoderGRU

def config():
    parser = argparse.ArgumentParser(description="seq2seq for machine translation")
    parser.add_argument("-C", "--configuration", default='config/seq2seq.json5', type=str, help="Configuration (*.json).")
    parser.add_argument("-R", "--resume", default=False, type=bool, help="Resume experiment from latest checkpoint.")
    parser.add_argument("-I", "--inference", default=False, type=bool, help="inference from the last breakpoint.")
    parser.add_argument("--visual", default=True, type=bool, help="Whether to visualize with tensorboard, it is best to set false when running on GPU")
    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration
    return configuration, args.resume, args.inference, args.visual

def main(config, resume, visual):
    """
    获取config, resume, visual_flag参数进行初始化
    Args:
        config: 配置文件json格式
        resume(bool): 是否继续训练
        visual(bool): 是否开启tensorboard相关代码
    Returns:
    """
    torch.manual_seed(int(config["seed"]))  # both CPU and CUDA
    np.random.seed(config["seed"])

    word2indexs = initialize_config(config["word2index"])
    dataset = initialize_config(config["train_dataset"])
    
    encoder = EncoderGRU(input_size=word2indexs.input_lang.n_words, hidden_size=256)
    decoder = DecoderGRU(hidden_size=256, output_size=word2indexs.output_lang.n_words)

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=config["optimizer"]["enc_lr"])
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=config["optimizer"]["dec_lr"])

    loss_function = initialize_config(config["loss_function"])

    trainer = Trainer(
        config=config,
        resume=resume,
        encoder=encoder,
        decoder=decoder,
        optim_enc=encoder_optimizer,
        optim_dec=decoder_optimizer,
        loss_fucntion=loss_function,
        visual=visual,
        dataset=dataset,
        word2indexs=word2indexs,
        sentence_max_length=10  # 数据集中sentence的max_length
    )
    trainer.train()



if __name__ == '__main__':
    configuration, resume, inference, visual = config()
    main(configuration, resume=resume, visual=visual)
