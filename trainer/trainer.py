# -*- coding: utf-8 -*-
# @Time : 2021/4/22 下午12:37

import torch
import logging
import numpy as np
from trainer.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self,
                 config: dict,
                 resume: bool,
                 model,
                 optim,
                 loss_fucntion,
                 train_dl,
                 validation_dl,
                 visual):
        super().__init__(config, resume, model, optim, loss_fucntion, visual)
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl
        self.visual = visual

    def _visualize_weights_and_grads(self, model, epoch):
        if self.visual:
            for name, param in model.named_parameters():
                self.writer.add_histogram("WEIGHT_" + name, param.clone().cpu().data.numpy(), epoch)
                self.writer.add_histogram("GRAD_" + name, param.grad.cpu().numpy(), epoch)

    def _train_epoch(self, epoch):
        for i, (srcdata, groundtruth) in enumerate(self.train_dataloader, start=1):
            # For visualization
            batch_size = self.train_dataloader.batch_size
            n_batch = len(self.train_dataloader)
            n_iter = n_batch * batch_size * (epoch - 1) + i * batch_size

            srcdata = srcdata.to(self.device)  # [batch_size, 1, sample_length] eg.[600,1,20]
            groundtruth = groundtruth.to(self.device)  # [600,1,20]
            predicted = self.model(srcdata)  # [600, 1, 20]
            loss_mean = []

            """================ Optimize model ================"""
            self.optimizer.zero_grad()
            loss = self.loss_function(predicted, groundtruth)
            loss.backward()
            self.optimizer.step()
            loss_mean.append(loss.data.cpu().numpy())
            # print("loss", loss)

            with torch.no_grad():
                predicted = self.model(srcdata)
                if self.visual:
                    self.writer.add_scalars(f"模型/损失值_n_iter", {
                        "模型优化前": loss,
                        "模型优化后": self.loss_function(predicted, groundtruth)
                    }, n_iter)
        logging.info("epoch / mean loss: {} / {:.4f}".format(epoch, np.mean(loss_mean)))


    def _validation_epoch(self, epoch):
        sample_length = self.validation_custom_config["sample_length"]
        mean_err_score = [] # predicted与groundtruth之间的差异

        for i, (srcdata, groundtruth) in enumerate(self.validation_dataloader, start=1):
            srcdata = srcdata.to(self.device)
            # 预测系数的这些数据没有必要归一化,归一化之后就预测不准了
            # norm_max = torch.max(srcdata).item()
            # norm_min = torch.min(srcdata).item()
            # srcdata = 2 * (srcdata - norm_min) / (norm_max - norm_min) - 1  # 将音频数据归一化到[-1.0,1.0]

            assert srcdata.dim() == 3
            predicted = self.model(srcdata)
            srcdata_value = srcdata.detach().cpu().numpy().reshape(-1)[:]
            groundtruth_value = groundtruth.detach().cpu().numpy().reshape(-1)[:]
            predicted_value = predicted.detach().cpu().numpy().reshape(-1)[:]
            # print(predicted_value, end="*******");print(groundtruth_value)
            single_data_err_score = [abs(groundtruth_value[i] - predicted_value[i]) for i in range(len(groundtruth_value))]
            mean_err_score.append(np.mean(single_data_err_score))
        # 返回平均指标
        return np.mean(mean_err_score)

