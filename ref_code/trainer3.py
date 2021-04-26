# -*- coding: utf-8 -*-
# Author：sen
# Date：2020/3/22 10:47

import torch
import numpy as np
from trainer.base_trainer import PLCBaseTrainer
import logging
from data_utils.big_dataset import batch_loader, plc_line_process

src_norm_max = np.array([288] * 12, dtype=np.float32)
src_norm_min = np.array([32] * 12, dtype=np.float32)

tar_norm_max = np.array([288] * 4, dtype=np.float32)
tar_norm_min = np.array([32] * 4, dtype=np.float32)

class PLCGeneralTrainer(PLCBaseTrainer):
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



class PLCNormGeneralTrainer(PLCBaseTrainer):
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
            print("loss", loss)

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
            # 预测系数的归一化
            src_norm_max_tensor, src_norm_min_tensor = torch.Tensor(src_norm_max).to(self.device), torch.Tensor(src_norm_min).to(self.device)
            srcdata = (srcdata - src_norm_min_tensor) / (src_norm_max_tensor - src_norm_min_tensor) # 将数据归一化到[0.0, 1.0]

            assert srcdata.dim() == 3
            predicted = self.model(srcdata)

            tar_norm_max_tensor, tar_norm_min_tensor = torch.Tensor(tar_norm_max).to(self.device), torch.Tensor(tar_norm_min).to(self.device)
            predicted = predicted * (tar_norm_max_tensor - tar_norm_min_tensor)  + tar_norm_min_tensor
            srcdata_value = srcdata.detach().cpu().numpy().reshape(-1)[:]
            groundtruth_value = groundtruth.detach().cpu().numpy().reshape(-1)[:]
            predicted_value = predicted.detach().cpu().numpy().reshape(-1)[:]
            # print(predicted_value, end="*******");print(groundtruth_value)
            single_data_err_score = [abs(groundtruth_value[i] - predicted_value[i]) for i in range(len(groundtruth_value))]
            mean_err_score.append(np.mean(single_data_err_score))
        # 返回平均指标
        return np.mean(mean_err_score)


class PLCGeneralTrainerRNN2(PLCBaseTrainer):
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

            srcdata = torch.tensor(srcdata, dtype=torch.float32)
            srcdata = srcdata.to(self.device)  # [batch_size, 1, sample_length] eg.[600,1,20]
            groundtruth = groundtruth.to(self.device)  # [batchsize,1,4]
            groundtruth = groundtruth.view(batch_size, groundtruth.size(2)) # tensor.size([batchsize, 4])
            groundtruth_trend = torch.zeros(batch_size, 3)
            for i in range(batch_size):
                for j in range(1,4):
                    groundtruth_trend[i][j-1] = groundtruth[i][j] - groundtruth[i][j-1]

            predicted = self.model(srcdata)  # [batchsize, 4]

            predicted_trend = torch.zeros(batch_size, 3)
            for i in range(batch_size):
                for j in range(1, 4):
                    predicted_trend[i][j-1] = predicted[i][j] - predicted[i][j-1]

            # TODO 目前rnn训练到最后阶段的时候会因为batchsize的原因报错，目前这个bug暂时没有修理
            print(predicted.shape)
            # predicted_trend =
            predicted_int = predicted.detach().cpu().numpy().astype(int)
            #
            # predicted_recover = (predicted.detach().cpu().numpy() * 257. + 32.).astype(int)
            # groundtruth_recover = (groundtruth.detach().cpu().numpy() * 257. + 32.)

            loss_mean = []
            # print(srcdata.shape)
            # print(groundtruth.shape)
            # print(predicted.shape)

            """================ Optimize model ================"""
            self.optimizer.zero_grad()
            loss_pre = self.loss_function(predicted, groundtruth)
            loss_trend = self.loss_function(predicted_trend, groundtruth_trend)
            loss = loss_pre + loss_trend
            loss.backward()
            self.optimizer.step()
            loss_mean.append(loss.data.cpu().numpy())
            # print(loss)
            print(loss,'------', loss_pre, '--------', loss_trend, '-------')
            print(predicted_int)
            print( groundtruth)
            # srcdata_value = srcdata.detach().cpu().numpy().reshape(-1)[:]
            # print("loss", loss.detach().cpu().numpy(), end="   ")
            # print("predict:", predicted.detach().cpu().numpy().reshape(-1)[:], end="   ")
            # print("groundtruth:", groundtruth.detach().cpu().numpy().reshape(-1)[:])

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
            # 预测系数的归一化
            src_norm_max_tensor, src_norm_min_tensor = torch.Tensor(src_norm_max).to(self.device), torch.Tensor(src_norm_min).to(self.device)
            srcdata = (srcdata - src_norm_min_tensor) / (src_norm_max_tensor - src_norm_min_tensor) # 将数据归一化到[0.0, 1.0]

            assert srcdata.dim() == 3
            predicted = self.model(srcdata)

            tar_norm_max_tensor, tar_norm_min_tensor = torch.Tensor(tar_norm_max).to(self.device), torch.Tensor(tar_norm_min).to(self.device)
            predicted = predicted * (tar_norm_max_tensor - tar_norm_min_tensor)  + tar_norm_min_tensor
            srcdata_value = srcdata.detach().cpu().numpy().reshape(-1)[:]
            groundtruth_value = groundtruth.detach().cpu().numpy().reshape(-1)[:]
            predicted_value = predicted.detach().cpu().numpy().reshape(-1)[:]
            # print(predicted_value, end="*******");print(groundtruth_value)
            single_data_err_score = [abs(groundtruth_value[i] - predicted_value[i]) for i in range(len(groundtruth_value))]
            mean_err_score.append(np.mean(single_data_err_score))
        # 返回平均指标
        return np.mean(mean_err_score)




class PLCBigGeneralTrainer(PLCBaseTrainer):

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
        for src_data, groundtruth in batch_loader(self.train_dataloader, line_process_func=plc_line_process, batch_size=5000,verbose_total_cnt=False):
            src_data = torch.FloatTensor(src_data).to(self.device) # torch.size[batchsize, sample_length] eg.[400, 16]
            groundtruth = torch.FloatTensor(groundtruth).to(self.device) # torch.size[batchsize, 16]
            src_data = src_data.view(src_data.size(0), 1, src_data.size(1)) # torch.size[batchsize, 1, sample_length]
            groundtruth = groundtruth.view(groundtruth.size(0), 1, groundtruth.size(1)) # torch.size [batchsize, 1, sample_length]
            predicted = self.model(src_data) # torch.size[batchsize, 16]
            loss_mean = []
            # print(predicted.shape)

            """================ Optimize model ================"""
            self.optimizer.zero_grad()
            loss = self.loss_function(predicted, groundtruth)
            loss.backward()
            self.optimizer.step()
            loss_mean.append(loss.data.cpu().numpy())

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
            single_data_err_score = [abs(groundtruth_value[i] - predicted_value[i]) for i in range(len(groundtruth_value))]
            mean_err_score.append(np.mean(single_data_err_score))
        # 返回平均指标
        return np.mean(mean_err_score)

    # def _validation_epoch(self, epoch):
    #     sample_length = self.validation_custom_config["sample_length"]
    #     mean_err_score = [] # predicted与groundtruth之间的差异
    #
    #     batch_err_score = []
    #     with torch.no_grad():
    #         self.model.eval()
    #         for src_data, groundtruth in batch_loader(self.train_dataloader, line_process_func=plc_line_process, batch_size=10,verbose_total_cnt=False):
    #             src_data = torch.FloatTensor(src_data).to(self.device)
    #             groundtruth = torch.FloatTensor(groundtruth).to(self.device)
    #             src_data = src_data.view(src_data.size(0), 1,src_data.size(1))  # torch.size[batchsize, 1, sample_length]
    #             groundtruth = groundtruth.view(groundtruth.size(0), 1,groundtruth.size(1))  # torch.size [batchsize, 1, sample_length]
    #             predicted = self.model(src_data) # [batchsize, sample_length]
    #             c = torch.sum(torch.abs(predicted - groundtruth)).cpu().item()
    #             batch_err_score.append(c / (groundtruth.shape[0] * groundtruth.shape[1]))
    #     return sum(batch_err_score) / len(batch_err_score)


class PLCBigGeneralTrainerCNN(PLCBaseTrainer):
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
        for src_data, groundtruth in batch_loader(self.train_dataloader, line_process_func=plc_line_process, batch_size=8000,verbose_total_cnt=False):
            src_data = torch.FloatTensor(src_data).to(self.device) # torch.size[batchsize, sample_length] eg.[400, 266]
            groundtruth = torch.FloatTensor(groundtruth).to(self.device) # torch.size[batchsize, 266]
            src_data = src_data.view(src_data.size(0), 1, src_data.size(1)) # torch.size[batchsize, 1, 266]
            groundtruth = groundtruth.view(groundtruth.size(0), 1, groundtruth.size(1)) # torch.size [batchsize, 1, sample_length]

            # print(src_data)
            # 将数据变成2d的,然后通过cnn增强后变回原来的样子
            segments = int(src_data.shape[2] / 38)  # 将7帧1维的magnitude重新整理为7帧(每帧为38个数据点) segments = 7
            batch = src_data.shape[0] # batch_size
            src_data2d = torch.zeros(batch, 1, segments, 38) # torch.size([batch_size, 1, 7, 38])
            for i in range(segments):
                src_data2d[:,:, i,:] = src_data[:,:,i * 38: i *38 + 38]
            src_data2d_1col = src_data2d[:,:,:,0].view(src_data2d.size(0), src_data2d.size(1), src_data2d.size(2), 1).to(self.device)  # 重组数据的第一个点为能量，不能算作magnitude的一部分 [batch_size, 1, 7, 1]
            src_data2d_mag = src_data2d[:,:,:,1:].to(self.device) # magnitude部分  [batch_size, 1, 7, 37]
            predicted = self.model(src_data2d_mag) # torch.size[batch_size, 1, 7, 37]
            predicted_recover = torch.cat((src_data2d_1col,predicted), dim=3) # torch.size[batch_size, 1, 7, 38]
            predicted_recover_38 = torch.flatten(predicted_recover, start_dim=2) # torch.size[batch_size, 1, 266]
            loss_mean = []
            # print(predicted.shape)

            """================ Optimize model ================"""
            self.optimizer.zero_grad()
            loss = self.loss_function(predicted_recover_38, groundtruth)
            loss.backward()
            self.optimizer.step()
            loss_mean.append(loss.data.cpu().numpy())

        logging.info("epoch / mean loss: {} / {:.4f}".format(epoch, np.mean(loss_mean)))

    def _validation_epoch(self, epoch):
        sample_length = self.validation_custom_config["sample_length"]
        mean_err_score = [] # predicted与groundtruth之间的差异

        for i, (src_data, groundtruth) in enumerate(self.validation_dataloader, start=1):
            # srcdata.shape = torch.Size([1, 1, 266]), groundtruth = torch.Size([1,1,266])
            src_data, groundtruth = src_data.to(self.device), groundtruth.to(self.device)
            assert src_data.dim() == 3

            # 使用模型预测，模型只能处理二维部分
            segments = int(src_data.shape[2] / 38)  # 将7帧1维的magnitude重新整理为7帧(每帧为38个数据点) segments = 7
            batch = src_data.shape[0]  # batch_size
            src_data2d = torch.zeros(batch, 1, segments, 38)  # torch.size([batch_size, 1, 7, 38])
            for i in range(segments):
                src_data2d[:, :, i, :] = src_data[:, :, i * 38: i * 38 + 38]
            src_data2d_1col = src_data2d[:, :, :, 0].view(src_data2d.size(0), src_data2d.size(1), src_data2d.size(2),1).to(self.device)  # 重组数据的第一个点为能量，不能算作magnitude的一部分 [batch_size, 1, 7, 1]
            src_data2d_mag = src_data2d[:, :, :, 1:].to(self.device)  # magnitude部分  [batch_size, 1, 7, 37]
            predicted = self.model(src_data2d_mag)  # torch.size[batch_size, 1, 7, 37]
            predicted_recover = torch.cat((src_data2d_1col, predicted), dim=3)  # torch.size[batch_size, 1, 7, 38]
            predicted_recover_38 = torch.flatten(predicted_recover, start_dim=2)  # torch.size[batch_size, 1, 266]

            # srcdata_value = src_data.detach().cpu().numpy().reshape(-1)[:]
            groundtruth_value = groundtruth.detach().cpu().numpy().reshape(-1)[:]
            predicted_value = predicted_recover_38.detach().cpu().numpy().reshape(-1)[:]
            single_data_err_score = [abs(groundtruth_value[i] - predicted_value[i]) for i in range(len(groundtruth_value))]
            mean_err_score.append(np.mean(single_data_err_score))

        # 返回平均指标
        return np.mean(mean_err_score)
