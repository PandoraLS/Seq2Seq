# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-1-5 下午10:36

import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from trainer.base_trainer import PLCBaseTrainer


plt.switch_backend("agg")

class PLCTester(PLCBaseTrainer):
    def __init__(self,
                 config: dict,
                 resume: bool,
                 test: bool,
                 model,
                 optim,
                 loss_function,
                 test_dl,
                 visual):
        super().__init__(config, resume, model, optim, loss_function, visual)
        self.test_dataloader = test_dl
        if test: self._test_checkpoint()

    def _test_checkpoint(self):
        """
        test experiment from latest checkpoint.
        Notes: To be careful at Loading model.
                if model is an instance of DataParallel, we need to set model.module.*
        :return: None
        """
        best_model_path = self.checkpoints_dir / "best_model.tar"
        assert best_model_path.exists(), f"{best_model_path} does not exist, can't load latest checkpoint."

        checkpoint = torch.load(best_model_path.as_posix(), map_location=self.device)
        self.start_epoch = checkpoint["epoch"]

        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        logging.info(f"Model checkpoint loaded. Testing in {self.start_epoch} epoch.")

    def _print_list(self, value_list):
        print("{",end="")
        for i in range(len(value_list)):
            if i == len(value_list) - 1:
                print("%d" % int(value_list[i]), end="")
            else:
                print("%d" % int(value_list[i]), end=",")
        print("},")

    def _test(self):
        mean_err_score = []  # predicted与groundtruth之间的差异的均值
        for i, (srcdata, groundtruth) in enumerate(self.test_dataloader, start=1):
            srcdata = srcdata.to(self.device)
            assert srcdata.dim() == 3
            predicted = self.model(srcdata)
            srcdata_value = srcdata.detach().cpu().numpy().reshape(-1)[:]
            groundtruth_value = groundtruth.detach().cpu().numpy().reshape(-1)[:]
            predicted_value = predicted.detach().cpu().numpy().reshape(-1)[:]
            single_data_err_score = [abs(groundtruth_value[i] - predicted_value[i]) for i in range(len(groundtruth_value))]
            # print("srcdata: {:.4f}, groundtruth: {:.4f}, predicted: {:.4f}".format(srcdata_value, groundtruth_value,
            #                                                                        predicted_value))
            # print("srcdata_value:");self._print_list(srcdata_value)
            # print("groundtruth_value:");self._print_list(groundtruth_value)
            # print("predicted_value:");self._print_list(predicted_value)
            # print("------------------------------------------")
            self._print_list(predicted_value)
            mean_err_score.append(np.mean(single_data_err_score))
        # 返回平均指标
        return np.mean(mean_err_score)
    
class PLCPredicter(PLCBaseTrainer):
    def __init__(self,
                 config: dict,
                 resume: bool,
                 test: bool,
                 model,
                 optim,
                 loss_function,
                 test_dl,
                 visual):
        super().__init__(config, resume, model, optim, loss_function, visual)
        self.test_dataloader = test_dl
        if test: self._test_checkpoint()

    def _test_checkpoint(self):
        """
        test experiment from latest checkpoint.
        Notes: To be careful at Loading model.
                if model is an instance of DataParallel, we need to set model.module.*
        :return: None
        """
        best_model_path = self.checkpoints_dir / "best_model.tar"
        assert best_model_path.exists(), f"{best_model_path} does not exist, can't load latest checkpoint."

        checkpoint = torch.load(best_model_path.as_posix(), map_location=self.device)
        self.start_epoch = checkpoint["epoch"]

        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        logging.info(f"Model checkpoint loaded. Testing in {self.start_epoch} epoch.")

    def _print_list(self, value_list):
        print("{",end="")
        for i in range(len(value_list)):
            if i == len(value_list) - 1:
                print("%d" % int(value_list[i]), end="")
            else:
                print("%d" % int(value_list[i]), end=",")
        print("},")

    def _test(self):
        predicted_list = []
        for i, srcdata in enumerate(self.test_dataloader, start=1):
            srcdata = srcdata.to(self.device)
            assert srcdata.dim() == 3
            predicted = self.model(srcdata)
            predicted_value = predicted.detach().cpu().numpy().reshape(-1)[:]
            # srcdata_value = srcdata.detach().cpu().numpy().reshape(-1)[:]
            predicted_list.append(predicted_value)
        # 返回预测结果的列表
        return predicted_list

class PLC_pitchL_Tester(PLCBaseTrainer):
    def __init__(self,
                 config: dict,
                 resume: bool,
                 test: bool,
                 model,
                 optim,
                 loss_function,
                 test_dl,
                 visual):
        super().__init__(config, resume, model, optim, loss_function, visual)
        self.test_dataloader = test_dl
        if test: self._test_checkpoint()

    def _test_checkpoint(self):
        """
        test experiment from latest checkpoint.
        Notes: To be careful at Loading model.
                if model is an instance of DataParallel, we need to set model.module.*
        :return: None
        """
        best_model_path = self.checkpoints_dir / "best_model.tar"
        assert best_model_path.exists(), f"{best_model_path} does not exist, can't load latest checkpoint."

        checkpoint = torch.load(best_model_path.as_posix(), map_location=self.device)
        self.start_epoch = checkpoint["epoch"]

        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        logging.info(f"Model checkpoint loaded. Testing in {self.start_epoch} epoch.")

    def _print_list(self, value_list):
        print("{",end="")
        for i in range(len(value_list)):
            if i == len(value_list) - 1:
                print("%d" % int(value_list[i]), end="")
            else:
                print("%d" % int(value_list[i]), end=",")
        print("},")

    def _test(self):
        predicted_list = []
        mean_err_score = [] # predicted与groundtruth之间的差异
        for i, (srcdata, groundtruth) in enumerate(self.test_dataloader, start=1):
            srcdata = srcdata.to(self.device)
            assert srcdata.dim() == 3
            predicted = self.model(srcdata)
            predicted_value = predicted.detach().cpu().numpy().reshape(-1)[:]
            groundtruth_value = groundtruth.detach().cpu().numpy().reshape(-1)[:]
            single_data_err_score = [abs(groundtruth_value[i] - predicted_value[i]) for i in range(len(groundtruth_value))]
            mean_err_score.append(np.mean(single_data_err_score))
            # srcdata_value = srcdata.detach().cpu().numpy().reshape(-1)[:]
            predicted_list.append(predicted_value)
        logging.info("score: {:.4f}".format(np.mean(mean_err_score)))
        # 返回预测结果的列表
        return predicted_list

class PLC_pitchL_Predicter(PLCBaseTrainer):
    def __init__(self,
                 config: dict,
                 resume: bool,
                 test: bool,
                 model,
                 optim,
                 loss_function,
                 test_dl,
                 visual):
        super().__init__(config, resume, model, optim, loss_function, visual)
        self.test_dataloader = test_dl
        if test: self._test_checkpoint()

    def _test_checkpoint(self):
        """
        test experiment from latest checkpoint.
        Notes: To be careful at Loading model.
                if model is an instance of DataParallel, we need to set model.module.*
        :return: None
        """
        best_model_path = self.checkpoints_dir / "best_model.tar"
        assert best_model_path.exists(), f"{best_model_path} does not exist, can't load latest checkpoint."

        checkpoint = torch.load(best_model_path.as_posix(), map_location=self.device)
        self.start_epoch = checkpoint["epoch"]

        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        logging.info(f"Model checkpoint loaded. Testing in {self.start_epoch} epoch.")

    def _print_list(self, value_list):
        print("{",end="")
        for i in range(len(value_list)):
            if i == len(value_list) - 1:
                print("%d" % int(value_list[i]), end="")
            else:
                print("%d" % int(value_list[i]), end=",")
        print("},")

    def _test(self):
        predicted_list = []
        for i, srcdata in enumerate(self.test_dataloader, start=1):
            srcdata = srcdata.to(self.device)
            assert srcdata.dim() == 3
            predicted = self.model(srcdata)
            predicted_value = predicted.detach().cpu().numpy().reshape(-1)[:]
            predicted_list.append(predicted_value)
        # 返回预测结果的列表
        return predicted_list