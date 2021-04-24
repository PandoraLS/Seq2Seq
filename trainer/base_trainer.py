# -*- coding: utf-8 -*-
# @Time : 2021/4/22 下午12:18

import time
import torch
import json5
import numpy as np
from pathlib import Path
from utils.util import prepare_empty_dir, ExecutionTime

class BaseTrainer:
    def __init__(self,
                 config,
                 resume,
                 encoder,
                 decoder,
                 optim_enc,
                 optim_dec,
                 loss_function,
                 visual):
        self.n_gpu = torch.cuda.device_count()
        self.device = self._prepare_device(self.n_gpu, cudnn_deterministic=config["cudnn_deterministic"])

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        if self.n_gpu > 1:
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=list(range(self.n_gpu)))
            self.decoder = torch.nn.DataParallel(self.decoder, device_ids=list(range(self.n_gpu)))

        self.optimizer_enc = optim_enc
        self.optimizer_dec = optim_dec
        self.optimizer_enc.zero_grad()
        self.optimizer_dec.zero_grad()

        self.loss_function = loss_function
        self.visual = visual
        
        # word2indexs
        self.word2indexs = config["word2index"]

        # Trainer
        self.epochs = config["trainer"]["epochs"]
        self.save_checkpoint_interval = config["trainer"]["save_checkpoint_interval"]
        self.teacher_forcing_ratio = config["trainer"]["teacher_forcing_ratio"] # 使用teacher_forcing 所占的比例
        self.validation_config = config["trainer"]["validation"]
        self.validation_interval = self.validation_config["interval"]
        self.find_max = self.validation_config["find_max"]
        self.validation_custom_config = self.validation_config["custom"]
        
        self.start_epoch = 1
        self.best_score = -np.inf if self.find_max else np.inf
        self.root_dir = Path(config["root_dir"]).expanduser().absolute() / "runs" / config["experiment_name"]
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.logs_dir = self.root_dir / "logs"
        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume)
        
        if self.visual:
            from utils import visualization
            self.writer = visualization.writer(self.logs_dir.as_posix())
            self.writer.add_text(
                tag="Configuration",
                text_string=f"<pre> \n{json5.dumps(config, indent=4, sort_keys=False)}  \n</pre>",
                global_step=1
            )

        if resume: self._resume_checkpoint()

        print('Configurations are as follow: ')
        print(json5.dumps(config, indent=2, sort_keys=False))

        with open((self.root_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.json").as_posix(), 'w') as handle:
            json5.dump(config, handle, indent=2, sort_keys=False)

        self._print_networks([self.encoder, self.decoder])

    def _resume_checkpoint(self):
        """
        Resume experiment from latest checkpoint.
        Notes: To be careful at Loading model.
                if model is an instance of DataParallel, we need to set model.module.*
        :return: None
        """
        latest_model_path = self.checkpoints_dir / "latest_model.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can't load latest checkpoint."

        checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.device)

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.optimizer_enc.load_state_dict(checkpoint["optimizer_enc"])
        self.optimizer_dec.load_state_dict(checkpoint["optimizer_dec"])

        if isinstance(self.encoder, torch.nn.DataParallel):
            self.encoder.module.load_state_dict(checkpoint["encoder"])
            self.decoder.module.load_state_dict(checkpoint["decoder"])
        else:
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])

        print(f"Model checkpoint loaded. Training will begin in {self.start_epoch} epoch.")

    def _save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoints to <root_dir>/checkpoints directory, which contains:
            - current epoch
            - best score in history
            - optimizer parameters
            - model parameters
        :param epoch: Epoch
        :param is_best(bool): if current checkpoint got the best score, it also will be saved in <root_dir>/checkpoints/best_model.tar.
        :return:
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

        # Construct checkpoint tar package
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer_enc": self.optimizer_enc.state_dict(),
            "optimizer_dec": self.optimizer_dec.state_dict()
        }

        if self.device.type == "cuda" and self.n_gpu > 1:  # Parallel
            state_dict["encoder"] = self.encoder.module.cpu().state_dict()
            state_dict["decoder"] = self.decoder.module.cpu().state_dict()
        else:
            state_dict["encoder"] = self.encoder.cpu().state_dict()
            state_dict["decoder"] = self.decoder.cpu().state_dict()
        """
        Notes:
            - latest_model.tar:
                Contains all checkpoint information, including optimizer parameters, model parameters, etc. 
                New checkpoint will overwrite old one.
            - model_<epoch>.pth: 
                The parameters of encoder's network. Follow-up we can specify epoch to inference.
            - best_model.tar:
                Like latest_model, but only saved when <is_best> is True.
        """
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())
        torch.save(state_dict["encoder"], (self.checkpoints_dir / f"encoder_{str(epoch).zfill(4)}.pth").as_posix())
        torch.save(state_dict["decoder"], (self.checkpoints_dir / f"decoder_{str(epoch).zfill(4)}.pth").as_posix())
        
        # 由于不通过score来评判模型，所以不存储best模型
        # if is_best:
        #     print(f"\t Found best score in {epoch} epoch, saving...")
        #     torch.save(state_dict, (self.checkpoints_dir / "best_model.tar").as_posix())
        #     # torch.save(state_dict["model"], (self.checkpoints_dir / f"model_best.pth").as_posix())

        # Use model.cpu(), model.to("cpu") will migrate the model to CPU, at which point we need re-migrate model back.
        # No matter tensor.cuda() or torch.to("cuda"), if tensor in CPU, the tensor will not be migrated to GPU, but the model will.
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    @staticmethod
    def _prepare_device(n_gpu: int, cudnn_deterministic=False):
        """
        Choose to use CPU or GPU depend on "n_gpu".
        :param n_gpu(int): the number of GPUs used in the experiment.
            if n_gpu is 0, use CPU;
            if n_gpu > 1, use GPU.
        :param cudnn_deterministic(bool):
                repeatability cudnn.benchmark will find algorithms to optimize training.
                if we need to consider the repeatability of experiment, set use_cudnn_deterministic to True
        :return: device
        """
        if n_gpu == 0:
            print("Using CPU in the experiment.")
            device = torch.device("cpu")
        else:
            if cudnn_deterministic:
                print("Using CuDNN deterministic mode in the experiment.")
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            device = torch.device("cuda:0")
        return device

    def _is_best(self, score, find_max=True):
        """
        Check if the current model is the best model
        Args:
            score: 评估指标
            find_max: True表示score越大模型越好, False表示score越小模型越好
        Returns (bools): 是最佳模型True, 不是最佳模型False
        """
        if find_max and score >= self.best_score: # score取最大值时为best
            self.best_score = score
            return True
        elif not find_max and score <= self.best_score: # score取最小值时为best
            self.best_score = score
            return True
        else:
            return False

    @staticmethod
    def _print_networks(nets: list):
        print(f"This project contain {len(nets)} networks, the number of the parameters: ")
        params_of_all_networks = 0
        for i, net in enumerate(nets, start=1):
            params_of_network = 0
            for param in net.parameters():
                params_of_network += param.numel()

            print(f"\t Network {i}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(f"The amount of parameters in project is {params_of_all_networks / 1e6} million.")

    def _set_models_to_train_mode(self):
        self.encoder.train()
        self.decoder.train()

    def _set_models_to_eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"================== {epoch} epoch ==================")
            print("[0 seconds] Begin training...")
            timer = ExecutionTime()

            self._set_models_to_train_mode()
            self._train_epoch(epoch)

            if self.save_checkpoint_interval != 0 and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch)

            if self.validation_interval != 0 and epoch % self.validation_interval == 0:
                print(f"[{timer.duration():.3f} seconds] Training is over, Validation is in progress...")

                self._set_models_to_eval_mode()
                
                # 由于不进行验证集的指标评估，所以不通过score保存模型
                # score = self._validation_epoch(epoch)
                # if self._is_best(score, find_max=self.find_max):
                #     print(f"\t Best score: {score:.4f}")
                #     self._save_checkpoint(epoch, is_best=True)

            print(f"[{timer.duration():.3f} seconds] End this epoch.")

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _validation_epoch(self, epoch):
        raise NotImplementedError

    def _test(self):
        raise NotImplementedError
    
if __name__ == '__main__':
    pass
