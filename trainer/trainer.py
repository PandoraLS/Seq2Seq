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
                 encoder,
                 decoder,
                 optim_enc,
                 optim_dec,
                 loss_fucntion,
                 word2indexs,
                 visual,
                 train_dl,
                 validation_dl):
        super().__init__(config, resume, encoder, decoder, optim_enc, optim_dec, loss_fucntion, word2indexs, visual)
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl
        self.SOS_token = 0
        self.EOS_token = 1

    def _visualize_weights_and_grads(self, model, epoch):
        # 绘制模型训练曲线
        if self.visual:
            for name, param in model.named_parameters():
                self.writer.add_histogram("WEIGHT_" + name, param.clone().cpu().data.numpy(), epoch)
                self.writer.add_histogram("GRAD_" + name, param.grad.cpu().numpy(), epoch)
    
    def _pair_to_tensor(self, input_sentence, output_sentence):
        """
        将dataloader得到的[法文语句, 英文语句]映射为index后，转换为tensor
        Args:
            input_sentence (str): 法文语句
            output_sentence (str): 英文语句 
        Returns (tuple):
            (input_tensor, output_tensor)
        """
        input_indexes = [self.word2indexs.input_lang.word2index[word] for word in input_sentence.split(' ')]
        input_indexes.append(self.EOS_token)
        input_tensor = torch.tensor(input_indexes, dtype=torch.long, device=self.device).view(-1, 1)
        
        output_indexes = [self.word2indexs.output_lang.word2index[word] for word in output_sentence.split(' ')]
        output_indexes.append(self.EOS_token)
        output_tensor = torch.tensor(output_indexes, dtype=torch.long, device=self.device).view(-1, 1)
        
        return (input_tensor, output_tensor)
        
    
    def _train_epoch(self, epoch):
        for i, (src_sentence, tar_sentence) in enumerate(self.train_dataloader, start=1):
            # For visualization
            batch_size = self.train_dataloader.batch_size
            n_batch = len(self.train_dataloader)
            # model共处理了n_iter条数据
            n_iter = n_batch * batch_size * (epoch - 1) + i * batch_size
            
            pair = self._pair_to_tensor(src_sentence, tar_sentence) # 把数据转换为tensor
            
            
            
            predicted = self.model(src_sentence)  # 
            loss_mean = []

            """================ Optimize model ================"""
            self.optimizer.zero_grad()
            loss = self.loss_function(predicted, tar_sentence)
            loss.backward()
            self.optimizer.step()
            loss_mean.append(loss.data.cpu().numpy())
            # print("loss", loss)

            with torch.no_grad():
                predicted = self.model(src_sentence)
                if self.visual:
                    self.writer.add_scalars(f"模型/损失值_n_iter", {
                        "模型优化前": loss,
                        "模型优化后": self.loss_function(predicted, tar_sentence)
                    }, n_iter)
        logging.info("epoch / mean loss: {} / {:.4f}".format(epoch, np.mean(loss_mean)))


    def _validation_epoch(self, epoch):
        raise NotImplementedError
