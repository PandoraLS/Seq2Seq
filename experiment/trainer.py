# -*- coding: utf-8 -*-
# @Time : 2021/4/22 下午12:37

import torch
import random
from experiment.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self,
                 config: dict,
                 resume: bool,
                 encoder,
                 decoder,
                 optim_enc,
                 optim_dec,
                 loss_fucntion,
                 visual,
                 dataset,
                 word2indexs,
                 sentence_max_length):
        super().__init__(config, resume, encoder, decoder, optim_enc, optim_dec, loss_fucntion, visual)
        self.SOS_token = 0
        self.EOS_token = 1
        self.dataset = dataset
        self.word2indexs = word2indexs
        self.sentence_max_length = sentence_max_length # 输入sentence的最大长度,与data_prep.py中的max_length保持一致
        self.print_loss_every = 1000 # 每1000个iter打印一次loss值
        self.print_loss_total = 0
        self.n_iter = 0

    def _train_epoch(self, epoch):
        # TODO 目前这种载入数据的方法非常难用，需要自定义dataloader方法,参考下面链接
        # https://github.com/PandoraLS/Chinese-Text-Classification-Pytorch/blob/master/utils.py
        for i in range(self.dataset.length):
            self.n_iter += 1
            (src_sentence, tar_sentence) = self.dataset.__getitem__(i)
            input_tensor, output_tensor = self._pair_to_tensor(src_sentence, tar_sentence)  # 把数据转换为tensor
            loss_iter = self._train_iter(input_tensor, output_tensor, self.sentence_max_length)
            self.print_loss_total += loss_iter

            if self.n_iter % self.print_loss_every == 0:
                print_loss_avg = self.print_loss_total / self.print_loss_every
                self.print_loss_total = 0
                print("iter / current 1000 iter mean loss: {} / {:.4f}".format(self.n_iter, print_loss_avg))

                # 验证当前翻译效果
                print("input:       ", src_sentence)
                output_words = self._eval_iter(input_tensor, self.sentence_max_length)
                print("predict:     ", ' '.join(output_words))
                print("groundtruth: ", tar_sentence)
                print()

            with torch.no_grad():
                if self.visual:
                    self.writer.add_scalars(f"模型/损失值_n_iter", {
                        "loss_iter": loss_iter
                    }, self.n_iter)
        
    def _train_iter(self, input_tensor, output_tensor, max_length):
        """
        对单个输入的input_tensor, output_tensor进行训练
        Args:
            input_tensor: tensor格式的输入sentence
            output_tensor: tensor格式的输出sentence
            max_length: 筛选得到的的sentence的最大长度, 这里为10
        Returns:
            当前iter的loss值
        """
        encoder_hidden = self.encoder.init_hidden().to(self.device)
        self.optimizer_enc.zero_grad()
        self.optimizer_dec.zero_grad()
        input_length, output_length = input_tensor.size(0), output_tensor.size(0)
        # 这里的10是因为数据集筛选max_length=10的sentence
        encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=self.device)
        
        loss = 0
        for ei in range(input_length):
            # encoder 每次读取一个词, 重复input_length次
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            # encoder_output: [1, 1, hidden_size]
            # encoder_output[ei]: [hidden_size]
            encoder_outputs[ei] = encoder_output[0, 0]
        
        decoder_input = torch.tensor([[self.SOS_token]], device=self.device)
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(output_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)  # , encoder_outputs)
                # decoder_output: [1, V] 值为每个单词的概率
                loss += self.loss_function(decoder_output, output_tensor[di])
                decoder_input = output_tensor[di]
        else:
            # without teacher forcing: use its own predictions as the next input
            for di in range(output_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)  # , encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.loss_function(decoder_output, output_tensor[di])
                if decoder_input.item() == self.EOS_token:
                    break
        
        loss.backward()
        self.optimizer_enc.step()
        self.optimizer_dec.step()
        
        return loss.item() / output_length

    def _eval_iter(self, input_tensor, max_length):
        with torch.no_grad():
            input_length = input_tensor.size(0)
            encoder_hidden = self.encoder.init_hidden().to(self.device)
            encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                # encoder 每次读取一个词, 重复input_length次
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                # encoder_output: [1, 1, hidden_size]
                # encoder_output[ei]: [hidden_size]
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[self.SOS_token]], device=self.device)
            decoder_hidden = encoder_hidden

            decoded_words = []
            for di in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)  # , encoder_outputs)
                topv, topi = decoder_output.topk(1)

                if topi.item() == self.EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.word2indexs.output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()  # detach from history as input
            return decoded_words[:-1]
    
    def _pair_to_tensor(self, input_sentence, output_sentence):
        """
        将dataloader得到的[法文语句, 英文语句]映射为index后，转换为tensor
        Args:
            input_sentence (str): 法文语句
            output_sentence (str): 英文语句 
        Returns (tuple):
            input_tensor, output_tensor
        """
        input_indexes = [self.word2indexs.input_lang.word2index[word] for word in input_sentence.split(' ')]
        input_indexes.append(self.EOS_token)
        input_tensor = torch.tensor(input_indexes, dtype=torch.long, device=self.device).view(-1, 1)

        output_indexes = [self.word2indexs.output_lang.word2index[word] for word in output_sentence.split(' ')]
        output_indexes.append(self.EOS_token)
        output_tensor = torch.tensor(output_indexes, dtype=torch.long, device=self.device).view(-1, 1)

        return input_tensor, output_tensor

    def _visualize_weights_and_grads(self, model, epoch):
        # 绘制模型训练曲线
        if self.visual:
            for name, param in model.named_parameters():
                self.writer.add_histogram("WEIGHT_" + name, param.clone().cpu().data.numpy(), epoch)
                self.writer.add_histogram("GRAD_" + name, param.grad.cpu().numpy(), epoch)

    def _validation_epoch(self, epoch):
        raise NotImplementedError
