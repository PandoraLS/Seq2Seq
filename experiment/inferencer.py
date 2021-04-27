# -*- coding: utf-8 -*-
# @Time : 2021/4/27 下午1:40

"""
载入模型，翻译单条语句
"""

import torch
from pathlib import Path

class Inference:
    def __init__(self,
                 root_dir,                  # 根目录(绝对路径)
                 experiment_name,           # 运行实验的名称,这里是 "seq2seq"
                 encoder,                   # encoder部分
                 decoder,                   # decoder部分
                 word2indexs,               # 单词-序号 映射关系
                 sentence_max_length):
        self.n_gpu = torch.cuda.device_count()
        self.device = self._prepare_device(self.n_gpu, cudnn_deterministic=False)
        
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        if self.n_gpu > 1:
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=list(range(self.n_gpu)))
            self.decoder = torch.nn.DataParallel(self.decoder, device_ids=list(range(self.n_gpu)))

        self.root_dir = Path(root_dir).expanduser().absolute() / "runs" / experiment_name
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.logs_dir = self.root_dir / "logs"

        self.SOS_token = 0
        self.EOS_token = 1
        self.word2indexs = word2indexs
        self.sentence_max_length = sentence_max_length  # 输入sentence的最大长度,与data_prep.py中的max_length保持一致
        self._inference_checkpoint() # 载入last_model
    
    def _inference_checkpoint(self):
        """
        test experiment from latest checkpoint.
        Notes: To be careful at Loading model.
                if model is an instance of DataParallel, we need to set model.module.*
        :return: None
        """
        latest_model_path = self.checkpoints_dir / "latest_model.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can't load latest checkpoint."
        checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.device)
        self.start_epoch = checkpoint["epoch"]

        if isinstance(self.encoder, torch.nn.DataParallel):
            self.encoder.module.load_state_dict(checkpoint["encoder"])
            self.decoder.module.load_state_dict(checkpoint["decoder"])
        else:
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])

        print(f"Model checkpoint loaded. Inferencing in {self.start_epoch} epoch.")
    
    def _sentence_to_tensor(self, input_sentence):
        """
        将输入的input_sentence(法语)映射为index后转换为tensor
        Args:
            input_sentence (str): 法文 
        Returns: 映射后的,tensor格式的输入sentence
        """
        input_indexes = [self.word2indexs.input_lang.word2index[word] for word in input_sentence.split(' ')]
        input_indexes.append(self.EOS_token)
        input_tensor = torch.tensor(input_indexes, dtype=torch.long, device=self.device).view(-1, 1)
        return input_tensor
        
    
    def _inference(self, input_sentence):
        """
        对输入的语句翻译
        Args:
            input_sentence (str): 法文语句 
        Returns:
        """
        input_tensor = self._sentence_to_tensor(input_sentence)
        with torch.no_grad():
            input_length = input_tensor.size(0)
            encoder_hidden = self.encoder.init_hidden().to(self.device)
            encoder_outputs = torch.zeros(self.sentence_max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                # encoder 每次读取一个词, 重复input_length次
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                # encoder_output: [1, 1, hidden_size]
                # encoder_output[ei]: [hidden_size]
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[self.SOS_token]], device=self.device)
            decoder_hidden = encoder_hidden

            decoded_words = []
            for di in range(self.sentence_max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)  # , encoder_outputs)
                topv, topi = decoder_output.topk(1)

                if topi.item() == self.EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.word2indexs.output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()  # detach from history as input
            return decoded_words[:-1]

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
        
        