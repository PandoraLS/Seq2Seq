# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 10:44
# @Author  : sen

"""
基于Seq2Seq神经网络的Pitch预测，模型结构为Encoder-Decoder模式

前向预测：利用前2帧预测后1帧(丢失帧)
在Encoder阶段将前2帧(8个子帧)数据逐子帧的输入到Encoder中进行编码，将编码器最后一次的hidden state输出
在Decoder阶段将Encoder最后一次的hidden state作为Decoder的hidden初始化，然后逐子帧的解码4次得到下一帧的数据

上下文预测：利用前1帧和后1帧预测中间帧(丢失帧)
模型本身不需要做任何改变，唯一改变的是数据集
在Encoder阶段将2帧(前1帧和后1帧直接拼起来=8个子帧)数据逐子帧的输入到Encoder中进行编码，将编码器最后一次的hidden state输出
在Decoder阶段将Encoder最后一次的hidden state作为Decoder的hidden初始化，然后逐子帧的解码4次得到中间缺失帧的数据
PS: 前向预测和上下文预测得到的模型应当区分开，因为两者应用的场景是不同的!!!

训练：
配置好数据集路径，可以直接run起来，训练主函数为train_iters
模型会在训练过程中保留最新和最近的batch测试下的best模型

测试：
注释掉train_iters()的运行，载入模型，运行evaluate相关函数

import logger会自动的把运行过程中的输出以逐行添加的形式记录到run.log文件中
logger.py应当与当前文件(seq2seq_plc.py)在同一文件夹下
"""

import time
import math
import torch
import random
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import logger

print("\nstart time:  ", time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time())))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)

class Vocab:
    """
    pitch数据的范围是32～288，将这个范围映射到0～256
    """
    def __init__(self, min_num, max_num):
        self.token2id = {}
        self.id2token = {}
        self.sos_id = self.add_token(-1)  # 起始符
        self.eos_id = self.add_token(-2)  # 结束符
        self.unk_id = self.add_token(-3)  # unknown符号

        for token in range(min_num, max_num + 1):
            self.add_token(token)

    def add_token(self, token):
        if token in self.token2id:
            _id = self.token2id[token]
        else:
            _id = len(self.token2id)
            self.token2id[token] = _id
            self.id2token[_id] = token
        return _id

    def get_id(self, token):
        return self.token2id.get(token, self.unk_id)

    def get_token(self, id):
        return self.id2token.get(id, '[UNKONWN]')

    def __len__(self):
        return len(self.token2id)


vocab = Vocab(min_num=32, max_num=288)
HAS_ATTENTION = False
SRC_length = 8 # 输入数据的长度
TAR_length = 4 # 输出数据的长度

print('has_attention', HAS_ATTENTION)

"""
Model
"""


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # input_size是词典大小, hidden_size是词向量维度
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 这里nn.GRU(x, h)两个参数指明输入x和隐藏层状态的维度, 这里都用hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        :param input: 这里的input是每次一个词, 具体形式为: [word_idx]
        :param hidden:
        :return:
        """
        # input: [1]
        # embedding(input): [1, emb_dim]
        # embedded: [1, 1, 1 * emb_dim]
        embedded = self.embedding(input).view(1, 1, -1)

        # 关于gru的输入输出参数
        # [seq_len, batch_size, feture_size]
        # output: [1, 1, 1 * emb_dim]
        output = embedded
        # hidden: [1, 1, hidden_size]
        # 这里hidden_size == emb_dim
        output, hidden = self.gru(output, hidden)
        # output: [seq_len, batch, num_directions * hidden_size]
        # output: [1, 1, hidden_size]
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # embedding层的结构，1. 有多少个词，2. 每个词多少维
        self.embedding = nn.Embedding(output_size, hidden_size)
        # GRU的参数: 1. 输入x的维度, 2. 隐藏层状态的维度; 这里都用了hidden_size
        # emb_dim == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        # [batch_size, hidden_size] -> [batch_size, output_size]
        # 这里output_size就是目标语言字典的大小V
        self.out = nn.Linear(hidden_size, output_size)
        # softmax层, 求每一个单词的概率
        self.softmax = nn.LogSoftmax(dim=1)  # ?

    def forward(self, input, hidden):
        # input: [1], 一个单词的下标
        # hidden: [1, 1, hidden_size]
        # embedding(input): [emb_dim]
        output = self.embedding(input).view(1, 1, -1)  # 展开
        # output: [1, 1, emb_dim]
        output = F.relu(output)

        # output: [1, 1, emb_dim]

        # 关于gru的输入输出参数
        # [seq_len, batch_size, input_size],  [num_layers * num_directions, batch_size, hidden_size]
        # output: [1, 1, emb_dim], hidden: [1, 1, hidden_size]
        output, hidden = self.gru(output, hidden)
        # output: [1, 1, hidden_size] # [seq_len, batch, num_directions * hidden_size] # 这里hidden_size == emb_dim
        # output[0]: [1, emb_dim]
        # self.out(output[0]): [1, V]
        # output: [1, V] 值为每个单词的概率
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


"""
训练
"""

def read_dataset(fp, left_count=8, right_count=4):
    """
    从文件读取数据
    Args:
        fp: 文件路径
        left_count: src_data
        right_count: tar_data groundtruth
    Returns: src和tar组成的元组
    """
    pairs = []
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            elems = line.strip().split('\t\t')
            src = [int(_) for _ in elems[:left_count]]
            dst = [int(_) for _ in elems[left_count + 1: left_count + 1 + right_count]]
            pairs.append((src, dst))
    print('load dataset', fp, 'total sample', len(pairs))
    return pairs


def input_to_tensor(inputs):
    # 将输入添加EOS结束符后，变成tensor
    ids = [vocab.get_id(_) for _ in inputs]
    return torch.tensor(ids + [vocab.eos_id], dtype=torch.long, device=device).view(-1, 1)


def output_to_token(outputs):
    # 输出到字符的映射
    tokens = [vocab.get_token(_) for _ in outputs[:-1]]
    return tokens


def tensor_from_pair(pair):
    # 从pair中获取对应的源src和预测目标dst
    src, dst = pair
    input_tensor = input_to_tensor(src)
    target_tensor = input_to_tensor(dst)
    return input_tensor, target_tensor


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor,
          encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion, src_length=SRC_length + 1):
    """
    训练过程
    Args:
        input_tensor: 输入tensor
        target_tensor: 输出tensor
        encoder: 编码神经网络
        decoder: 解码神经网络
        encoder_optimizer: 编码器优化器
        decoder_optimizer: 解码器优化器
        criterion: 评估准则
        src_length: 输入帧的长度
    Returns: loss
    """
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(src_length, encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        # encoder 每次读取一个词, 重复input_length次
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # encoder_output: [1, 1, hidden_size]
        # encoder_output[ei]: [hidden_size]
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[vocab.sos_id]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  # , encoder_outputs)
            # decoder_output: [1, V] 值为每个单词的概率
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        # without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  # , encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == vocab.eos_id:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


"""
开始训练
"""


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (-%s)' % (as_minutes(s), as_minutes(rs))


pairs = read_dataset(r'/Users/seenli/Documents/workspace/code/pytorch_learn/seq2seq/hidden_20200717/data/pitch_2to1_clean.txt',
                     left_count=8, right_count=4)
pairs_val = read_dataset(r'/Users/seenli/Documents/workspace/code/pytorch_learn/seq2seq/hidden_20200717/data/pitch_2to1_clean_head2000.txt',
                         left_count=8, right_count=4)
# pairs = None
# pairs_val = None

min_loss_avg = float('inf')  # 用于判别最小值的loss


def evaluate(encoder, decoder, src_ids, src_length=SRC_length + 1, tar_length=TAR_length + 1):
    with torch.no_grad():
        input_tensor = input_to_tensor(src_ids)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(src_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            # encoder 每次读取一个词, 重复input_length次
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # encoder_output: [1, 1, hidden_size]
            # encoder_output[ei]: [hidden_size]
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[vocab.sos_id]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(tar_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  # , encoder_outputs)
            topv, topi = decoder_output.topk(1)

            if topi.item() == vocab.eos_id:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(topi.item())
            decoder_input = topi.squeeze().detach()  # detach from history as input
        decoded_words = output_to_token(decoded_words)
        return decoded_words


def evaluate_randomly(encoder, decoder, n=3000, has_attention=False):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        if has_attention:
            raise NotImplemented
            # output_words, attentions = evaluate_attention(encoder, decoder, pair[0])
        else:
            output = evaluate(encoder, decoder, pair[0])
        print('<', output)
        print('')


def evaluate_head(encoder, decoder, n=1000, has_attention=False):
    # eval 测试集中的前n行
    print("evaluate validation dataset ...")
    for i in range(n):
        pair = pairs_val[i]
        print(pair[0], ' > ', pair[1])
        if has_attention:
            raise NotImplemented
            # output_words, attentions = evaluate_attention(encoder, decoder, pair[0])
        else:
            output = evaluate(encoder, decoder, pair[0])
        print('<', output)
        print('')


def train_iters(encoder, decoder, n_iters, print_every=1000,
                plot_every=100, eval_every=100,
                learning_rate=0.01, has_attention=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # reset each print_every
    plot_loss_total = 0  # reset each plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [tensor_from_pair(random.choice(pairs)) for i in range(n_iters)]
    training_pairs = [random.choice(pairs) for i in range(n_iters)]

    # nn.NLLLoss(): The negative log likelihood loss. It is useful to train a classification problem with C classes.
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = tensor_from_pair(training_pairs[iter - 1])
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        if has_attention:
            raise NotImplemented
        else:
            loss = train(input_tensor, target_tensor,
                         encoder, decoder,
                         encoder_optimizer, decoder_optimizer,
                         criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))
            global min_loss_avg
            if print_loss_avg < min_loss_avg:
                min_loss_avg = print_loss_avg
                torch.save(encoder.state_dict(), 'encoder_best.pth')
                torch.save(decoder.state_dict(), 'decoder_best.pth')
                print('save best model, iter: %d, min_loss: %.4f' % (iter, min_loss_avg))

            # 每1000iter就会将最新的存储一下
            torch.save(encoder.state_dict(), 'encoder_last.pth')
            torch.save(decoder.state_dict(), 'decoder_last.pth')
            print_loss_total = 0

        if iter % plot_every == 0:
            plot_loss_avg = print_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if iter % eval_every == 0:
            print('evaluate random 3 sentence with last model ...')
            encoder.load_state_dict(torch.load('encoder_last.pth'))
            decoder.load_state_dict(torch.load('decoder_last.pth'))
            evaluate_randomly(encoder, decoder, n=3)


hidden_size = 64
print("hidden_size = ", hidden_size)

encoder = EncoderRNN(input_size=len(vocab), hidden_size=hidden_size).to(device)
decoder = DecoderRNN(hidden_size=hidden_size, output_size=len(vocab)).to(device)
train_iters(encoder=encoder,
            decoder=decoder,
            n_iters=1600000,
            print_every=1000,
            eval_every=10000) # 每eval_every小测一下

# 最后一轮没有必要进行测试了，最后测试出来的结果其实主要用于了解模型性能
# encoder.load_state_dict(torch.load('encoder_last.pth'))
# decoder.load_state_dict(torch.load('decoder_last.pth'))
# evaluate_head(encoder, decoder, n=300)

print("end time:  ", time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time())))

