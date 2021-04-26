# -*- coding: utf-8 -*-
# @Time : 2021/4/23 下午7:46
import importlib

class Language:
    def __init__(self, name):
        """
        word与index之间形成映射
        Args:
            name (str): 语种 eg.'eng'(英文) or 'fra'(法语)
        """
        super(Language, self).__init__()
        self.name = name
        self.word2index = {}  # [单词-索引]映射字典
        self.word_count = {}  # [单词-数量]字典
        self.index2word = {0: 'SOS', 1: 'EOS'}  # [索引－单词]映射字典
        self.n_words = 2  # include 'SOS' and 'EOS'　＃　单词数量

    def add_sentence(self, sentence):
        for word in sentence.split():  # 句子中的单词默认以空格隔开
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word_count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_count[word] += 1


class Word2Index:
    def __init__(self, data_path):
        """
        根据数据集all构建word到index的映射关系
        Args:
            data_path: 数据集(训练集+验证集)
        """
        super(Word2Index, self).__init__()
        self.dataset_lines = open(data_path, encoding='utf-8').read().strip().split('\n')
        self.pairs = [[s for s in l.split('\t')] for l in self.dataset_lines]
        self.input_lang, self.output_lang = Language('fra'), Language('eng')  # 输入法语,输出英文
        for pair in self.pairs:
            self.input_lang.add_sentence(pair[0])
            self.output_lang.add_sentence(pair[1])
        print('word2index init:', self.input_lang.name, self.input_lang.n_words, ',', self.output_lang.name,
              self.output_lang.n_words)

def initialize_config(module_cfg):
    """
    根据配置项，动态加载对应的模块， 并将参数传入模块内部的制定函数
    eg. 配置文件如下:
        module_cfg = {
            "module": "models.unet", 
            "main": "UNet",
            "args": {...}
        }
    1. 加载 type 参数对应的模块
    2. 调用(实例化)模块内部对应 main 参数的函数(类)
    3. 再调用(实例化)时将 args 参数输入函数(类)
    :param module_cfg: 配置信息， 参见json文件
    :return: 实例化后的函数(类)
    """
    module = importlib.import_module(module_cfg["module"])
    return getattr(module, module_cfg["main"])(**module_cfg["args"])



if __name__ == '__main__':
    pass
    cfg = {
        "module": "Word2Index_demo",
        "main": "Word2Index",
        "args": {
            "data_path": "/home/lisen/lisen/code/Seq2Seq/data/fra-eng-all.txt"
        }
    }
    word2indexs = initialize_config(cfg)
    print(word2indexs.input_lang)
    print()