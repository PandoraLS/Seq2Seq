# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 下午4:12
"""
seq2seq载入数据
"""
import re
import unicodedata
from torch.utils.data import Dataset

class Language:
    def __init__(self, name):
        """
        word与index之间形成映射
        Args:
            name (str): 语种 eg.'eng'(英文) or 'fra'(法语)
        """
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2  # include 'SOS' and 'EOS'

    def add_sentence(self, sentence):
        for word in sentence.split():  # 句子中的单词默认以空格隔开
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class TextDataset(Dataset):
    def __init__(self, data_file_path, offset=0, limit=None, sample_length=10, do_filter=True):
        """
        训练数据组织结构
        Args:
            data_file_path (str): 文本数据的路径，详情见 'Notes'
            offset (int): 从第offset行开始读取数据
            limit (int): dataset的行数限制
            sample_length (int): 限制 输入sentence的长度
            do_filter (bool): 是否需要根据sentence的长度对数据集进行筛选
        Notes:
            data_file_path文件的格式如下

            文本文件中
            <language1 sentence 1><\t><language2 sentence 1>
            <language1 sentence 2><\t><language2 sentence 2>
            ...
            <language1 sentence n><\t><language2 sentence n>

            eg. 在文件'eng-fra.txt'中
            I'm cold.	J'ai froid.
            I'm done.	J'en ai fini.
            ...
            I'm fine.	Tout va bien.

        Returns:
        """
        super(TextDataset, self).__init__()

        dataset_lines = open(data_file_path, encoding='utf-8').read().strip().split('\n')
        dataset_lines = dataset_lines[offset:]
        if limit:  dataset_lines = dataset_lines[:limit]
        self.dataset_lines = dataset_lines
        self.pairs = [[self.normalize_string(s) for s in l.split('\t')] for l in self.dataset_lines]
        print("Read %s sentence pairs" % len(self.pairs))
        if do_filter:
            self.sample_length = sample_length  # 限制sentence的长度
            self.pairs = self.filter_pairs(self.pairs)
            print("Trimmed to %s sentence pairs" % len(self.pairs))

        input_lang, output_lang = Language('eng'), Language('fra')
        print("Counting words...")
        for pair in self.pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        self.length = len(self.pairs)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        pair = self.pairs[item]
        return pair

    # 法语具有和英语相同体系的字母表, 但法语的单词包含'声调', 可以转换掉, 以简化过程
    def unicode_to_ascii(self, s):
        """
        将unicode字符串转换为ascii形式
        Args:
            s (str): 输入的unicode字符串
        Returns (str): 转换为unicode格式的字符串
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(self, s):
        """
        对字符清洗
        Args:
            s (str): 待清洗字符串
        Returns (str): 清洗后的字符串
        """
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r'([.!?])', r' \1', s)  # 把标点[.!?]和单词以空格隔开
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # 所有非字幕和[.!?]的符号直接用空格替换
        return s

    # 全部数据太多，所以需要过滤掉一部分数据
    def filter_pair(self, p):
        """
        根据 self.sample_length 长度进行筛选
        """
        eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )
        return len(p[0].split(' ')) < self.sample_length and \
               len(p[1].split(' ')) < self.sample_length and \
               p[1].startswith(eng_prefixes)

    def filter_pairs(self, pairs):
        """
        根据sentence的长度，对sentence组成的pairs进行筛选，最终筛选得到小于 self.sample_length 长度的pairs
        Args:
            pairs: 从文件中读取处理得到的pairs
        Returns:
            筛选后的pairs
        """
        return [pair for pair in pairs if self.filter_pair(pair)]


if __name__ == '__main__':
    from pprint import pprint
    text_data = TextDataset(
        data_file_path='../data/eng-fra-val.txt',
        sample_length=10,
        do_filter=True
    )

    dataset_list_length = text_data.__len__()
    dataset_list = text_data.dataset_lines
    # pprint(dataset_list_length)
    # pprint(dataset_list)

    for item in text_data.__getitem__(0):
        pprint(item)