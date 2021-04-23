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
    def __init__(self, data_file_path, offset=0, limit=None):
        """
        训练数据组织结构，将所有的数据读取到内存后再处理，不可处理太大的文件
        Args:
            data_file_path (str): 文本数据的路径，详情见 'Notes'
            offset (int): 从第offset行开始读取数据
            limit (int): dataset的行数限制
        Notes:
            data_file_path文件的格式如下

            文本文件中
            <language1 sentence 1><\t><language2 sentence 1>
            <language1 sentence 2><\t><language2 sentence 2>
            ...
            <language1 sentence n><\t><language2 sentence n>

            eg. 在文件'fra-eng.txt'(法-英)中
            Il a laissé tomber.	He quit.
            Il court.	He runs.
            ...
            Aide-moi !	Help me!
        Returns:
        """
        super(TextDataset, self).__init__()

        dataset_lines = open(data_file_path, encoding='utf-8').read().strip().split('\n')
        dataset_lines = dataset_lines[offset:]
        if limit:  dataset_lines = dataset_lines[:limit]
        self.dataset_lines = dataset_lines
        self.pairs = [[s for s in l.split('\t')] for l in self.dataset_lines]
        print("Read %s sentence pairs" % len(self.pairs))
        input_lang, output_lang = Language('eng'), Language('fra')
        print("Counting words...")
        for pair in self.pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
        print(input_lang.name, input_lang.n_words, "\t", output_lang.name, output_lang.n_words)
        self.length = len(self.pairs)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        pair = self.pairs[item]
        return pair


if __name__ == '__main__':
    from pprint import pprint
    text_data = TextDataset(data_file_path='../data/fra-eng-val.txt')

    dataset_list_length = text_data.__len__()
    dataset_list = text_data.dataset_lines
    pairs = text_data.pairs
    print("dataset list length: ", dataset_list_length)
    # pprint(dataset_list)
    # pprint(pairs)

    for item in text_data.__getitem__(2):
        pprint(item)