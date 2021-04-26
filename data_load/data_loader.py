# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 下午4:12
"""
seq2seq载入数据
"""
import numpy as np
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_path, offset=0, limit=None):
        """
        训练数据组织结构，将所有的数据读取到内存后再处理，不可处理太大的文件
        Args:
            data_path (str): 文本数据的路径，详情见 'Notes'
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
            [language1 sentence, language2 sentence]
        """
        super(TextDataset, self).__init__()
        dataset_lines = open(data_path, encoding='utf-8').read().strip().split('\n')
        dataset_lines = dataset_lines[offset:]
        if limit:  dataset_lines = dataset_lines[:limit]
        self.dataset_lines = dataset_lines
        self.pairs = [[s for s in l.split('\t')] for l in self.dataset_lines]
        print("Read %s sentence pairs" % len(self.pairs))
        self.length = len(self.pairs)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        pair = self.pairs[item]
        return pair[0], pair[1]  # 返回法文and英文
    
if __name__ == '__main__':
    text_data = TextDataset(data_path='../data/fra-eng-val.txt')
    dataset_list_length = text_data.__len__()
    dataset_list = text_data.dataset_lines
    pairs = text_data.pairs
    print("dataset list length: ", dataset_list_length)
    # input_lang = text_data.input_lang
    print()
    # pprint(dataset_list)
    # pprint(pairs)
    for item in text_data.__getitem__(2):
        print(item)
    print(text_data.__getitem__(2))
    print('--------------------------------------------------------')
    