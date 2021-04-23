# -*- coding: utf-8 -*-
# @Time    : 2021/4/21 下午3:47

"""
对训练数据预处理，训练数据来源于dataset/eng-fra.txt
其格式如下
<english sentence 1><\t><french sentence 1>
<english sentence 2><\t><french sentence 2>
...
<english sentence n><\t><french sentence n>

将数据处理为可以训练和验证的组织形式
data/fra-eng-train.txt
data/fra-eng-val.txt
"""

import os
import re
import random
import unicodedata

SOS_token = 0
EOS_token = 1

# 法语具有和英语相同体系的字母表, 但法语的单词包含'声调', 可以转换掉, 以简化过程
def unicode_to_ascii(s):
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

def normalize_string(s):
    """
    对字符清洗
    Args:
        s (str): 待清洗字符串
    Returns (str): 清洗后的字符串
    """
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)  # 把标点[.!?]和单词以空格隔开
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # 所有非字幕和[.!?]的符号直接用空格替换
    return s


# 全部数据太多，所以需要过滤掉一部分数据
def filter_pair(p, max_length):
    """
    根据max_length对输入的pair进行筛选
    p的格式为fra-eng：['c est vous la chef .', 'you re the leader .']
    """
    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )
    # 只保留eng_prefixes开头且句子长度小于10的sentence
    return len(p[0].split(' ')) < max_length and \
           len(p[1].split(' ')) < max_length and \
           p[1].startswith(eng_prefixes)

def filter_pairs(pairs):
    """
    根据sentence的长度，对sentence组成的pairs进行筛选，最终筛选得到小于 self.sample_length 长度的pairs
    Args:
        pairs: 从文件中读取处理得到的pairs
    Returns:
        筛选后的pairs
    """
    return [pair for pair in pairs if filter_pair(pair, max_length=10)]

def reverse_lang(src_path, tar_path):
    """
    将数据集eng-fra.txt英法的顺序交换一下
    Args:
        src_path (str): 原始文件的路径
        tar_path (str): 目标文件路径
    Returns:
    """
    # src_path = r'../data/eng-fra-val.txt'
    # tar_path = r'../data/fra-eng-val.txt'
    tar_file = open(tar_path, 'w', encoding='utf8')
    with open(src_path) as src_file:
        for line in src_file:
            lang1, lang2 = line.strip().split('\t')
            tar_file.write(lang2)
            tar_file.write('\t')
            tar_file.write(lang1)
            tar_file.write('\n')
            
def data_preprocess(do_filter=True):
    """
    准备训练验证数据集，将原始数据集dataset/eng-fra.txt处理后放到data/文件夹下
    Args:
        do_filter (bool): 对数据集筛选，去掉较长的数据
    data/文件结构如下：
        fra-eng-all.txt         # 筛选清洗后的　乱序版　法－英　训练集＋验证集
        fra-eng-train.txt       # 筛选清洗后的 乱序版 法-英 训练集
        fra-eng-val.txt         # 筛选清洗后的 乱序版 法-英 验证集
    Returns:
    """
    assert os.path.exists('dataset/eng-fra.txt'), "dataset/eng-fra.txt not exist"
    dataset_lines = open('dataset/eng-fra.txt', encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in dataset_lines]
    pairs = [[pair[1], pair[0]] for pair in pairs] # 交换语言顺序为 法-英
    print("Read %s sentence pairs" % len(pairs))
    
    if do_filter:
        pairs = filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
    
    # 设置seed保证数据可重复
    SEED = 1
    random.seed(SEED)
    random.shuffle(pairs) # 随机打乱数据集
    train_pairs = pairs[:int(len(pairs) * 0.9)] # 9/10 用于训练集
    val_pairs = pairs[int(len(pairs) * 0.9):] # 1/10 用于验证集
    
    if not os.path.exists('data/'):
        os.makedirs('data/')
    all_pair_file = open('data/fra-eng-all.txt', 'w', encoding='utf8')
    for pair in pairs:
        all_pair_file.write(pair[0] + '\t' + pair[1] + '\n')
    train_file = open('data/fra-eng-train.txt', 'w', encoding='utf8')
    for pair in train_pairs:
        train_file.write(pair[0] + '\t' + pair[1] + '\n')
    val_file = open('data/fra-eng-val.txt', 'w', encoding='utf8')
    for pair in val_pairs:
        val_file.write(pair[0] + '\t' + pair[1] + '\n')
    

if __name__ == '__main__':
    data_preprocess()
