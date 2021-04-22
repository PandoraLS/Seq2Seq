# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 下午4:46

"""
对原始数据清洗用到的辅助函数
"""
import re
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
    """
    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )
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

if __name__ == '__main__':
    pass