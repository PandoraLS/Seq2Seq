# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 下午4:46

"""
数据处理的一些辅助代码
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

