# -*- coding: utf-8 -*-
# @Time    : 2021/4/21 下午3:47

"""
处理data的辅助性代码
"""

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


if __name__ == '__main__':
    reverse_lang()