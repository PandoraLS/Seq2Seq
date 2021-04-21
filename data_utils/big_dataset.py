# -*- coding: utf-8 -*-
# @Time    : 2020/6/22 5:27 下午
# @Author  : sen

import torch
from collections import deque


def batch_loader(filepath, line_process_func, batch_size=64, verbose_total_cnt=False):
    """
    按顺序读取文件，并返回一个batch
    :param filepath:
    :param line_process_func: 对一行的处理函数
    :param batch_size:
    :param verbose_total_cnt: 是否预先显示总行数，会多遍历一遍文件，默认False
    :return:
    """
    if verbose_total_cnt:
        with open(filepath, 'r', encoding='utf-8') as f:
            line_cnt = 0
            for line in f:
                line_cnt += 1
            print('total', line_cnt, 'lines(samples)')

    with open(filepath, 'r', encoding='utf-8') as f:
        queue = deque([])
        for line in f:
            queue.append(line)
            if len(queue) == batch_size:
                # 需要根据情况，改这里，例子中只有四列
                batch_feature1, batch_feature2= [], []
                for _ in queue:
                    _1, _2 = line_process_func(_)
                    batch_feature1.append(_1)
                    batch_feature2.append(_2)
                yield batch_feature1,batch_feature2
                queue = deque([])

        # 剩余的line
        if len(queue) != 0:
            # 需要根据情况，改这里，例子中只有四列
            batch_feature1, batch_feature2= [], []
            for _ in queue:
                _1, _2 = line_process_func(_)
                batch_feature1.append(_1)
                batch_feature2.append(_2)

            yield batch_feature1, batch_feature2

def plc_line_process(line):
    elems = line.strip().split("\t\t")
    src_data = [float(elems[i]) for i in range(266)]
    tar_data = [float(_) for _ in elems[267:]]
    return src_data, tar_data

if __name__ == '__main__':
    for _1, _2 in batch_loader("/home/aone/lisen/code/Seq2Seq/data/tmp2.txt", line_process_func=plc_line_process, batch_size=2, verbose_total_cnt=True):
        print(_1, _2)
        