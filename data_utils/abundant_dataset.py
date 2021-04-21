# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-3-8 下午3:33

import numpy as np
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, dataset, limit=None, offset=0, sample_length=1, train=True):
        """
        Construct training dataset
        Args:
            dataset (str): the path of dataset list，see "Notes"
            limit (int): the limit of dataset
            offset (int): the offset of dataset
            sample_length(int): the model only support fixed-length input in training, this parameter specify the input size of the model.
            train(bool): In training, the model need fixed-length input; in test, the model need fully-length input.

        Notes:
            the format of the waveform dataset is as follows.

            In list file:
            <srcdata 1><space><targetdata 1>
            <srcdata 2><space><targetdata 2>
            ...
            <srcdata n><space><targetdata n>

            e.g.
            In "dataset.txt":
            -1. 1.
            0.4 0.16
            0.5 0.25
            1. 1
        Return:
            src_data_array, target_data_array
        """
        super(RegressionDataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(dataset)]
        dataset_list = dataset_list[offset:]  # # 从第offset行开始取数据
        if limit: # 限制dataset_list最多取多少行
            dataset_list = dataset_list[:limit]
        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        src_data, tar_data = self.dataset_list[item].split(" ")
        src_data, tar_data = np.array(np.float32(src_data)), np.array(np.float32(tar_data))
        # tmp = src_data.reshape(1, -1)
        # print(tmp)
        return src_data.reshape(1, -1), tar_data.reshape(1, -1) # 将数据reshape为 [1, 1] 一行一列的二维数组

if __name__ == '__main__':

    from pprint import pprint

    wave_data = RegressionDataset(
        dataset="yx_dataset.txt",
        train=True
    )
    dataset_list_length = wave_data.__len__()
    dataset_list = wave_data.dataset_list
    pprint(dataset_list_length)
    pprint(dataset_list)

    for item in wave_data.__getitem__(1):
        pprint(item)
    
    
    
