# -*- coding: utf-8 -*-
# @Time : 2021/4/27 下午1:20

"""
翻译一条语句
"""

from data_load.word_dict_loader import Word2Indexs
from experiment.inferencer import Inference
from model.seq2seq import EncoderGRU, DecoderGRU

def init():
    """
    初始化模型和相关配置
    Returns: 实例化inference对象
    """
    word2indexs = Word2Indexs(data_path='data/fra-eng-all.txt')
    encoder = EncoderGRU(input_size=word2indexs.input_lang.n_words, hidden_size=256)
    decoder = DecoderGRU(hidden_size=256, output_size=word2indexs.output_lang.n_words)
    
    inference = Inference(
        root_dir="/home/lisen/lisen/code/Seq2Seq",
        experiment_name="seq2seq",
        encoder=encoder,
        decoder=decoder,
        word2indexs=word2indexs,
        sentence_max_length=10
    )
    return inference
    
if __name__ == '__main__':
    """
    il a a faire .	he is busy .
    tu es beau .	you are beautiful .
    elles cherchent tom .	they re looking for tom .
    il est en train de te regarder .	he s looking at you .
    """
    french_sentence_list = ["il a a faire .",
                            "tu es beau .",
                            "elles cherchent tom .",
                            "il est en train de te regarder ."]
    inference = init()
    for sentence in french_sentence_list:
        res = inference._inference(sentence)
        print(sentence, '>', ' '.join(res))
    