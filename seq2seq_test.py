# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 10:55
# @Author  : sen

"""
与opus codec配合着进行编解码操作，本程序的主要功能是
- 载入神经网络模型(单向模型 and 上下文模型)
- 在offline对pitch进行休整
- 使用Opus Codec产生的exe文件对读取的修正后的pitch文件进行解码
- 载入wbpesq的exe文件对解码后的结果进行测试
"""

import torch
import os, re
import shutil
import time
import logger

from seq2seq.seq2seq_plc import EncoderRNN, DecoderRNN, Vocab, evaluate

def creat_dir(d):
    """
    新建文件夹
    Args:
        d: 文件夹路径
    Returns:
    """
    if not os.path.exists(d):
        os.makedirs(d)

def model_init_single_track():
    """
    初始化模型参数,加载模型参数
    单向预测模型，可以基于前2帧数据，预测后一帧的数据
    Returns:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab(min_num=32, max_num=288)
    hidden_size = 256
    encoder = EncoderRNN(input_size=len(vocab), hidden_size=hidden_size).to(device)
    decoder = DecoderRNN(hidden_size=hidden_size, output_size=len(vocab)).to(device)
    encoder.load_state_dict(torch.load('C:\Education\code\plc\seq2seq\hidden_20200717\history\model_single_0717_20oclock\encoder_single_last.pth'))
    decoder.load_state_dict(torch.load('C:\Education\code\plc\seq2seq\hidden_20200717\history\model_single_0717_20oclock\decoder_single_last.pth'))
    print('single track model init complete...')
    return encoder, decoder

def model_init_context():
    """
    初始化模型，加载模型参数
    根据上下文预测单帧数据，基于前一帧和后一帧预测中间丢失的帧
    Returns:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab(min_num=32, max_num=288)
    hidden_size = 256
    encoder = EncoderRNN(input_size=len(vocab), hidden_size=hidden_size).to(device)
    decoder = DecoderRNN(hidden_size=hidden_size, output_size=len(vocab)).to(device)
    encoder.load_state_dict(torch.load('C:\Education\code\plc\seq2seq\hidden_20200717\history\model_context_0717_20oclock\encoder_context_last.pth'))
    decoder.load_state_dict(torch.load('C:\Education\code\plc\seq2seq\hidden_20200717\history\model_context_0717_20oclock\decoder_context_last.pth'))
    print('context model init complete...')
    return encoder, decoder

def single_frame_predicte(enc, dec, src_list):
    """
    输入前3帧的数据(int型的list)，返回模型预测的结果
    Args:
        enc: 编码器
        dec: 解码器
        src_list: 待预测的3帧pitch数据(数据是int型的)
    Returns(list): 1帧pitch数据(数据也是int型的)
    """
    from seq2seq.seq2seq_plc import evaluate
    output = evaluate(enc, dec, src_list)
    # print(output)
    return output

def single_frame_predicte_124to3(enc, dec, src_list):
    """
    输入第1, 2, 4帧的数据, 返回第3帧的预测结果，以利用后文信息
    Args:
        enc: 编码器
        dec: 解码器
        src_list: 用于预测的src数据:第1,2,4帧的pitch数据(12个数),数据类型为(int)
    Returns: 第3帧(4个数)pitch,数据类型为(int)
    """
    output = evaluate(enc, dec, src_list)
    print(output)
    return output


def single_frame_predicte_12to3(enc, dec, src_list):
    """
    输入前3帧的数据(int型的list)，返回模型预测的结果
    Args:
        enc: 编码器
        dec: 解码器
        src_list: 待预测的2帧pitch数据(数据是int型的)
    Returns(list): 1帧pitch数据(数据也是int型的)
    """
    from seq2seq.seq2seq_plc import evaluate
    output = evaluate(enc, dec, src_list)
    # print(output)
    return output

def single_frame_predicte_13to2(enc, dec, src_list):
    """
    输入第1, 3帧的数据, 返回第2帧的预测结果，以利用后文信息
    Args:
        enc: 编码器
        dec: 解码器
        src_list: 用于预测的src数据:第1,3帧的pitch数据(8个数),数据类型为(int)
    Returns: 第2帧(4个数)pitch,数据类型为(int)
    """
    output = evaluate(enc, dec, src_list)
    # print(output)
    return output


def forceCopyFile(sfile, dfile):
    """
    将文件覆盖拷贝
    Args:
        sfile: 源文件path
        dfile: 目标文件path
    Returns:
    """
    if os.path.isfile(sfile):
        shutil.copy2(sfile, dfile)


def decoder_no_loss(opus_no_loss_path, pcm_file):
    """
    对pcm_file解码，将原始pcm_file和解码的pcm_dec_file都放到<pcm_file>子目录下
    同时生成的pitch_4cols_noloss.txt和pitch_5cols_noloss.txt也放到<pcm_file>子目录下
    Args:
        opus_no_loss_path: opus_no_loss.exe所在的路径
        pcm_file: 原始pcm语音path
    Returns: None
    """
    curr_dir = os.path.split(os.path.realpath(__file__))[0] # 当前运行文件所在目录
    sub_dir = os.path.join(curr_dir, os.path.split(pcm_file)[-1][:-4])
    creat_dir(sub_dir) # 以pcm_file的名字创建文件

    # 处理pitch_4cols_noloss.txt,pitch_5cols_noloss.txt文件
    pitch_4cols_noloss_txt = os.path.join(curr_dir, 'pitch_4cols_noloss.txt')
    pitch_5cols_noloss_txt = os.path.join(curr_dir, 'pitch_5cols_noloss.txt')
    pcm_dec_file = os.path.join(sub_dir, os.path.split(pcm_file)[-1][:-4] +'_dec_noloss.pcm')
    if os.path.exists(pitch_4cols_noloss_txt):
        os.remove(pitch_4cols_noloss_txt)
    if os.path.exists(pitch_5cols_noloss_txt):
        os.remove(pitch_5cols_noloss_txt)
    # generate pitch系数文件
    with os.popen(opus_no_loss_path + ' voip 16000 1 16000 ' + pcm_file + ' ' + pcm_dec_file) as pipe:
        str_config = pipe.read()
        # print(str_config) # 终端输出
    # shutil.move(pitch_4cols_noloss_txt, sub_dir)
    # shutil.move(pitch_5cols_noloss_txt, sub_dir)
    forceCopyFile(pcm_file, os.path.join(sub_dir, os.path.split(pcm_file)[-1]))
    forceCopyFile(pitch_4cols_noloss_txt, os.path.join(sub_dir, 'pitch_4cols_noloss.txt'))
    forceCopyFile(pitch_5cols_noloss_txt, os.path.join(sub_dir, 'pitch_5cols_noloss.txt'))
    os.remove(pitch_4cols_noloss_txt)
    os.remove(pitch_5cols_noloss_txt)
    return curr_dir, sub_dir, os.path.join(sub_dir, 'pitch_5cols_noloss.txt')


def decoder_loss(opus_demo_path, pcm_file, lost=10):
    """
    PS: 这个函数在decoder_no_loss之后运行
    对pcm_file解码，将解码的pcm_dec_file都放到<pcm_file>子目录下
    产生3个文件pitch_4cols.txt, pitch_5cols.txt, pitch_loss_flag.txt也放到<pcm_file>子目录下
    Args:
        opus_demo_path: opus_demo.exe所在的路径
        pcm_file: 原始pcm语音path
        lost(int): 丢包率(整数)  eg.  lost=40
    Returns: None
    """
    curr_dir = os.path.split(os.path.realpath(__file__))[0] # 当前运行文件所在目录
    sub_dir = os.path.join(curr_dir, os.path.split(pcm_file)[-1][:-4])
    if not os.path.exists(sub_dir):
        raise Exception('没有' + os.path.split(pcm_file)[-1][:-4] + ' 文件夹，请先运行decoder_no_loss()函数')
    pcm_dec_file = os.path.join(curr_dir, os.path.split(pcm_file)[-1][:-4] + '_dec_loss' + str(lost) + '.pcm')
    # 解码输出使用的是自带plc

    # 处理pitch_4cols.txt, pitch_5cols.txt, pitch_loss_flag.txt文件
    pitch_4cols_txt = os.path.join(curr_dir, 'pitch_4cols.txt')
    pitch_5cols_txt = os.path.join(curr_dir, 'pitch_5cols.txt')
    pitch_loss_flag_txt = os.path.join(curr_dir, 'pitch_loss_flag.txt')
    if os.path.exists(pitch_4cols_txt):
        os.remove(pitch_4cols_txt)
    if os.path.exists(pitch_5cols_txt):
        os.remove(pitch_5cols_txt)
    if os.path.exists(pitch_loss_flag_txt):
        os.remove(pitch_loss_flag_txt)
    # generate pitch系数文件
    with os.popen(opus_demo_path + ' voip 16000 1 16000 -loss ' + str(lost) + ' ' + pcm_file + ' ' + pcm_dec_file) as pipe:
        str_config = pipe.read()
        # print(str_config) # 终端输出
    forceCopyFile(pcm_dec_file, os.path.join(sub_dir, os.path.split(pcm_dec_file)[-1]))
    forceCopyFile(pitch_4cols_txt, os.path.join(sub_dir, 'pitch_4cols_loss_' + str(lost) + '.txt'))
    forceCopyFile(pitch_5cols_txt, os.path.join(sub_dir, 'pitch_5cols_loss_' + str(lost) + '.txt'))
    forceCopyFile(pitch_loss_flag_txt, os.path.join(sub_dir, 'pitch_lossflag_loss_' + str(lost) + '.txt'))
    os.remove(pitch_4cols_txt)
    os.remove(pitch_5cols_txt)
    os.remove(pitch_loss_flag_txt)
    return os.path.join(sub_dir, 'pitch_4cols_loss_' + str(lost) + '.txt'), \
           os.path.join(sub_dir, 'pitch_5cols_loss_' + str(lost) + '.txt'), \
           os.path.join(sub_dir, 'pitch_lossflag_loss_' + str(lost) + '.txt'), \
           pcm_dec_file

def combine_pitch_lost_txt(pitch_5cols_noloss_txt_path, pitch5cols_loss_txt_path):
    """
    预处理输入到rolling_prediction中的文件,前3行采用的是pitch_5cols_loss的，然后后面的采用的是no_loss数据
    但在第5列添加对应loss帧的标志位，返回处理后数据的路径
    Args:
        pitch_5cols_noloss_txt_path: noloss_data的路径，用于获取no_loss数据
        pitch5cols_loss_txt_path: loss_data的路径，用于获取lost_flag
    Returns:
    """
    tar_file_dir = os.path.split(pitch5cols_loss_txt_path)[0]
    tar_file_name = os.path.split(pitch5cols_loss_txt_path)[1][:-4] + '_with_noloss_data.txt'
    tar_file_path = os.path.join(tar_file_dir, tar_file_name)
    tar_file = open(tar_file_path, "w", encoding="utf8")
    with open(pitch5cols_loss_txt_path) as loss_src_file:
        frame_m = 0
        for line in loss_src_file:
            frame_m += 1
            if frame_m > 3:
                break
            tar_file.write(line)
    pitch_5cols_noloss_file = open(pitch_5cols_noloss_txt_path)
    pitch_5cols_noloss_file_data = pitch_5cols_noloss_file.readlines()
    pitch5cols_loss_file = open(pitch5cols_loss_txt_path)
    pitch5cols_loss_file_data = pitch5cols_loss_file.readlines()
    frame_m = 0
    for pitch_5cols_noloss_file_data_item, pitch5cols_loss_file_data_item in zip(pitch_5cols_noloss_file_data, pitch5cols_loss_file_data):
        pitch_5cols_noloss_file_data_item_split = pitch_5cols_noloss_file_data_item.strip().split()
        pitch5cols_loss_file_data_item_split = pitch5cols_loss_file_data_item.strip().split()
        new_item = pitch_5cols_noloss_file_data_item_split[:-1] + pitch5cols_loss_file_data_item_split[-1:]
        frame_m += 1
        if frame_m > 3:
            for l in range(len(new_item)):
                tar_file.write(new_item[l])
                tar_file.write("\t\t")
            tar_file.write("\n")
    # print(tar_file_path)
    return tar_file_path

def correcte_pitch5cols_lost_txt(pitch_5cols_loss_txt_path, pitch_5cols_noloss_txt_path):
    """
    对pitch_5cols_loss_txt_path的文件进行修正，去除其中非正常帧，比如no_loss帧有pitch的情况
    Args:
        pitch_5cols_loss_txt_path: 自带plc处理的数据,
        pitch_5cols_noloss_txt_path: no_loss的数据
    Returns:
    """
    tar_file_dir = os.path.split(pitch_5cols_loss_txt_path)[0]
    tar_file_name = os.path.split(pitch_5cols_loss_txt_path)[1][:-4] + '_correction.txt'
    tar_file_path = os.path.join(tar_file_dir, tar_file_name)
    tar_file = open(tar_file_path, "w", encoding="utf8")
    pitch_5cols_noloss_file = open(pitch_5cols_noloss_txt_path)
    pitch_5cols_noloss_file_data = pitch_5cols_noloss_file.readlines()
    pitch_5cols_loss_file = open(pitch_5cols_loss_txt_path)
    pitch_5cols_loss_file_data = pitch_5cols_loss_file.readlines()

    frame_m = 0
    for pitch_5cols_loss_file_data_item, pitch_5cols_noloss_file_data_item in zip(pitch_5cols_loss_file_data, pitch_5cols_noloss_file_data):
        pitch_5cols_loss_file_data_item_split = pitch_5cols_loss_file_data_item.strip().split()
        frame_m += 1
        if pitch_5cols_loss_file_data_item_split[-1] == '1':
            tar_file.write(pitch_5cols_loss_file_data_item)
        else:
            tar_file.write(pitch_5cols_noloss_file_data_item)

    return tar_file_path # 修正结果的路径

def correcte_pitch4cols_lost_txt(pitch_5cols_loss_txt_path, pitch_5cols_noloss_txt_path):
    """
    对pitch_5cols_loss_txt_path的文件进行修正，去除其中非正常帧，比如no_loss帧有pitch的情况
    Args:
        pitch_5cols_loss_txt_path: 自带plc处理的数据
        pitch_5cols_noloss_txt_path: no_loss的数据
    Returns: 输出结果是4列
    """
    tar_file_dir = os.path.split(pitch_5cols_loss_txt_path)[0]
    tar_file_name = os.path.split(pitch_5cols_loss_txt_path)[1][:-4] + '_correction4.txt'
    tar_file_path = os.path.join(tar_file_dir, tar_file_name)
    tar_file = open(tar_file_path, "w", encoding="utf8")
    pitch_5cols_noloss_file = open(pitch_5cols_noloss_txt_path)
    pitch_5cols_noloss_file_data = pitch_5cols_noloss_file.readlines()
    pitch_5cols_loss_file = open(pitch_5cols_loss_txt_path)
    pitch_5cols_loss_file_data = pitch_5cols_loss_file.readlines()

    frame_m = 0
    for pitch_5cols_loss_file_data_item, pitch_5cols_noloss_file_data_item in zip(pitch_5cols_loss_file_data, pitch_5cols_noloss_file_data):
        pitch_5cols_loss_file_data_item_split = pitch_5cols_loss_file_data_item.strip().split()
        pitch_5cols_noloss_file_data_item_split = pitch_5cols_noloss_file_data_item.strip().split()
        frame_m += 1
        if pitch_5cols_loss_file_data_item_split[-1] == '1':
            for ii in pitch_5cols_loss_file_data_item_split[:-1]:
                tar_file.write(ii + '\t\t')
            tar_file.write('\n')
        else:
            for ii in pitch_5cols_noloss_file_data_item_split[:-1]:
                tar_file.write(ii + '\t\t')
            tar_file.write('\n')
    return tar_file_path # 修正结果的路径

def rolling_prediction(pitch5cols_loss_txt_path, pitch4cols_predict_txt_path, enc, dec):
    """
    对单个语音pcm的pitch进行滚动预测,最开始的3帧使用opus默认plc
    Args:

        pitch5cols_loss_txt_path: no_loss语音pitch系数的txt文件，其中第5列是lost文件对应的lost_flag
        pitch4cols_predict_txt_path: 将预测好的系数，写到pitch4cols_predict_txt_path文件中供后续程序调用
        enc: seq2seq 的 encoder
        dec: seq2seq 的 decoder
    Returns: 总帧数,即文件的行数
    """
    # pitch5cols_loss_txt_path = "C:\Education\code\plc\seq2seq\opus_exe\pitch_5cols.txt"
    # pitch5cols_predict_txt_path = "C:\Education\code\plc\seq2seq\opus_exe\pitch_5cols_predict.txt"
    from collections import deque
    q = deque(maxlen=12)
    tar_file = open(pitch4cols_predict_txt_path, "w", encoding="utf8")
    with open(pitch5cols_loss_txt_path) as src_file:
        frame_m = 0 # 代表文件的行数
        for line in src_file:
            line = line.split()
            frame_m += 1
            if frame_m <= 3: # 如果是前3帧，则直接使用读取的文件
                q.extend(line[:-1])
                tar_file.write('\t\t'.join(line[:-1]))
                tar_file.write('\n')
            else: # 后面的帧进行滚动预测
                if line[-1] == '1':
                    assert len(q) == 12, "len (%d): for predict should be 12" % len(q)
                    new_line = [int(_) for _ in q]
                    predicted_frame = single_frame_predicte(enc, dec, new_line)

                    # 对输出结果长度不是4的情况进行处理,简单的将其最后一个值复制一下
                    last_value_of_predicted_frame = predicted_frame[-1]
                    if not len(predicted_frame) == 4:
                        for i in range(4):
                            predicted_frame.append(last_value_of_predicted_frame)
                    predicted_frame = predicted_frame[:4]
                    assert len(predicted_frame) == 4, "len: predicted should be 4" % len(q)
                    predicted_frame = [str(_) for _ in predicted_frame]
                    tar_file.write('\t\t'.join(predicted_frame))
                    tar_file.write('\n')

                    for i in range(4):
                        q.popleft()
                    q.extend(predicted_frame)
                else: # line[-1] = '0'
                    tar_file.write('\t\t'.join(line[:-1]))
                    tar_file.write('\n')
                    for i in range(4):
                        q.popleft()
                    q.extend(line[:-1])
    # print(frame_m)
    return frame_m

def rolling_preiction_context(pitch4cols_loss_txt_path, pitch_lossflag_txt_path, pitch4cols_preidct_txt_path, enc, dec):
    """
    使用1，2，4帧预测第三帧，对于多丢包的情况，使用自带plc
    Args:
        pitch4cols_loss_txt_path: loss语音pitch系数文件,4列,使用的loss系数是修正后的
        pitch_lossflag_txt_path: loss_flag_txt文件
        pitch4cols_preidct_txt_path:将预测好的系数，写到pitch4cols_preidct_txt_path中供后序调用(需要再来一个程序把这个和loss_flag拼接起来不然就看着太麻烦了)
        enc: seq2seq 的 encoder
        dec: seq2seq 的 decoder
    Returns: 总帧数(文件行数)
    """
    from collections import deque
    import copy
    window_q = deque(maxlen=4) # 前4帧的值
    flag_q = deque(maxlen=4) # 前4帧的flag
    tar_file = open(pitch4cols_preidct_txt_path, "w", encoding="utf8")
    src_4colsfile = open(pitch4cols_loss_txt_path)
    src_lines = src_4colsfile.readlines()
    flag_file = open(pitch_lossflag_txt_path)
    flag_lines = flag_file.readlines()
    frame_m = len(src_lines)

    # 对数据稍微处理
    for i in range(len(src_lines)):
        src_lines[i] = src_lines[i].strip().split()
        flag_lines[i] = flag_lines[i].strip().split()
    flag_lines = sum(flag_lines, []) # 将flag_lines展开
    train_X, train_Y = [], [] # train_X, train_Y分别为用于预测的124帧数据和，被预测的第3帧
    window_q, flag_q = deque(src_lines[:4]), deque(flag_lines[:4])
    for i in range(4, len(src_lines)):
        window_q.popleft()
        flag_q.popleft()
        window_q.append(src_lines[i])
        flag_q.append(flag_lines[i])
        if list(flag_q) == ['0','0','1','0']:
            train_X.append([window_q[i] for i in (0, 1, 3)]) # 用于预测的数据
            train_Y.append(window_q[2]) # 被预测项

            # 将用于预测的数据展开
            train_X_unfold = sum(train_X[-1], [])
            train_X_unfold = [int(_) for _ in train_X_unfold]
            # print(train_X_unfold)
            assert len(train_X_unfold) == 12, "len(%d): for plc predicted should be 12" % len(train_X_unfold)
            predicted_frame = single_frame_predicte_124to3(enc, dec, train_X_unfold)

            # 对输出结果长度不是4的情况进行处理,简单的将其最后一个值复制一下
            last_value_of_predicted_frame = predicted_frame[-1]
            if not len(predicted_frame) == 4:
                for i in range(4):
                    predicted_frame.append(last_value_of_predicted_frame)
            predicted_frame = predicted_frame[:4]
            predicted_frame = [str(_) for _ in predicted_frame]
            assert len(predicted_frame) == 4, "len(%d): predicted should be 4" % len(predicted_frame)

            src_lines[i-1] = predicted_frame

            window_q[-2] = predicted_frame
            flag_q[-2] = '0'
        elif flag_lines[i-1] == '1':
            flag_q[-2] = '0'
        else:
            pass

    for i in range(len(src_lines)):
        line = '\t\t'.join(src_lines[i])
        tar_file.write(line)
        tar_file.write('\n')
    return frame_m


def rolling_preiction_single_and_context(pitch4cols_loss_txt_path, pitch_lossflag_txt_path, pitch4cols_preidct_txt_path, enc_single, dec_single, enc_context, dec_context):
    """
    使用1，2，4帧预测第三帧，对于多丢包的情况，使用自带plc
    Args:
        pitch4cols_loss_txt_path: loss语音pitch系数文件,4列,使用的loss系数是修正后的
        pitch_lossflag_txt_path: loss_flag_txt文件
        pitch4cols_preidct_txt_path:将预测好的系数，写到pitch4cols_preidct_txt_path中供后序调用(需要再来一个程序把这个和loss_flag拼接起来不然就看着太麻烦了)
        enc_single: seq2seq 的 encoder 用于单向预测
        dec_single: seq2seq 的 decoder 用于单向预测
        enc_context: seq2seq 的 encoder 用于上下文预测
        dec_context: seq2seq 的 decoder 用于上下文预测
    Returns: 总帧数(文件行数)
    """
    from collections import deque
    window_q, flag_q = deque(maxlen=4), deque(maxlen=4) # 前4帧的pitch值 和 前4帧的flag
    tar_file = open(pitch4cols_preidct_txt_path, "w", encoding="utf8")
    src_4colsfile = open(pitch4cols_loss_txt_path)
    src_lines = src_4colsfile.readlines()
    flag_file = open(pitch_lossflag_txt_path)
    flag_lines = flag_file.readlines()
    frame_m = len(src_lines)

    # 对数据稍微处理
    for i in range(len(src_lines)):
        src_lines[i] = src_lines[i].strip().split()
        src_lines[i] = [int(_) for _ in src_lines[i]] # 把数据全部变为整数
        flag_lines[i] = flag_lines[i].strip().split()
    flag_lines = sum(flag_lines, [])  # 将flag_lines展开
    flag_lines = [int(_) for _ in flag_lines] # 把数据全部变为整数

    window_q, flag_q = deque(src_lines[:6]), deque(flag_lines[:6])
    for i in range(6, len(src_lines)):
        window_q.popleft()
        flag_q.popleft()
        window_q.append(src_lines[i])
        flag_q.append(flag_lines[i])

        if flag_q[-1] == 1: # 新添加进来的一帧是丢失的情况
            if window_q[-1] == [288, 288, 288, 288]:
                window_q[-1] = [288, 288, 288, 288]
                src_lines[i] = [288, 288, 288, 288]
            elif window_q[-3] == [288, 288, 288, 288] and flag_q[-2] == 0: # 此时针对的是队列-2帧是voice帧，且前面都是288
                j = 0
                while j < 4:
                    if src_lines[i - 2 - j] == [288, 288, 288, 288]:
                        window_q[-(3 + j)] = window_q[-2]
                        src_lines[i - 2 - j] = window_q[-2]
                    j += 1
            else:
                train_X = sum([window_q[i] for i in (3, 4)], []) # 将用于预测的两帧系数准备好并展开

                assert len(train_X) == 8, "len(%d): for plc predicted should be 8" % len(train_X)
                predicted_frame = single_frame_predicte_12to3(enc_single, dec_single, train_X)

                # 对输出结果长度不是4的情况进行处理,简单的将其最后一个值复制一下
                last_value_of_predicted_frame = predicted_frame[-1]
                if not len(predicted_frame) == 4:
                    for i in range(4):
                        predicted_frame.append(last_value_of_predicted_frame)
                predicted_frame = predicted_frame[:4]
                assert len(predicted_frame) == 4, "len(%d): predicted should be 4" % len(predicted_frame)

                window_q[-1] = predicted_frame
                src_lines[i] = predicted_frame
        elif flag_q[-1] == 0:
            if flag_q[-2] == 0 and window_q[-3] == [288, 288, 288, 288] and window_q[-1] != [0, 0, 0, 0]:
                # 此时的情况是长度为6的队列最后两帧是voice帧，队列前面部分全是288，针对【从unvoice开始连续丢包的情况】
                # 因为前面都是unvoice 所以激励比较小，所以pitch预测对了 可能作用不大，表现出来就是即便预测对了也没啥影响
                window_q.reverse()
                for j in range(2, 6):
                    train_X = [window_q[j] for j in (j - 2, j - 1)]
                    train_X = sum(train_X, [])
                    predicted = single_frame_predicte_12to3(enc_single, dec_single, train_X)
                    window_q[j] = predicted
                window_q.reverse()
                j = 0
                while j < 6:
                    if src_lines[i - j] == [288, 288, 288, 288]:
                        src_lines[i - j] = window_q[-(j + 1)]
                    j += 1
            elif flag_q[-2] == 1 and window_q[-2] != [288, 288, 288, 288] and window_q[-1] != [0, 0, 0, 0]:
                # 利用后一帧voice帧对前一帧进行修正
                context_X = sum([window_q[i] for i in (3, 5)], [])
                assert len(context_X) == 8, "len(%d): for plc predicted should be 8" % len(context_X)

                predicted_context  = single_frame_predicte_13to2(enc_context, dec_context, context_X)

                # 对输出结果长度不是4的情况进行处理,简单的将其最后一个值复制一下
                last_value_of_predicted_frame = predicted_context[-1]
                if not len(predicted_context) == 4:
                    for i in range(4):
                        predicted_context.append(last_value_of_predicted_frame)
                predicted_context = predicted_context[:4]
                assert len(predicted_context) == 4, "len(%d): predicted should be 4" % len(predicted_context)

                window_q[-2] = predicted_context
                src_lines[i - 1] = predicted_context

            elif flag_q[-2] == 1 and window_q[-2] == [288, 288, 288, 288] and window_q[-3] == [0, 0, 0, 0]:
                context_X = sum([window_q[i] for i in (3, 5)], [])
                assert len(context_X) == 8, "len(%d): for plc predicted should be 8" % len(context_X)

                predicted_context = single_frame_predicte_13to2(enc_context, dec_context, context_X)

                # 对输出结果长度不是4的情况进行处理,简单的将其最后一个值复制一下
                last_value_of_predicted_frame = predicted_context[-1]
                if not len(predicted_context) == 4:
                    for i in range(4):
                        predicted_context.append(last_value_of_predicted_frame)
                predicted_context = predicted_context[:4]
                assert len(predicted_context) == 4, "len(%d): predicted should be 4" % len(predicted_context)

                window_q[-2] = predicted_context
                src_lines[i - 1] = predicted_context
            else:
                pass
        else:
            pass

    # 对数据稍微处理，以写入文件
    for i in range(len(src_lines)):
        src_lines[i] = [str(_) for _ in src_lines[i]]  # 把数据全部变为整数
    for i in range(len(src_lines)):
        line = '\t\t'.join(src_lines[i])
        tar_file.write(line)
        tar_file.write('\n')
    return frame_m

def decoder_with_preidcted(opus_demo_with_pitch_and_lost_path,
                           pitch_m,
                           pitch_n,
                           loss_flag_path,
                           pitch_4cols_predict_path,
                           pcm_file,
                           pcm_dec_loss_file,
                           lost):
    """
    对pcm_file解码，解码时使用预测的系数, 将解码的pcm_dec_predicted_file都放到<pcm_file>子目录下
    Args:
        opus_demo_with_pitch_and_lost_path:opus_demo_with_pitch_and_lost.exe所在的路径
        pitch_m: 帧数,对应参数文件的行数
        pitch_n: 一帧多少数据,pitch为4
        loss_flag_path: 丢失帧的flag_txt文件
        pitch_4cols_predict_path: 预测的pitch系数的文件
        pcm_file: 原始pcm文件
        pcm_dec_loss_file: 丢帧的解码文件，主要作用是辅助命名, eg. Ch_f1_dec_loss10.pcm
        lost: 丢包率，这个必须加上，不然会对载入数据的结果产生影响，从而导致结果不准
    Returns:
    """
    curr_dir = os.path.split(os.path.realpath(__file__))[0] # 当前运行文件所在目录
    sub_dir = os.path.join(curr_dir, os.path.split(pcm_file)[-1][:-4])
    if not os.path.exists(sub_dir):
        raise Exception('没有' + os.path.split(pcm_file)[-1][:-4] + ' 文件夹，请先运行decoder_no_loss()函数')
    pcm_dec_with_predict_file = os.path.join(sub_dir, os.path.split(pcm_dec_loss_file)[-1][:-4] + '_predict.pcm')
    decode_cmd = opus_demo_with_pitch_and_lost_path + ' voip 16000 1 16000 -pitch_m ' + str(pitch_m) + ' -pitch_n ' + str(pitch_n) + ' -loss ' + str(lost) + ' ' + loss_flag_path + ' ' + pitch_4cols_predict_path + ' ' + pcm_file + ' ' + pcm_dec_with_predict_file
    with os.popen(decode_cmd) as pipe:
        str_config = pipe.read()
        # print(str_config) # 终端输出
    return pcm_dec_with_predict_file

def caculate_pesq(wbpesq_path, src_pcm, noisy_pcm, bit_rate):
    """
    Args:
        wbpesq_path: wbpesq.exe文件的路径
        src_pcm: 原始pcm
        noisy_pcm: 降级pcm
        bit_rate: 码率 eg. 16000
    Returns: 计算得到的pesq值
    """
    # src_pcm = "C:\Education\code\plc\seq2seq\opus_exe\Ch_f1.pcm"
    # noisy_pcm = "C:\Education\code\plc\seq2seq\opus_exe\Ch_f1_dec.pcm"

    with os.popen(wbpesq_path + ' +' + str(bit_rate) +' ' + src_pcm + ' ' + noisy_pcm) as pipe:
        str_config = pipe.read()
        # print(str_config) # wbpesq输出的完整信息
        pesq_re_compile = re.compile(r"PESQ_MOS = [\d\.]+")
        pesq_str = pesq_re_compile.findall(str_config)[0]
        pesq = re.findall(r"\d+\.?\d*", pesq_str)
        # print("&&&&&"*4)
        # print("%.3f" % float(pesq[0]))

    curr_dir = os.path.split(os.path.realpath(__file__))[0]  # 当前运行文件所在目录
    if os.path.exists(os.path.join(curr_dir, "_pesq_itu_results.txt")):
        os.remove(os.path.join(curr_dir, "_pesq_itu_results.txt"))
    if os.path.exists(os.path.join(curr_dir, "_pesq_results.txt")):
        os.remove(os.path.join(curr_dir, "_pesq_results.txt"))

    return float(pesq[0])


def predict_and_decoder(pcm_file_path, lost=10, enc=None, dec=None):
    """
    对单条pcm_file进行解码
    Args:
        pcm_file_path: pcm文件
        lost: 丢包率
        enc: encoder
        dec: decoder
    Returns: opus自带plc解码的pesq, predict得到的plc解码的pesq
    """
    curr_dir = os.path.split(os.path.realpath(__file__))[0]  # 当前运行文件所在目录
    root_dir = os.path.split(curr_dir)[:-1][0] # seq2seq目录
    opus_exe_dir = os.path.join(root_dir, 'opus_exe') # opus_exe/文件夹目录

    # 1 解码noloss的pcm，主要用于产生no_loss的参数
    opus_no_loss_exe = os.path.join(opus_exe_dir, 'opus_no_loss.exe')
    _, sub_dir, pitch_5cols_noloss_txt_path_ = decoder_no_loss(opus_no_loss_exe, pcm_file_path)

    # 2 解码lost的pcm，用于产生loss的参数
    opus_demo_exe = os.path.join(opus_exe_dir, 'opus_demo.exe')
    pitch4cols_txt_path, pitch5cols_txt_path, loss_flag_path, pcm_dec_loss_file = decoder_loss(opus_demo_exe, pcm_file_path, lost)

    # 3 对pitch参数进行修正，用于滚动预测
    correct_pitch4cols_lost_txt = correcte_pitch4cols_lost_txt(pitch5cols_txt_path, pitch_5cols_noloss_txt_path_)

    # 4 对lost的pitch参数滚动预测,输出预测的系数
    pitch4cols_predict_txt_path = pitch4cols_txt_path[:-4] + '_predict.txt'
    # frame_m = rolling_preiction_context(correct_pitch4cols_lost_txt, loss_flag_path, pitch4cols_predict_txt_path, enc, dec)
    frame_m = rolling_preiction_single_and_context(correct_pitch4cols_lost_txt, loss_flag_path, pitch4cols_predict_txt_path, encoder_single, decoder_single, encoder_context, decoder_context)


    # 5 使用预测系数解码
    opus_demo_with_pitch_and_lost_exe = os.path.join(opus_exe_dir, 'opus_demo_with_pitch_and_lost.exe')
    pitch_m = frame_m
    pitch_n = 4
    pcm_dec_with_predict_file = decoder_with_preidcted(opus_demo_with_pitch_and_lost_exe,
                           pitch_m,
                           pitch_n,
                           loss_flag_path,
                           pitch4cols_predict_txt_path,
                           pcm_file_path,
                           pcm_dec_loss_file,
                           lost)

    # 6 计算pesq指标
    wbpesq_exe = os.path.join(opus_exe_dir, 'wbpesq.exe')
    pesq_of_origin_opus = caculate_pesq(wbpesq_exe, pcm_file_path, pcm_dec_loss_file, bit_rate=16000)
    pesq_of_predicted = caculate_pesq(wbpesq_exe, pcm_file_path, pcm_dec_with_predict_file, bit_rate=16000)
    if pesq_of_origin_opus <= pesq_of_predicted:
        print("pesq of %s lost=%d in average >>> opus plc : %.3f, predict plc : %.3f  metric up" % (pcm_file_path, lost, pesq_of_origin_opus, pesq_of_predicted))
    else:
        print("pesq of %s lost=%d in average >>> opus plc : %.3f, predict plc : %.3f  metric down" % (pcm_file_path, lost, pesq_of_origin_opus, pesq_of_predicted))
    # logging.info("pesq of %s lost=%d in average >>> opus plc : %.3f, predict plc : %.3f" % (pcm_file_path, lost, pesq_of_origin_opus, pesq_of_predicted))
    # print('pesq of opus plc %.3f' % pesq_of_origin_opus)
    # print('pesq of predict plc %.3f' % pesq_of_predicted)
    return pesq_of_origin_opus, pesq_of_predicted



def predict_many(src_dir, lost=10, enc=None, dec=None):
    """
    对多条pcm文件进行解码
    Args:
        src_dir: 包含多条pcm文件的文件夹
        lost: 丢包率
        enc: encoder
        dec: decoder
    Returns:
    """
    pcm_files = os.listdir(src_dir)
    count = 0
    pesqs_of_opus = []
    pesqs_of_predict = []
    for pcm_file in pcm_files:
        count += 1
        pcm_path = os.path.join(src_dir, pcm_file)
        # print(pcm_path)
        pesq_of_origin_opus, pesq_of_predicted = predict_and_decoder(pcm_path, lost, enc, dec)
        # print("pesq of %s lost=%d in average >>> opus plc : %.3f, predict plc : %.3f" % (pcm_file, lost, pesq_of_origin_opus, pesq_of_predicted))
        pesqs_of_opus.append(pesq_of_origin_opus)
        pesqs_of_predict.append(pesq_of_predicted)
    pesqs_of_opus_average = sum(pesqs_of_opus) / count
    pesqs_of_predict_average = sum(pesqs_of_predict) / count
    print("pesq lost=%d in average >>> opus plc : %.3f, predict plc : %.3f" % (lost, pesqs_of_opus_average, pesqs_of_predict_average))

def clean():
    """
    清理文件夹下运行过程中产出的文件,主要是解码得到的pcm文件
    Returns: None
    """
    # 清理当前文件夹下的解码pcm文件
    print("clean files....")
    curr_dir = os.path.split(os.path.realpath(__file__))[0]  # 当前运行文件所在目录
    files = os.listdir(curr_dir)
    for file in files:
        if file[-4:] == '.pcm':
            # print(file)
            os.remove(os.path.join(curr_dir, file))

    # 清理opus_exe/文件夹下的解码pcm文件
    root_dir = os.path.split(curr_dir)[:-1][0]
    opus_exe_dir = os.path.join(root_dir, 'opus_exe')
    files = os.listdir(opus_exe_dir)
    for file in files:
        if file[-4:] == '.pcm':
            # print(file)
            os.remove(os.path.join(opus_exe_dir, file))


if __name__ == '__main__':
    # 初始化
    print("\nstart time:  ", time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time())))
    encoder_single, decoder_single = model_init_single_track()
    encoder_context, decoder_context = model_init_context()
    # src_list = [78, 78, 78, 78, 78, 78, 78, 78, 161, 158, 155, 152]
    # src_list2 = [106, 107, 107, 108, 108, 108, 107, 107, 106, 106, 107, 107]
    # src_list3 = [93, 94, 94, 95, 95, 95, 95, 95, 99, 100, 101, 102]
    # single_frame_predicte_124to3(encoder, decoder, src_list)
    # single_frame_predicte_124to3(encoder, decoder, src_list2)
    # single_frame_predicte_124to3(encoder, decoder, src_list3)
    # generate_pitch_Coef(20)

    # 测试pesq计算
    # wbpesq_path = "C:\Education\code\plc\seq2seq\opus_exe\wbpesq.exe"
    # src_pcm = "C:\Education\code\plc\seq2seq\hidden64_20200704\Ch_f6\Ch_f6.pcm"
    # noisy_pcm = "C:\Education\code\plc\seq2seq\hidden64_20200704\Ch_f6\Ch_f6_dec_loss10_predict.pcm"
    # res = caculate_pesq(wbpesq_path, src_pcm, noisy_pcm, 16000)
    # print(res)

    # 生成pitch_4cols.txt文件
    # generate_pitch_file_for_opus()

    # 对pitch_5cols_loss.txt进行修正
    # pitch_5cols_loss_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_5cols_loss_30.txt"
    # pitch_5cols_noloss_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_5cols_noloss.txt"
    # correcte_pitch5cols_lost_txt(pitch_5cols_loss_txt_path, pitch_5cols_noloss_txt_path)

    # 对pitch_5cols_loss.txt进行修正，修正的结果仅保存pitch值
    # pitch_5cols_loss_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_5cols_loss_30.txt"
    # pitch_5cols_noloss_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_5cols_noloss.txt"
    # correcte_pitch4cols_lost_txt(pitch_5cols_loss_txt_path, pitch_5cols_noloss_txt_path)

    # 滚动预测前对文件处理
    # pitch5cols_loss_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_5cols_loss_30.txt"
    # pitch_5cols_noloss_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_5cols_noloss.txt"
    # combine_pitch_lost_txt(pitch5cols_loss_txt_path, pitch_5cols_noloss_txt_path)

    # 测试滚动预测
    # pitch5cols_loss_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_m1\pitch_5cols_loss_20_with_noloss_data.txt"
    # pitch4cols_predict_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_m1\pitch_4cols_loss_20_predict.txt"
    # rolling_prediction(pitch5cols_loss_txt_path, pitch4cols_predict_txt_path, encoder, decoder)


    # 测试上下文的滚动预测
    # pitch4cols_noloss_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_4cols_noloss.txt"
    # pitch4cols_loss_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_5cols_loss_30_correction4.txt"
    # pitch_lossflag_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_lossflag_loss_30.txt"
    # pitch4cols_preidct_txt_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_4cols_loss_30_context_predict.txt"
    # rolling_preiction_context(pitch4cols_noloss_txt_path, pitch4cols_loss_txt_path, pitch_lossflag_txt_path, pitch4cols_preidct_txt_path, encoder, decoder)

    # 测试单向预测+上下文的滚动预测相结合的方式
    # pitch4cols_loss_txt_path = "C:\Education\code\plc\seq2seq\hidden_20200715\Ch_f1\pitch_5cols_loss_30_correction4.txt"
    # pitch_lossflag_txt_path = "C:\Education\code\plc\seq2seq\hidden_20200715\Ch_f1\pitch_lossflag_loss_30.txt"
    # pitch4cols_preidct_txt_path = "C:\Education\code\plc\seq2seq\hidden_20200715\Ch_f1\pitch_4cols_loss_30_context_predict.txt"
    # rolling_preiction_single_and_context(pitch4cols_loss_txt_path, pitch_lossflag_txt_path, pitch4cols_preidct_txt_path, encoder_single, decoder_single, encoder_context, decoder_context)

    # 测试解码流程
    # opus_no_loss_path = "C:\Education\code\plc\seq2seq\opus_exe\opus_no_loss.exe"
    # pcm_file = "C:\Education\code\plc\dataset\denny_pcm\Ch_f1.pcm"
    # decoder_no_loss(opus_no_loss_path, pcm_file)
    #
    #
    # # 测试opus_demo.exe运行
    # opus_demo_path = "C:\Education\code\plc\seq2seq\opus_exe\opus_demo.exe"
    # pcm_file = "C:\Education\code\plc\dataset\denny_pcm\Ch_f1.pcm"
    # decoder_loss(opus_demo_path, pcm_file,lost=20)


    # 使用预测系数来解码
    # opus_demo_with_pitch_and_lost_path = "C:\Education\code\plc\seq2seq\opus_exe\opus_demo_with_pitch_and_lost.exe"
    # pcm_file = "C:\Education\code\plc\dataset\denny_pcm\Ch_f1.pcm"
    # pitch_m = 383
    # pitch_n = 4
    # loss = 10
    # loss_flag_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_lossflag_loss_10.txt"
    # pitch_4cols_predict_path = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\pitch_4cols_loss_10_predict.txt"
    # pcm_dec_loss_file = "C:\Education\code\plc\seq2seq\hidden256_20200704\Ch_f1\Ch_f1_dec_loss10.pcm"
    # decoder_with_preidcted(opus_demo_with_pitch_and_lost_path,
    #                        pitch_m,
    #                        pitch_n,
    #                        loss_flag_path,
    #                        pitch_4cols_predict_path,
    #                        pcm_file,
    #                        pcm_dec_loss_file,
    #                        lost=loss)

    # 使用预测系数来解码-上下文信息
    # opus_demo_with_pitch_and_lost_path = "C:\Education\code\plc\seq2seq\opus_exe\opus_demo_with_pitch_and_lost.exe"
    # pcm_file = "C:\Education\code\plc\dataset\denny_pcm\Ch_f1.pcm"
    # pitch_m = 383
    # pitch_n = 4
    # loss_flag_path = "C:\Education\code\plc\seq2seq\hidden_20200715\Ch_f1\pitch_lossflag_loss_30.txt"
    # pitch_4cols_predict_path = "C:\Education\code\plc\seq2seq\hidden_20200715\Ch_f1\pitch_4cols_loss_30_predict.txt"
    # pcm_dec_loss_file = "C:\Education\code\plc\seq2seq\hidden_20200715\Ch_f1\Ch_f1_dec_loss30.pcm"
    # decoder_with_preidcted(opus_demo_with_pitch_and_lost_path,
    #                        pitch_m,
    #                        pitch_n,
    #                        loss_flag_path,
    #                        pitch_4cols_predict_path,
    #                        pcm_file,
    #                        pcm_dec_loss_file,
    #                        lost=30)

    # 测试解码单个文件
    # pcm_file = "C:\Education\code\plc\dataset\denny_pcm\Ch_f1.pcm"
    # predict_and_decoder(pcm_file_path=pcm_file, lost=30, enc=encoder, dec=decoder)


    # 测试多个文件
    print("use seq2seq model trained 2020-07-17 20:00")
    src_dir = r"C:\Education\code\plc\dataset\denny_pcm"
    predict_many(src_dir, lost=5, enc=None, dec=None)
    predict_many(src_dir, lost=10, enc=None, dec=None)
    predict_many(src_dir, lost=20, enc=None, dec=None)
    predict_many(src_dir, lost=30, enc=None, dec=None)

    # 适当的清理工作
    clean()
    print("end time:  ", time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time())))


