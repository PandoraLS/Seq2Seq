# Seq2Seq
Seq2Seq模型demo

文件结构


eng-fra-val.txt是从eng-fra.txt中随机选了2000条语句
fra-eng.txt与eng-fra.txt对应，不过英法的顺序变了

## TODO
print添加log功能，每次运行都新建一个log文件
inference.py需要进一步提升
添加log功能

seq2seq中是以iter来训练的，需要将save_checkoutpoint修改为以iter为准

当前模型先按照epoch来训练和保存，没必要按照iter进行
self.train_dataloader非常难用，且臃肿不堪，需要重构

ref_code/中是参考的各种代码


## 训练模型使用的命令
```shell script
python train.py
tensorboard --logdir=runs
```

