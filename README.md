# PaperReadingWeekly




BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
# 2021-3-12

**tag**
**Knowledge Points**
**Innovation**
**Question**

# 2021-3-5

**tag**
**Knowledge Points**
**Innovation**
**Question**

# 2021-2-26

**tag**
**Knowledge Points**
**Innovation**
**Question**

# 2021-2-19

**tag**
**Knowledge Points**
**Innovation**
**Question**


# 2021-2-12

**tag**
**Knowledge Points**
**Innovation**
**Question**

# 2021-2-5

**tag**
**Knowledge Points**
**Innovation**
**Question**

# 2021-1-29

**tag**
**Knowledge Points**
**Innovation**
**Question**

# 2021-1-22

**tag**
**Knowledge Points**
**Innovation**
**Question**

# 2021-1-15

**tag**
**Knowledge Points**
**Innovation**
**Question**

# 2021-1-8

**Transformer-Transducer: End-to-End Speech Recognition with Self-Attention (2019 Facebook)**

**tag**

ASR

**Knowledge Points**

无

**Innovation**

利用transformer替换RNN-T的encoder中的RNN，便于并行计算

利用convolutional approaches代替transformer本身的位置信息(position encoding)，防止sequense的时序错乱问题

利用truncated self-attention 代替transformer中原有的multi-head attention，实现streamable read and 识别

以上三点为wer值和其他性能的一种这种折中 Language model

**Question**

# 2021-1-1

**CTC: Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks (2006)**

**Tag**

ASR

**Knowledge Points**

#CTC loss仅仅是深度学习训练时的loss表示，解决序列化问题（如序列标注），目的在于实现input和output的alignment; 在loss目标函数之前还是要自己进行网络的搭建，如RNN、CNN等

#CTC decoder在进行decoder时，用到类似于HMM中的前后向算法，计算score; 插入空白标签，便于解码

#在解码时加入language model，有助于解码正确率的提高

**Innovation**

无

**Question**

无


