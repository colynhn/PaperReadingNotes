# PaperReadingWeekly




BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
# 2021-3-12

**#Tag**
**#Knowledge Points**
**#Innovation**
**#Question**

# 2021-3-5

**#Tag**
**#Knowledge Points**
**#Innovation**
**#Question**

# 2021-2-26

**#Tag**
**#Knowledge Points**
**#Innovation**
**#Question**

# 2021-2-19

**#Tag**
**#Knowledge Points**
**#Innovation**
**#Question**


# 2021-2-12

**#Tag**
**#Knowledge Points**
**#Innovation**
**#Question**

# 2021-2-5

**#Tag**
**#Knowledge Points**
**#Innovation**
**#Question**

# 2021-1-29

**Evaluating Approaches to Personalizing Language Models（2020）**

**#Tag**

ASR

**#Knowledge Points**

（1）（The assumption being made is that people of similar demographics tend to write similarly）即相似社会属性关系的人说话习惯也差不多（如性别、年纪、性格等）

（2） 个性化语言模型也是一个Domain adaptation problem

（3） Perplexity不能用于评价不同词典的model，因为小的词典的语言模型就会替换更多的词为<UNK>，会严重影响最终的结果
  
（4） 拿到一个深度学习网络，如何构建模型：模型种类 + loss func + 超参数

（5） 使用的语言模型依然是n-gram和rnnLM（LSTM LM）较多，也可用在纠错中


**#Innovation**

介绍三种语言模型个性化的方法：

（1）个性化数据模型和老模型做插值融合

（2）使用个性化数据对老模型进行参数调节

（3）add data 继续 training


**#Question**

无


# 2021-1-22

**On the Comparison of Popular End-to-End Models for Large Scale Speech Recognition（2020）**

**#Tag**

ASR

**#Knowledge Points**

end2end主流的DL框架:

（1）CTC (Connectionist Tem- poral Classification)

（2）RNN-T (可以看作是CTC的一种改进)

（3）RNN-AED（RNN-Attention encoder-decoder, LAS model的不同说法）

（4）Transformer-AED （best-performance）

其中：AED:  Attention-based Encoder- Decoder (AED)

在线和离线的各模型对比（在原论文中可查阅）

DL中的正则化方法：
（1）BN：Batch Normalization          纵向规范化（使用较多）

（2）LN：Layer Normalization           横向规范化（使用较多）

（3）WN： Weight Normalization      参数规范化

（4）CN： Cosine Normalization        余弦规范化


**#Innovation**

无

**#Question**

无


# 2021-1-15

**Estimation of gap between current language models and human performance（2017）**

**#Tag**

ASR

**#Knowledge Points**

无

**#Innovation**

文章主要对比了LM中的 ngram model、ME model、RNN with ME model & RNN/LMSM NNLM四种model的效果以及和人类的水平的对比实验（采用众包方式）

**#Question**

无

# 2021-1-8

**Transformer-Transducer: End-to-End Speech Recognition with Self-Attention (2019 Facebook)**

**#Tag**

ASR

**#Knowledge Points**

无

**#Innovation**

利用transformer替换RNN-T的encoder中的RNN，便于并行计算

利用convolutional approaches代替transformer本身的位置信息(position encoding)，防止sequense的时序错乱问题

利用truncated self-attention 代替transformer中原有的multi-head attention，实现streamable read and 识别

以上三点为wer值和其他性能的一种这种折中 Language model

对比 based on RNN-T model, 而RNN-T可以看作是CTC的改进版，不是一个input对应一个output，而是一个input对应多个output( 即复制encoder的outputs tokens进行decoder)
即为encoder-decoder结构；缺点：RNNs are difficult to compute in parallel

相较于RNNs的优点：

Compared with RNNs：the attention mechanism is non-recurrent and can compute in parallel easily; the attention mechanism can ”attend” to longer contexts explicitly

CTC假设每一个输出的token都是独立的，而RNN-T正是改善了这一点，即通过增加前一个非空 output token的输入

RNN-T其中均是每一帧input vector在经过decoder后的output token上操作， 即属于encoder-decoder framework

在没有LM的情况下，RNN-T优于CTC

**#Question**

无

# 2021-1-1

**CTC: Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks (2006)**

**#Tag**

ASR

**#Knowledge Points**

CTC loss仅仅是深度学习训练时的loss表示，解决序列化问题（如序列标注），目的在于实现input和output的alignment; 在loss目标函数之前还是要自己进行网络的搭建，如RNN、CNN等

CTC decoder在进行decoder时，用到类似于HMM中的前后向算法，计算score; 插入空白标签，便于解码

在解码时加入language model，有助于解码正确率的提高

**#Innovation**

无

**#Question**

无


