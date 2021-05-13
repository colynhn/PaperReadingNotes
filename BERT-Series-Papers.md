# 2021-4-30

**待定**

**#Tag**


**#Knowledge Points**


**#Innovation**

**#Question**



# 2021-2-26

**ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS（2019）**

**#Tag**

    ALBERT

**#Knowledge Points**

    本论文主要解决两个问题：降低内存占用、提高bert的速度 and 使用自监督loss，focus下游NLP任务的句子之间的关系
    
    Transformer的权重共享，Transformer在哪里做了权重共享：

    在翻译任务中：
        
    （1）Encoder和Decoder间的Embedding层权重共享
        
        公用一张词表，以及一些subword的共用（subword在中文中没有意义）
                        
    （2）Decoder中Embedding层和FC层权重共享
                    
        Embedding层可以说是通过onehot去取到对应的embedding向量，FC层可以说是相反的，通过向量（定义为 x）去得到它可能是某个词的softmax概率，取概率最大（贪婪情况下）的作为预测值
            
        Embedding层参数维度是：(v,d)，FC层参数维度是：(d,v)，可以直接共享嘛，还是要转置？（实际使用的时候发现不需要进行转置可直接共享）其中v是词表大小，d是embedding维度

**#Innovation**

    1 Factorized embedding parameterization 因式分解嵌入参数化
    
      embedding在学习上下文无关表示，而hidden layer学习上下文有关表示，但是bert中是 E=H，albert将其拆分
      
      O(V × H) to O(V × E + E × H) 主要还是针对multi-head attention 中input的维度d_model和output的维度d_model是一样的
    
    2 Cross-layer parameter sharing 跨层参数共享
    
      所有层的参数共享（减少参数，并且根据input和output的 embedding vector的L2和cosin distance发现在层数之间更加平稳）
    
    3 Inter-sentence coherence loss 相关句子loss

**#Question**

    无


# 2021-2-19

**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（2018）**

**#Tag**

    BERT

**#Knowledge Points**

    BERT：Bidirectional Encoder Representations from Transformers

    train from scratch：从头训练学习

    将模型训练拉入 pre-training + fine-tune阶段

    pre-trained language model 分为两种：feature-based 和 fine-tuning

    不管是ELMO还是openAI GPT，都是单向的语言模型（基于传统语言模型假设）

**#Innovation**

    (1) Masked LM
    
    (2) Next Sentence Prediction (NSP)
    
    (3) using Transformer Encoder (可实现并行化)
    
    (4) 将NLP任务拉入 pre-train + fine-tune 时代

**#Question**
    
    BERT中的WordPiece如何做
    
    WordPiece字面理解是把word拆成piece一片一片；使用BPE（Byte-Pair Encoding）双字节编码(WordPiece中的一种)，即生成subword
    
    可参考 https://www.cnblogs.com/huangyc/p/10223075.html 生成过程，但是中文一般就是使用字
    
    BERT中fine-tune就进行该如何进行（实际操作）(doing)

