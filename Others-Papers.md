# Paper1

**Cold-Start and Interpretability: Turning Regular Expressions into Trainable Recurrent Neural Networks（EMNLP 2020）**

**#Tag**

    Regular-expression、FA-RNNs

**#Knowledge Points**

    （1）这篇论文给人眼前一亮的感觉
    
        神经网络缺乏可解释性，需要大量的labeled data去train；基于规则方法（如正则表达式）不需要data进行训练，并且在大数据量的情况下不能运用数据，效果也不如大数据量下的NN
    
    （2）针对规则和NN相结合的研究
    
        1）使用规则去约束模型
        
           knowledge distillation、multi-task learning
        
        2）设置新的架构（参考规则系统）

**#Innovation**

    （1）FA-RNNs（finite-automaton recurrent neural networks）：上述基于规则和NN结合，既可以冷启动，也可以从数据中train（本文中的方法）。并且REs和FA-RNNs可以互相转化
    
        论文其中有句话：REs和有限状态机的等价性？（可以参考原论文Table 1 for answers）
     
    （2）在WFA上进行前向算法/viterbi算法（类似于语音识别解码器），计算出score最大的那条路径
    
    （3）利用前向算法的通式和rnn的循环做类比

**#Question**

    根据本论文的思路，是不是可以将神经网络的过程理解为一个寻找某条路径，打分的过程呢？
