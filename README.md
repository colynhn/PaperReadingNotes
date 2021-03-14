# PaperReadingWeekly

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  Reading notes of some recent papers on NLP, ASR, ML, etc.

  Using command + F you can find the Tag that you want.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# 2021-3-12

**BERT for Joint Intent Classification and Slot Filling（2019）**

**#Tag**

    SLU

**#Knowledge Points**

    NLU Models 大致可以分为两类：Independent modeling approaches、Joint modeling approaches
    
    Independent modeling approaches
    
    (1) Intent classification
    
        CNN :  Convolutional neural networks for sentence classification(EMNLP 2014)
               
               Character-level convolutional networks for text classification(NIPS 2015)
        
        LSTM : Recurrent neural network and LSTM models for lexical utterance classification(INTERSPEECH 2015)
        
        Attention Based CNN : Attention based convolutional neural networks for sentence classification(Interspeech 2016)
        
        Hierarchical Attention Networks : Hierarchical attention networks for document classification(NAACL HLT 2016)
        
        Adversarial Multi-task Learning : Adversarial multi-task learning for text classification(ACL 2017)
    
    (2) Slot filling
    
        CNN : Sequential convolutional neu- ral networks for slot filling in spoken language un- derstanding(Interspeech 2016)
        
        LSTM : Spoken lan- guage understanding using long short-term memory neural networks(2014 IEEE)
        
        RNN-EM : Recurrent neural networks with ex- ternal memory for spoken language understanding(NLPCC 2015)
        
        Encoder-labeler Deep LSTM : Leveraging sentence-level information with encoder LSTM for natural language understanding(2016)
        
        Joint Pointer and Attention : Improving slot filling in spoken language understanding with joint pointer and attention(ACL 2018)
    
    Joint modeling approaches
    
        CNN-CRF: Convolutional neural network based triangular CRF for joint in- tent detection and slot filling(2013 IEEE)
        
        RecNN : Joint semantic utterance classification and slot filling with recursive neural networks(2014 IEEE)
        
        Joint RNN-LSTM : Multi-domain joint semantic frame parsing using bi-directional RNN-LSTM（Interspeech 2016）
        
        Attention-based BiRNN : Attention-based recurrent neural network models for joint intent detection and slot filling(Interspeech 2016)(done 见 # 2021-3-5)
        
        Slot-gated Attention-based Model :  Slot-gated modeling for joint slot filling and intent prediction(NAACL-HLT 2018)
        

**#Innovation**

    （1）WordPiece 的处理: 为了适应WordPiece，在slot filling的序列标注任务中，使用子词的第一个标记的隐藏状态的output过softmax进行预测

    （2）不用全部WordPiece的原因：（个人见解）为了对齐，这个对齐不是指input和output的对齐，是为了和句子的字级别slot对齐，
    
        1）不管在汉语还是英语中，往往一个子词是没有什么独立意义的，仅仅代表辅助含意作用
    
        2）在Tansformer的多层encode，也就是bert中，slef-attention已经提取了足够的sub-word的信息，可以获得足够的表征信息；于为什么[CLS]的位置放前放后都可以的原理类似

    （3）由于 Intent Detection 和 Slot Filling的任务关联性，所以采取两个目标函数的乘积形式进行联合优化

**#Question**

    如果使用全部sub-word，并且一个word的概率等于所有sub-word的概率乘积，是否效果会好一点儿？（待验证）

# 2021-3-12

**Learning Task-Oriented Dialog with Neural Network Methods（2018）**

**#Tag**

    SLU

**#Knowledge Points**

    无    

**#Innovation**

    CMU的一篇博士论文，如果想要了解2018年之前的SLU的研究现状，是值得细细阅读的一篇好论文

**#Question**

    无


# 2021-3-5

**Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling（2016）**

**#Tag**

    SLU、Intent Detection、Slot Filing

**#Knowledge Points**

    intent detection：意图检测 可以视为语义分类任务 ，如SVMs、NN

    slot filling：槽位填充 可以视为序列标注任务（NER，命名实体识别）： maximum entropy Markov models (MEMMs)、CRFs、RNNs

    semantic：语义的

    slot filling 的对齐是显式的，如rnn可以做到对齐，单纯的slot filling任务不需要额外的对齐手段（对齐一般存在于input 和 output不一样长的情况）

    以往的joint的方案是 使用一个model train，然后去fine-tune适应两个任务

**#Innovation**

    本文所探讨
    
        本文主要讨论如何在encoder-decoder的nn中去应用已知的slot filling中的对齐信息；

        如何利用encoder-decoder的attention的机制去改善slot-filling的model；

        如何将intent detection和slot filling结合去进行joint优化
    
    模型
    
        将传统的bi-rnn加入attention进行比较，并且是将intent detection 和 slot filling进行joint训练在bi-rnn的最后一个output进行copy，一部分用于intent detection，一部分进行slot filling的decoder的initial hidden input，借助encoder的hidden outputs去进行alignment，以及attention的context信息去预测下一个label，整体decoder就是一个序列标注问题

    效果
    
        在2016年，在ATIS数据集上打到了SOTA的效果，其中intent detection为2%-2.5%的error rate以及slot filling为95.5%-96%（仁者见仁，智者见智）


**#Question**

    无

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


# 2021-2-12

**LSTM Language Models for LVCSR in First-Pass Decoding and Lattice-Rescoring（2019）**

**#Tag**

    ASR

**#Knowledge Points**

    无

**#Innovation**

    主要是在big dataset、big vocabulary的条件下，通过使用char CNN + 改进的softmax去达到加速的目的，并且比较了KN n-gram模型和我RNNLM（LSTM）以及不同units的效果；得出结论ensemble的效果最好

**#Question**

    无


# 2021-2-5

**Scalable Multi Corpora Neural Language Models for ASR（2019 Amazon Alexa）**


**#Tag**

    ASR

**#Knowledge Points**

    LM：language model.   NLM: Neural language model.

    由于NLM有不受限制的context，所以在进行decode的时候就会产生指数增加的计算量

    第一种做法：先使用n-gram语言模型进行一次解码，然后再用NLM去再lattice上进行二次解码（缺点是在进行一个解码的时候，二次解码一直在等待，时间是累加的；而且在一边解码的ngram可以会损失一些假设）

    第二种做法：即本文的优化

**#Innovation**

    本文所解决的问题：

    1 异构数据适应性训练问题 
  
      （1）Data mixing
          
          首先将不同domain的数据使用ngram模型进行训练，线性插值，求出权重
      
          再根据权重从不同domain中抽取训练数据加入到mini-batch中进行训练
  
      （2）Transfer learning through fine tuning
      
          使用非领域数据进行NLM的训练，然后再使用领域数据进行参数调节
  
      （3）两者结合的方式
  
          The model is first pre-trained on the out-of-domain data, and the data mixing strategy is used during the fine tuning stage
  
    2 模型快速推理的解决方案

      （1）Self-normalized models 
      
          权重量化等（不理解，等用到再进一步了解）
      
          https://www.gtcevent.cn/session-catalog/ 参考即可，是一种模型加速的一种方式； 比如：基于 Tensor Core 的 CNN INT8 定点训练加速
      
  
      （2）Post-training Quantization
  
    3 Generating synthetic data for first-pass LM （并没有感觉到好在哪儿？？？）

      从训练好的NLM中去抽取sub-words，作为ngram的值，再去进行一遍解码
  
    实验

      首先，对于out of domain dataset pre-train + mixture（out of domain + in domain） 效果最好，PPL最低

      小的知识点：reference 参考（也就是真实标签，即标注文本序列）
          
                hypothesis 假说/假定（也就是模型识别出来的文本，即预测文本序列）

    概述：（不一定准确）

    1 寻求最优的线性加权参数lamda
  
      minimize perplexity：lamda * in-doamin_n-gram_mdoel+(1-lamda) * out-of-domain_n-gram_model 
  
    2 在上述的基础上（存在疑问），分配给各domain数据集score（表示关联性），根据score比例决定贡献给train-data比例，解决数据稀疏性问题

    3 如何把神经网络语言模型在不增加明显时延的情况下，合并到带有n-gram语言模型的ASR系统中：

     （1）首先将传入的数据通过具有常规n-gram语言模型的语音识别器传递，然后使用神经模型完善第一个模型的假设
  
     （2）two-pass approach：first pass已经将概率数目降下来了，所以second pass就可以仅仅只考虑降下来之后的这部分概率（使用噪声对比估计进行训练）
  
    （3）使用噪声对比估计可以将基于词典的多维（比如词典是百万，需要百万级别）的概率选择计算转化为二分类

**#Question**

    无


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

    Compared with RNNs：the attention mechanism is non-recurrent and can compute in parallel easily; the attention mechanism can ”attend” to longer contexts       explicitly

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


