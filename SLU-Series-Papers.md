# Paper11

**A Survey on Spoken Language Understanding: Recent Advances and New Frontiers（2021）**

**#Tag**

    SLU

**#Knowledge Points**

    半个月前发表的论文，算是比较新的关于SLU的survey，特别是里边的对于SLU的研究论文以及代码实现的整理，对SLU的无论是初学者、研究者还是从业者，都很有帮助
    
    代码仓库：https://github.com/yizhen20133868/Awesome-SLU-Survey
    
    评价指标（也算是解答了2021-3-14中提到的accury的问题）
    
    工程中的多轮模型一般在论文中是以 Contextual SLU 表示

**#Innovation**

    本文就是一篇最新的研究总结：包括对现有研究的分类，整理；对将来研究挑战的陈述

    将SLU的研究重新分类
    
    （1） single model vs. joint model
    
    （2） implicit joint model vs. explicit joint model（single flow interaction and bidirectional flow interaction）
          
          single flow interaction：一般是使用Intent信息指导Slot filling
          
          bidirectional flow interaction：Intent信息和Slot filling相互影响
          
          也是在优化SLU时，经常做的一件事情（另外，在rnn/bert的output中接CRF一般会有很好的效果，可以参考2021-3-14论文）
      
    （3） non pre-trained models vs. pre-trained models
    
    从本文总结可以得出，pre-trained models 在ID和SF的指标中，均占到了很大的优势，主要是因为pre-trained model比如bert在预训练时很好的学习到了句子以及词语直接的潜在表示（类似于以前在ML中所做的特征工程）
    
    但是，在2016年使用Attention的RNN化石级模型仍然是具有很强的Intent Acc和Slot F1的，也不能忽视
    
    
    将来的挑战和思考
    
    （1）Contextual SLU
    
        不同对话的历史信息（back-and-forth conversations）与当前对话的相关性
        
        对话过程中的长距离问题
     
    （2）Multi-Intent SLU
    
        如何整合多intent的问题
        
        数据lack问题
    
    （3）Chinese SLU
    
         在多个分词标准下，好好利用分词信息去（不仅仅是char级别）指导ID&SF（中文SLU）
         
         其中中文的开源数据集（CAIS）可在 CM-net: A novel collaborative memory network for spoken language understanding中寻找
    
    （4）Cross Domain SLU
    
        在数据lack的情况下，如何快速适应新的domain
        
        解答：在这里，本人解答一下这个问题，两个层面：关于跨domain问题，从model的角度可以使用增量训练去解决；也可将该问题下方到DST层面，将DST问题看作QA问题，去解新domain的问题
        
        而作者主要是从模型提取domain-shared feature角度解答这个问题（但是里边也涉及到知识迁移也就是迁移学习的问题）
        
        Zero-shot Setting：（也就是工业中的冷启动问题，没有任务训练数据，仍然可以将该问题下方到DST去解决）
    
    （5）Cross-Lingual SLU
    
        注释：An SLU system trained on En- glish can be directly applied to other low-resource languages
        
        其中具有语言之间的对齐问题（可能涉及到翻译中的相关技术，也可以参考语音识别中的相关技术，因为语音识别不涉及两种语言的问题，而是两类X，Y之间的转换，类似于语言的一种抽象）
    
    （6）Low-resource SLU
    
        包括 Few-shot SLU, Zero-shot SLU, and Unsupervised SLU，一般像这种问题都可以归结为工业上的冷启动问题，而对于冷地冻问题，均可以使用规则 or ML（CRF、HMM）等尝试
        
        关于 Unsupervised SLU，对工业界是很有帮助的；可参考 Dialogue state induction using neural latent variable models获取更多知识，主要还是更多以Intent和Slot的内在联系出发
    
**#Question**

      针对对话过程中的长距离对话问题，能否参考LSTM在解决长距离以来或者self-attention的长距离以来问题，将token的级别扩展到snetence去解决呢？

    
# Paper10

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
    
    （4）Adding CRF for modeling slot label dependencies, on top of the joint BERT model
    
        可参考 End-to-end learning of semantic role labeling using recurrent neural networks(ACL 2015)
        
    （5）Joint训练会比非Joint训练效果差

**#Question**

    （1）如果使用全部sub-word，并且一个word的概率等于所有sub-word的概率乘积，是否效果会好一点儿？（待验证，可能效果一般）
    
    （2）sentence-level semantic frame accuracy 评价指标如何定义 ？

# Paper9

**Learning Task-Oriented Dialog with Neural Network Methods（2018）**

**#Tag**

    SLU

**#Knowledge Points**

    无    

**#Innovation**

    CMU的一篇博士论文，如果想要了解2018年之前的SLU的研究现状，是值得细细阅读的一篇好论文

**#Question**

    无


# Paper8

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
