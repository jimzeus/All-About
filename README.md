关于神经网络的一切（迫真）

All About Neural Network（AANN）

2021.04.25

Coldwind

jimzeus\@gmail.com

前言

本文是作者最近一段时间学习**神经网络**的笔记，部分内容原创，部分内容来自网上，部分则是对他人文章和论文的总结，有参考到的内容尽量给出了连接，如果忘了给出欢迎指出来。

**它不是什么？**

-   它不是一篇《从入门到精通》那一类一步步带领你学习深度学习的书，这一类书已经不少，并且都是专业人士所著。

-   它不是针对某一个知识点进行详细解说的文章，如果想了解其中某个具体的细节，可以去网上自行了解。

    **它是什么？**

    本文的目录结构可以让你知道NN：

-   包括哪些知识领域

-   各个知识领域包括哪些知识点

-   知识点之间是什么关系

    总之，这是个类似关于神经网络的知识点结构介绍的文章，有点像字典，并**不建议**以本文按顺序学习神经网络。

**为什么要写这个？**

作者记性不好，学过了解的过的东西经常会忘，下次碰到了又得重新学习一遍，为了减少回忆的时间，于是会将自己的理解进行文档化（俗称写笔记）。

**为什么要po出来？**

作者工作期间有过几次大规模系统学习的文档化，其中第一次是在将近20年前，当时作者还在做系统移植、驱动开发等Linux内核相关的工作（当时经常上China
Linux
Forum的人应该知道我），彼时的文章和书籍很少，主要就是ULK、LDD、深入理解Linux内核几本。为了能全面理解，作者通读了当时的Linux某个版本（2.4.xx），并且形成了系统的笔记。这些笔记现在都找不到了，除了其中部分放到网上的（因为论坛上有人问才po出来解答，其中还有一篇被人收录在书里，好像是叫Linux什么宝典）。现在想来有点后悔，如果放出来能帮到其它人岂不是比消失不见好？

**为什么没有做成网页版本？**

精力有限。

**为什么感觉没有完成？**

迄今为止下面各章都是作者业余时间独力完成，限于作者个人精力，因此：

-   部分章节尚处于未完成状态

-   有些章节并未细化、深入、展开

-   由于作者并非NN方面的专业人士，很可能多有错误

-   NN发展日新月异，有些知识和经验过时得很快

    自然也不可能做到标题中的"一切"这么厉害。

    **如果你想补充、扩展、修正文中的内容，欢迎通过以下方式联系作者：**

    邮箱地址：[[jimzeus\@gmail.com]{.underline}](mailto:jimzeus@gmail.com)

    知乎帐号：[[https://www.zhihu.com/people/jimzeus]{.underline}](https://www.zhihu.com/people/jimzeus)

    简介

    介绍各章大致的内容，这里没有注明第几章，因为未来章节可能会有调整。

-   "**概念&定义**"介绍了与NN相关的一些基本概念，作为基础。

-   "**Python相关**"介绍了一些Python相关的易混淆的概念，这章和NN关系不大，只是作者在工作过程中遇到的相关问题，需要搞清楚这些概念。

-   "**框架&接口**"介绍了和机器学习相关的库和框架，基本都是基于Python的。

-   "**NN框架**"接续上一章，具体介绍了各种神经网络框架，包括各个公司（Google等）、各个类型

-   "**机器学习**"具体介绍了各种非深度学习的传统机器学习算法、库。

-   "**图像视频处理-OpenCV**"介绍了传统的图像/视频处理的概念和算法，及OpenCV中的实现，这章本身和NN关系也不大，但是CV是NN的应用中的一个大头，了解这些有助于理解和解决CV相关的问题。

-   "**网络构成**"介绍了NN的各种结构、微结构、激活函数、损失函数、优化方法、标准化方法等等组成部件。

-   "**研究方向：XX**"介绍了NN的各个主要研究方向，目前NN最大的研究方向还是CV（计算机视觉）相关，其次就是NLP（自然语言处理）。对于每个子方向，我希望能给出"数据集"、"衡量标准"、"传统方法"及各个神经网络的实现这些内容，而实际上在很多方向上差得还很远。

-   "**实践应用**"NN的某些具体的应用，是上述某些研究方向的工程化/实用化。

-   "**元学习**"是和具体的研究方向无关的研究，比如迁移学习、集成学习、模型压缩、AutoML等。

-   "**硬件支持**"，和硬件相关的杂项放在这里，虽然作者是底层（操作系统/驱动）出身，但对这方面并未有深入研究

-   "**NVidia
    GPU**"本来是硬件支持中的单独一部分，后被单独拿出来作为一章，介绍了英伟达GPU的构成和型号

-   "**边缘计算**"介绍了移动设备上的部分硬件及各个框架，以及（并不那么）相关的ARM指令集/芯片/架构。

-   "**其他**"：其他与NN相关的东西，大部分未完成或者有待调整。

**目录**

一、 概念&定义 20

[（一） 希腊字母列表 20](\l)

[（二） 微积分 21](\l)

[1． 导数和积分 21](\l)

[2． 链式法则 22](\l)

[3． 偏导数 22](\l)

[4． 梯度（Gradient） 23](\l)

[（三） 线性代数/几何学 23](\l)

[1． 张量/标量/向量/矩阵 23](\l)

[2． 特征向量/特征值 24](\l)

[3． 逐点操作 24](\l)

[4． 欧式距离 25](\l)

[5． 余弦距离（余弦相似度） 25](\l)

[6． 曼哈顿距离 26](\l)

[7． 闵可夫斯基距离 26](\l)

[8． 范数 26](\l)

[9． 齐次坐标 27](\l)

[10． 仿射变换（Affine Transformation） 29](\l)

[（四） 概率论 29](\l)

[1． 概率 30](\l)

[2． 条件概率 30](\l)

[3． 贝叶斯定理 30](\l)

[4． 期望值（EV） 31](\l)

[5． 概率密度函数 (PDF) 31](\l)

[6． 累积分布函数 (CDF) 31](\l)

[7． 概率质量函数 32](\l)

[8． 连续分级概率评分（CRPS） 32](\l)

[9． 分位数 (Quantile) 32](\l)

[10． 独立 33](\l)

[11． 独立同分布 33](\l)

[12． 联合分布 33](\l)

[13． 伯努利分布（0-1分布） 34](\l)

[14． 二项分布 34](\l)

[15． 泊松分布 34](\l)

[16． 指数分布 35](\l)

[17． 伽玛分布 36](\l)

[18． 正态分布（高斯分布） 37](\l)

[19． 学生t分布 37](\l)

[20． 多元正态分布 39](\l)

[（五） 信息论 39](\l)

[1． 信息熵（Entropy） 39](\l)

[2． 联合熵（Joint entropy） 40](\l)

[3． 条件熵（Conditional entropy） 40](\l)

[4． 相对熵（Relative entropy） 41](\l)

[5． 交叉熵（Cross entropy） 41](\l)

[（六） 统计学 42](\l)

[1． 方差（Variance） 42](\l)

[2． 标准差（Standard Deviation） 42](\l)

[3． 协方差（Covariance） 42](\l)

[4． 自相关（Auto correlation） 43](\l)

[5． 互相关（Correlation） 44](\l)

[6． 蒙特卡洛法 / 拉斯维加斯法 44](\l)

[7． 马尔可夫链（Markov Chain） 44](\l)

[8． 回归分析 44](\l)

[9． 线性回归（Linear Regression） 45](\l)

[10． 逻辑回归（Logistic Regression） 45](\l)

[11． softmax回归 45](\l)

[12． 最大似然估计（MLE） 45](\l)

[（七） 数字信号处理 46](\l)

[1． 傅立叶变换 46](\l)

[2． 离散傅立叶变换 46](\l)

[3． 快速傅立叶变换 46](\l)

[4． 基波和谐波 46](\l)

[5． 滤波器 47](\l)

[6． FIR滤波器 47](\l)

[7． IIR滤波器 47](\l)

[8． 小波变换 47](\l)

[9． ACF（自相关函数） 47](\l)

[10． XCF（互相关函数） 48](\l)

[（八） 机器学习 48](\l)

[1． 监督学习 49](\l)

[2． 统计分类 49](\l)

[3． 回归分析 50](\l)

[4． 无监督学习 50](\l)

[5． 聚类分析 50](\l)

[6． 强化学习 51](\l)

[7． SVM 51](\l)

[8． k-means 52](\l)

[9． 感知器 52](\l)

[10． 朴素贝叶斯方法 53](\l)

[11． 决策树 53](\l)

[12． 随机森林 54](\l)

[13． kNN算法 54](\l)

[14． 隐马尔可夫模型 55](\l)

[15． 神经网络 56](\l)

[（九） NN相关 56](\l)

[1． 数据集 56](\l)

[2． 层 57](\l)

[3． 参数和FLOPs 58](\l)

[4． Ground Truth 58](\l)

[5． 置信度 58](\l)

[6． One-hot标签 58](\l)

[7． 降维/升维 59](\l)

[8． 上采样/下采样 59](\l)

[9． 编码器/解码器 59](\l)

[10． 开集/闭集 59](\l)

[11． 类内/类间 59](\l)

[12． 二分类任务衡量标准 60](\l)

[13． 感受野 61](\l)

[14． 表现力 61](\l)

[15． 嵌入 61](\l)

[16． 二分类/多分类/多标签分类 62](\l)

[17． 骨干网络（Backbone） 62](\l)

[18． 分布（Distribution） 62](\l)

[（十） 超参数 64](\l)

[1． 学习率 64](\l)

[2． Dropout Ratio 64](\l)

[3． Batch size 64](\l)

[（十一） NN训练相关 64](\l)

[1． Epoch / Batch 64](\l)

[2． 过拟合（Overfit） 65](\l)

[3． 学习率衰减 65](\l)

[4． 权值衰减 65](\l)

[5． 梯度消失 65](\l)

[6． 超参数 66](\l)

[7． 标准化（Normalization） 66](\l)

[8． 泛化（Generalization） 66](\l)

[（十二） CNN相关 66](\l)

[1． 全连接层 66](\l)

[2． 卷积层 66](\l)

[3． 池化层 66](\l)

[4． 卷积核 66](\l)

[5． 特征图 67](\l)

[6． 填充（Padding） 67](\l)

[7． 步幅（Stride） 67](\l)

[8． 通道（Channel） 67](\l)

[二、 Python相关 68](\l)

[（一） 包、模块、属性 68](\l)

[（二） 类和对象 70](\l)

[1． 对象 70](\l)

[2． 类=对象 70](\l)

[3． metaclass 71](\l)

[4． 实例化的过程 72](\l)

[（三） 类 73](\l)

[1． Final和Virtual 73](\l)

[2． 抽象基类 75](\l)

[3． 内嵌抽象类（Collections ABC） 76](\l)

[4． 内嵌类型（built-in types） 78](\l)

[5． 类的层次 78](\l)

[（四） 泛型别名 79](\l)

[（五） 方法 80](\l)

[1． 标准库抽象类、内嵌函数、类特殊方法 80](\l)

[2． 内嵌函数（Built-in Functions） 81](\l)

[3． 类特殊方法（Special method） 83](\l)

[（1） 基本方法 83](\l)

[（2） 属性访问 84](\l)

[（3） 其他 84](\l)

[（4） with语句上下文 85](\l)

[（5） 协程相关 85](\l)

[4． 修饰符 85](\l)

[（六） 迭代器 86](\l)

[1． 定义 87](\l)

[（1） iterable 87](\l)

[（2） iterator 87](\l)

[（3） generator 87](\l)

[（4） generator iterator 88](\l)

[2． 内嵌抽象基类 88](\l)

[（1） Iterable 88](\l)

[（2） Iterator 88](\l)

[（3） Reversible 89](\l)

[（4） Generator 89](\l)

[3． 类特殊方法 89](\l)

[三、 框架&接口 89](\l)

[（一） 框架简介 89](\l)

[1． 传统库 90](\l)

[2． 通用框架 91](\l)

[3． 专用框架 91](\l)

[4． 边缘计算 92](\l)

[5． 快捷接口 93](\l)

[（二） 不同格式间的转换 94](\l)

[1． 文件格式 94](\l)

[2． 格式转换 95](\l)

[3． MMdnn转换 98](\l)

[4． ncnn转换 98](\l)

[（三） 传统库 99](\l)

[1． wave 99](\l)

[2． scipy 99](\l)

[3． numpy 100](\l)

[4． pandas 101](\l)

[5． matplotlib 104](\l)

[四、 NN框架 105](\l)

[（一） Google 105](\l)

[1． Tensorflow（基础框架） 105](\l)

[2． Keras（高级接口） 111](\l)

[（二） Facebook 116](\l)

[1． Pytorch（基础框架） 116](\l)

[2． torchvision（图像处理） 119](\l)

[3． Detectron（已废弃） 121](\l)

[4． Maskrcnn-benchmark（已废弃） 121](\l)

[5． Detectron2（计算机视觉） 121](\l)

[6． PySlowFast（视频理解） 123](\l)

[7． Prophet（时间序列分析） 123](\l)

[（三） Amazon 124](\l)

[1． MXNet（基础框架） 124](\l)

[2． Gluon（高级接口） 125](\l)

[3． gluoncv（图像处理） 125](\l)

[4． gluonts（时间序列） 125](\l)

[5． gluonnlp（自然语言处理） 148](\l)

[（四） CUHK 148](\l)

[1． mmdetection（图像处理） 148](\l)

[2． mmsegmentation（图像分割） 149](\l)

[3． mmaction（视频理解） 149](\l)

[（五） 图森未来 151](\l)

[1． SimpleDet（图像处理） 151](\l)

[（六） Hugging Face 151](\l)

[1． Transformers 152](\l)

[（七） 阿里巴巴 152](\l)

[1． MNN 152](\l)

[（八） 腾讯 152](\l)

[1． ncnn 152](\l)

[（九） Redmon 152](\l)

[1． Darknet 152](\l)

[（十） 其它 155](\l)

[1． PyVideoResearch 155](\l)

[2． Theano 156](\l)

[3． sklearn 156](\l)

[4． DNN (OpenCV) 156](\l)

[五、 传统图像视频处理（OpenCV） 157](\l)

[（一） 基本绘图 157](\l)

[（二） 色彩空间（Colorspace） 158](\l)

[（三） 几何变换（Geometry Transformation） 159](\l)

[1． 刚体变换（Rigid Transformation） 160](\l)

[2． 仿射变换（Affine Transformation） 161](\l)

[3． 投影变换（Projective Transformation） 161](\l)

[4． OpenCV Python 168](\l)

[（四） 阈值处理（Threshold） 169](\l)

[1． 大津算法（Otsu's Method） 169](\l)

[（五） 过滤器-模糊 170](\l)

[1． 2D卷积（2D Convolution） 170](\l)

[2． 均值模糊（Averaging Blur） 170](\l)

[3． 中值模糊（Median Blur） 171](\l)

[4． 高斯模糊（Gaussian Blur） 171](\l)

[5． 双边滤波（Bilateral Filter） 172](\l)

[（六） 形态变换（Morphological Transformation） 173](\l)

[（七） 图像导数（Image Gradient） 174](\l)

[1． 索伯算子（Sobel Operator） 174](\l)

[2． Scharr算子（Scharr Operator） 175](\l)

[3． 拉普拉斯算子（Laplace Operator） 176](\l)

[（八） Canny边缘检测算法 177](\l)

[1． Douglas-Peucker算法（TODO） 178](\l)

[（九） 图像金字塔（Image Pyramids） 178](\l)

[（十） 轮廓（Contour） 179](\l)

[1． 矩（moment） 179](\l)

[（十一） 直方图（Histogram） 181](\l)

[（十二） 模板匹配（Matching Template） 181](\l)

[（十三） 霍夫变换（Hough Transform） 182](\l)

[（十四） Harris角检测器（TODO） 183](\l)

[（十五） SIFT算法（TODO） 183](\l)

[（十六） SURF算法（TODO） 183](\l)

[（十七） 图像分割（Image Segmentation） 183](\l)

[（十八） 视频处理（Video Process） 183](\l)

[1． 均值漂移（mean-shift） 183](\l)

[2． 光流 184](\l)

[（十九） 相机标定（Camera Calibration） 184](\l)

[1． 光心、焦距、焦点 184](\l)

[2． 坐标系 184](\l)

[3． 畸变（distortion） 187](\l)

[4． 相机标定 189](\l)

[5． Opencv Python 191](\l)

[六、 网络构成 193](\l)

[（一） DNN及CNN微结构 193](\l)

[1． 全连接层（Dense） 193](\l)

[2． 池化层（Pooling） 194](\l)

[3． 全局池化层 194](\l)

[4． 反池化层 194](\l)

[5． 常规卷积层 195](\l)

[6． 本地卷积层 196](\l)

[7． Dropout层 196](\l)

[8． Deconv（ZFNet） 196](\l)

[9． Group Conv（AlexNet） 197](\l)

[10． Depthwise Separable卷积（Xception） 197](\l)

[11． 3\*3卷积核（VGGNet） 199](\l)

[12． 1\*1卷积核 200](\l)

[13． Spatial Separable卷积 200](\l)

[14． 带孔卷积 201](\l)

[15． Bottleneck结构 201](\l)

[16． Residual Block（ResNet） 202](\l)

[17． Inverted Residual Block（MobileNet V2） 203](\l)

[18． Linear bottleneck（MobileNet V2） 204](\l)

[19． Dense Block（DenseNet） 204](\l)

[20． Inception结构（GoogleNet） 205](\l)

[21． ResNeXt结构（ResNeXt） 207](\l)

[22． Fire Module（SqueezeNet） 207](\l)

[23． SE结构（SENet） 208](\l)

[24． NASNet单元（TODO） 209](\l)

[25． AmoebaNet单元（TODO） 209](\l)

[26． SPP层（SPP-net） 209](\l)

[27． RoI Pooling层（Fast R-CNN） 210](\l)

[（二） CNN结构 210](\l)

[（三） RNN结构 211](\l)

[1． 通用结构 211](\l)

[2． 变种结构 213](\l)

[3． 双向RNN 217](\l)

[（四） RNN微结构 217](\l)

[1． 基本RNN单元 218](\l)

[2． LSTM单元 219](\l)

[3． GRU单元 221](\l)

[（五） 注意力机制（Attention） 221](\l)

[1． CNN中的Attention 224](\l)

[2． RNN中的Attention (201409) 224](\l)

[3． Soft & Hard attention (201502) 225](\l)

[4． Global & Local Attention (201508) 226](\l)

[5． Transformer & Self-Attention (201706) 227](\l)

[6． Non-Local Network (201711） 227](\l)

[（六） GAN结构 229](\l)

[（七） 激活函数（Activation Function） 229](\l)

[1． sigmoid函数 232](\l)

[2． tanh函数 233](\l)

[3． ReLU函数系列 233](\l)

[4． softmax函数 235](\l)

[5． softsign函数 235](\l)

[（八） 损失函数（Loss Function） 236](\l)

[1． MSE（均方误差） 236](\l)

[2． RMSE 236](\l)

[3． MAE（平均绝对误差） 236](\l)

[4． MAPE 237](\l)

[5． sMAPE 237](\l)

[6． MASE 237](\l)

[7． MSIS 238](\l)

[8． ND 238](\l)

[9． NRMSE 238](\l)

[10． OWA 238](\l)

[11． Cross-entropy Loss（交叉熵损失） 238](\l)

[12． Softmax Loss 239](\l)

[13． Triplet Loss（FaceNet） 239](\l)

[14． Contrastive Loss (TODO) 240](\l)

[15． Center Loss（2016） 240](\l)

[16． A-Softmax Loss（SphereFace） 242](\l)

[17． Focal Loss (RetinaNet) 243](\l)

[（九） 优化方法（Optimizer） 244](\l)

[1． 梯度下降法（GD） 245](\l)

[2． 动量法（Momentum） 246](\l)

[3． NAG 247](\l)

[4． AdaGrad 248](\l)

[5． RMSprop 248](\l)

[6． Adam 248](\l)

[（十） 标准化方法（Normalization） 249](\l)

[1． LRN (AlexNet) 250](\l)

[2． Min-max Normalization 250](\l)

[3． Z-score Normalization 250](\l)

[4． L1 Normalization 251](\l)

[5． L2 Normalization 251](\l)

[6． Batch Normalization (201502) 251](\l)

[7． Layer Normalization (201607) 251](\l)

[8． Instance Normalization (201607) 252](\l)

[9． Group Normalization (201803) 253](\l)

[10． Switchable Normalization (201806) 253](\l)

[七、 研究方向：图像 255](\l)

[（一） 图像分类（Image Classification） 256](\l)

[1． 衡量标准 257](\l)

[2． 数据集 258](\l)

[3． 代码 260](\l)

[4． LeNet（1998） 260](\l)

[5． AlexNet（2012） 261](\l)

[6． ZFNet（201311） 262](\l)

[7． VGGNet（201409） 263](\l)

[8． GoogLeNet（201409） 264](\l)

[9． ResNet（201509） 266](\l)

[10． SqueezeNet（201602） 266](\l)

[11． DenseNet（201608） 267](\l)

[12． Xception（201610） 268](\l)

[13． ResNeXt (201611) 269](\l)

[14． MobileNet V1（201704） 269](\l)

[15． NasNet (201707) 271](\l)

[16． ShuffleNet V1 (201707) 271](\l)

[17． SENet（201709） 272](\l)

[18． MobileNet V2（201801） 273](\l)

[19． MNasNet (201807)(TODO) 274](\l)

[20． ShuffleNet V2 (201807) 274](\l)

[21． EfficientNet (201905)(TODO) 275](\l)

[22． RegNet (TODO) 276](\l)

[（二） 目标检测（Object Detection） 276](\l)

[1． 衡量标准 279](\l)

[2． 数据集 281](\l)

[3． 代码 284](\l)

[4． R-CNN系列 284](\l)

[5． YOLO系列 289](\l)

[6． SSD（201512） 298](\l)

[7． FPN (201612)(TODO) 300](\l)

[8． RetinaNet（201708） 300](\l)

[9． MaskX R-CNN (201711)(TODO) 301](\l)

[10． CenterNet (201904)(TODO) 301](\l)

[11． EfficientDet (201911)(TODO) 301](\l)

[（三） 图像分割（Image Segmentation） 301](\l)

[1． FCN（201411） 302](\l)

[2． SegNet（201511） 303](\l)

[3． DeepLab系列(TODO) 304](\l)

[4． Mask R-CNN（201703） 305](\l)

[5． PointRend（201912）(TODO) 305](\l)

[（四） 人脸识别和建模（Face Recognition） 305](\l)

[1． 数据集 307](\l)

[2． 代码 310](\l)

[3． DeepFace（201406） 310](\l)

[4． DeepID系列（2014） 311](\l)

[5． FaceNet（201503） 313](\l)

[6． MTCNN（201604） 313](\l)

[7． CenterLoss（2016） 314](\l)

[8． SphereFace（201704） 314](\l)

[9． FacePoseNet（201708） 314](\l)

[10． CosFace（201801） 315](\l)

[11． ArcFace/InsightFace（201801） 315](\l)

[12． SeqFace（201803） 315](\l)

[13． MobileFaceNets（201804） 315](\l)

[（五） 人体姿态估计（Pose Estimation） 316](\l)

[1． 数据集 317](\l)

[2． CPM (201602) 317](\l)

[3． HourGlass (201603) 318](\l)

[4． OpenPose (201611) 318](\l)

[5． CPN (201711) 318](\l)

[6． MSPN (201901) 319](\l)

[7． HRNet (201902) 319](\l)

[八、 研究方向：视频 320](\l)

[（一） 行为识别（Video Action Classification） 320](\l)

[1． 数据集 322](\l)

[2． 衡量标准 324](\l)

[3． 传统方法 324](\l)

[4． Two-Stream 324](\l)

[5． 3D Conv 329](\l)

[6． Skeleton-based 335](\l)

[7． LSTM-based 336](\l)

[（二） 时序行为识别（Temporal Action Recognition） 336](\l)

[1． 数据集 337](\l)

[2． 衡量标准 337](\l)

[3． SSN 337](\l)

[（三） 时空行为识别 338](\l)

[1． 数据集 338](\l)

[（四） 视频字幕（Video Captioning） 338](\l)

[1． 数据集 338](\l)

[（五） 视频问答（Video QA） 338](\l)

[1． Zhou (201804) 339](\l)

[2． Video-BERT (201904) 339](\l)

[（六） 视频目标跟踪（Video Object Tracking） 339](\l)

[1． 衡量标准 340](\l)

[2． 数据集 341](\l)

[（七） 多目标跟踪（Multiple Object Tracking） 341](\l)

[1． 衡量标准 342](\l)

[2． 数据集 343](\l)

[3． 传统算法 344](\l)

[4． SORT (201602) 345](\l)

[5． DeepSORT (201703) 346](\l)

[6． MOTDT (201809) 346](\l)

[7． JDE(201909) 346](\l)

[8． FairMOT (202004) 346](\l)

[（八） 行人识别（Person Recognition） 347](\l)

[1． 数据集 347](\l)

[2． MLCNN（2015ICB） 348](\l)

[3． OIM损失函数（201604） 349](\l)

[4． PCB和RPP（201711） 350](\l)

[5． Height, Color, Gender（201810） 351](\l)

[6． st-ReID（201812） 352](\l)

[7． DG-net（201904） 352](\l)

[九、 研究方向：自然语言处理（NLP） 353](\l)

[（一） 概念 353](\l)

[1． 语言学相关 353](\l)

[2． Tokenize（分词） 354](\l)

[3． TF-IDF 354](\l)

[4． 语言模型（LM） 354](\l)

[5． n-grams（n元语法） 354](\l)

[（二） 任务 355](\l)

[1． 语言学NLP任务分类 356](\l)

[2． 神经网络NLP任务分类 357](\l)

[（三） 衡量标准 359](\l)

[1． Accuracy 359](\l)

[2． F1-Score 359](\l)

[3． BLEU 359](\l)

[（四） 数据集 359](\l)

[1． ChnSentiCorp 359](\l)

[2． SQuAD 359](\l)

[3． GLUE 360](\l)

[4． SuperGLUE 362](\l)

[5． CLUE 362](\l)

[（五） 框架 364](\l)

[1． SpaCy 364](\l)

[2． NLTK 364](\l)

[3． Stanford CoreNLP（2014） 364](\l)

[4． Gensim 365](\l)

[5． OpenNMT 365](\l)

[6． ParlAI 366](\l)

[7． DeepPavlov 366](\l)

[8． SnowNLP 366](\l)

[9． Senta 367](\l)

[10． HuggingFace Transformers 367](\l)

[11． HanLP 373](\l)

[12． AllenNLP 373](\l)

[（六） NN：词的表征 373](\l)

[1． NNLM（2003） 374](\l)

[2． Word2Vec（201301） 375](\l)

[3． GloVe(2014) 380](\l)

[4． fastText(201607) 380](\l)

[5． ELMo(201802) 381](\l)

[（七） NN：基于RNN 382](\l)

[1． Encoder-Decoder（201406） 382](\l)

[2． Seq2Seq（201409） 383](\l)

[3． Attention机制（201409） 384](\l)

[（八） NN：基于Transformer 384](\l)

[1． Transformer(201706) 385](\l)

[2． GPT系列 395](\l)

[3． BERT系列 398](\l)

[4． T5（201910） 410](\l)

[十、 研究方向：时间序列（TS） 410](\l)

[1． 数据集 413](\l)

[2． 衡量标准 413](\l)

[（二） 传统方法及概念 413](\l)

[1． AR模型 413](\l)

[2． VAR模型 414](\l)

[3． MA模型 415](\l)

[4． 白噪声 415](\l)

[5． ARMA模型 415](\l)

[6． ARIMA模型 416](\l)

[7． ARFIMA模型 417](\l)

[8． ARCH模型 417](\l)

[9． DTW 417](\l)

[10． COTE 418](\l)

[11． HIVE-COTE (2016ICDM) 418](\l)

[（三） 神经网络方法 418](\l)

[1． WaveNet (201609) (TODO) 418](\l)

[2． DeepAR (201704) 418](\l)

[3． Deep state(201800) (TODO) 419](\l)

[4． Deep factor (201905) (TODO) 419](\l)

[十一、 实践应用 421](\l)

[（一） 车牌识别 421](\l)

[1． 数据集 421](\l)

[2． 代码：EasyPR 422](\l)

[3． 代码：HyperLPR 422](\l)

[（二） 无人机识别 422](\l)

[1． 数据集 422](\l)

[（三） 视频监控异常检测 422](\l)

[1． Real-World Anomaly Detection in Surveillance Videos (201801) (TODO)
423](\l)

[（四） 红绿灯识别 423](\l)

[1． 数据集 423](\l)

[2． 基于OpenCV的红绿灯识别 424](\l)

[3． 用深度学习识别交通灯 424](\l)

[十二、 元学习 424](\l)

[（一） 迁移学习 424](\l)

[1． Fine-tuning 425](\l)

[（二） 知识蒸馏 425](\l)

[（三） 集成学习 425](\l)

[1． Boosting 426](\l)

[（四） 模型压缩 426](\l)

[1． 剪枝 427](\l)

[2． 量化 428](\l)

[（五） AutoML 429](\l)

[1． AutoKeras(开源) 431](\l)

[2． NNI（Microsoft开源） 432](\l)

[3． Cloud AutoML（Google） 435](\l)

[4． AutoGluon（Amazon） 436](\l)

[5． 算法：NAS 437](\l)

[6． 算法：特征工程 442](\l)

[（六） 多示例学习 442](\l)

[（七） 半监督学习 442](\l)

[十三、 硬件支持 443](\l)

[（一） GPGPU 443](\l)

[（二） OpenCL 443](\l)

[（三） CUDA（NVIDIA） 444](\l)

[（四） cuDNN（NVIDIA） 444](\l)

[（五） TPU（Google） 444](\l)

[（六） Linux下Nvidia/CUDA/cuDNN的安装 444](\l)

[1． NVidia驱动 445](\l)

[2． CUDA 445](\l)

[3． CuDNN 446](\l)

[（七） Keras中GPU/CPU的切换 447](\l)

[十四、 Nvidia GPU 448](\l)

[（一） 图形渲染流水线 448](\l)

[1． 应用程序阶段 449](\l)

[2． 几何阶段 449](\l)

[3． 光栅化阶段 & 像素处理 452](\l)

[（二） GPU的构成 454](\l)

[1． 着色器 454](\l)

[2． 统一着色器/流处理器 455](\l)

[3． CUDA核心 455](\l)

[4． 纹理单元 456](\l)

[5． 光栅单元 456](\l)

[6． 光线追踪核心 456](\l)

[7． 张量核心 456](\l)

[8． GPU大核 456](\l)

[（三） NVidia微架构 (Microarchitecture) 459](\l)

[1． Pascal架构 459](\l)

[2． Volta架构 459](\l)

[3． Turing架构 459](\l)

[4． Ampere架构 460](\l)

[（四） NVidia产品系列及型号 460](\l)

[1． GeForce 460](\l)

[2． Quadro 465](\l)

[3． Tesla 465](\l)

[（五） API 466](\l)

[1． DirectX 466](\l)

[2． OpenGL 467](\l)

[3． OpenGL ES 468](\l)

[4． Mesa 468](\l)

[5． Vulkan 468](\l)

[十五、 边缘计算 469](\l)

[（一） 达芬奇NPU（华为）(TODO) 469](\l)

[（二） 寒武纪（寒武纪） 469](\l)

[（三） TensorRT（NVIDIA） 469](\l)

[（四） ncnn（腾讯） 469](\l)

[（五） TNN（腾讯） 470](\l)

[（六） MNN（阿里） 470](\l)

[（七） mace（小米） 470](\l)

[（八） Paddle-Lite（百度） 470](\l)

[（九） Google 470](\l)

[1． TensorFlow Lite（Google） 471](\l)

[2． NNAPI（Google） 476](\l)

[（十） Facebook 477](\l)

[1． Pytorch Mobile (Facebook) 477](\l)

[2． QNNPACK (Facebook) 477](\l)

[（十一） ARM 478](\l)

[1． 指令集/架构/核心 478](\l)

[2． big.LITTLE (ARM) 483](\l)

[3． ARM NN（ARM） 483](\l)

[4． Neon(ARM) 485](\l)

[（十二） Vulkan 485](\l)

[（十三） NNIE(TODO) 485](\l)

[（十四） AidLearning 485](\l)

[十六、 其它 487](\l)

[（一） NN上的工作 487](\l)

[（二） 神经网络类型 488](\l)

[1． 多层感知机（MLP） 488](\l)

[2． 卷积神经网络（CNN） 488](\l)

[3． 递归神经网络（RNN） 488](\l)

[4． 生成对抗网络（GAN） 489](\l)

[5． 受限玻尔兹曼机（RBM） 489](\l)

[6． 深度置信网络（DBN） 490](\l)

[（三） 神经网络可视化 490](\l)

[1． 模型可视化 491](\l)

[2． 训练可视化 492](\l)

[3． 卷积核/特征图可视化 492](\l)

[（四） 相关学习资料 493](\l)

[1． 视频 493](\l)

[2． 课程 493](\l)

[3． 电子书 493](\l)

[（五） 著名人士 (TODO) 494](\l)

[1． Michael Jordan 494](\l)

[2． Bengio 494](\l)

[3． Hinton 494](\l)

[4． Yann LeCun 494](\l)

[5． 李飞飞 494](\l)

[6． 吴恩达 494](\l)

[7． Ian Goodfellow 494](\l)

[8． 汤晓欧 494](\l)

概念&定义
=========

这一章介绍了**神经网络**涉及到的**数学**及**工程**方面的（包括所用到的数学的各个分支，及神经网络的各个研究方向）的一些基本概念、定义、算法等等。

**混淆注意**！

首先来搞清楚几个基本的概念，这些概念之间的区别让很多人都挺迷惑。

NN（神经网络）、ANN（人工神经网络）、DNN（深度神经网络）、DL（深度学习）、ML（机器学习）、AI（人工智能）、MLP（多层感知机）、CNN（卷积神经网络）、RNN（循环神经网络）这些词之间是什么关系？

-   AI \> ML \> DL

这些定义里，**AI**是最宽泛的定义，而**ML**是**AI**的一种实现方式，其特点是不明确编码，而让计算机自行学习参数。**DL**则是**ML**的子集，**DL**指的是使用**DNN**方式实现的**ML**，相对于普通**ML**的特征需要自己选择（即特征工程），**DL**的特征无须手动选择（所谓端到端）。

-   DL = NN

**DL**和**NN**基本是同一个东西，区别在于强调不同方面，**DL**是跟**ML**等概念在一个范围内的，而**NN**则主要强调是其实现的方式。

-   NN = ANN = DNN

一般来说，**NN**=**ANN**=**DNN**，指的都是人工神经网络。

-   DNN \> MLP/CNN/RNN

**MLP**、**CNN**和**RNN**则算是**DNN**的子类（严格来说应该算其要素特点），主要指代其网络中的**全连接**、**卷积**和**循环**特点。

希腊字母列表
------------

下表用于输入公式时复制：

  ---------------- ---------- --------------
  **字母大小写**   **名称**   **拉丁转写**
  Α α              Alpha      a
  Β β              Beta       v
  Γ γ              Gamma      g
  Δ δ              Delta      d
  Ε ε              Epsilon    e
  Ζ ζ              Zeta       z
  Η η              Eta        i
  Θ θ              Theta      th
  Ι ι              Iota       i
  Κ κ              Kappa      k
  Λ λ              Lambda     l
  Μ μ              Mu         m
  Ν ν              Nu         n
  Ξ ξ              Xi         x
  Ο ο              Omicron    o
  Π π              Pi         p
  Ρ ρ              Rho        r
  Σ σ              Sigma      s
  Τ τ              Tau        t
  Υ υ              Upsilon    i
  Φ φ              Phi        f
  Χ χ              Chi        ch
  Ψ ψ              Psi        ps
  Ω ω              Omega      o
  ---------------- ---------- --------------

数学符号
--------

$\propto$:正比于

$\sum_{}^{}:$求多项的和

$\prod_{}^{}:$求多项的积

微积分
------

### 导数和积分

假设函数f:ℝ→ℝ的输入和输出都是标量。函数f的导数为：

![2019-12-27
14-11-01屏幕截图](/home/jimzeus/outputs/AANN/images/media/image1.png){width="2.6951388888888888in"
height="0.71875in"}

且假定该极限存在。给定y=f(x)y=f(x)，其中x和y分别是函数ff的自变量和因变量。以下有关导数和微分的表达式等价：

![2019-12-27 14-11-01屏幕截图
(复件)](/home/jimzeus/outputs/AANN/images/media/image2.png){width="4.302777777777778in"
height="0.6465277777777778in"}

其中符号D和d/dx也叫微分运算符。常见的微分演算有：

-   DC=0（C为常数）

-   Dx^n^=nx^n-1^（n为常数）

-   De^x^=e^x^

-   Dln(x)=1/x等。

如果函数f和g都可导，设C为常数，那么有以下法则：

![2019-12-27 14-18-43屏幕截图
(第4个复件)](/home/jimzeus/outputs/AANN/images/media/image3.png){width="3.6284722222222223in"
height="1.9652777777777777in"}

### 链式法则

如果y=f(u)和u=g(x)都是可导函数，那么有链式法则：

$\frac{\text{dy}}{\text{dx}}$ = $\frac{\text{dy}}{\text{du}}$
$\bullet \frac{\text{du}}{\text{dx}}$

### 偏导数

设u为一个有n个自变量的函数，u=f(x1,x2,...,xn)，它有关第i个变量xi的偏导数为：

![2019-12-27
14-35-46屏幕截图](/home/jimzeus/outputs/AANN/images/media/image4.png){width="5.449305555555555in"
height="0.5333333333333333in"}

以下有关偏导数的表达式等价：

![2019-12-27 14-35-46屏幕截图
(第3个复件)](/home/jimzeus/outputs/AANN/images/media/image5.png){width="3.1326388888888888in"
height="0.6541666666666667in"}

为了计算∂u/∂x~i~，只需将x~1~,...,x~i−1~,x~i+1~,...,x~n~视为常数并求u有关x~i~的导数。

### 梯度（Gradient）

假设函数f:ℝn→ℝ的输入是一个n维向量x=\[x~1~,x~2~,...,x~n~\]^⊤^，输出是标量。函数f(x)有关x的**梯度（Gradient）**是一个由n个偏导数组成的向量：

![2019-12-27 14-35-46屏幕截图
(复件)](/home/jimzeus/outputs/AANN/images/media/image6.png){width="3.1590277777777778in"
height="0.68125in"}

为表示简洁，我们有时用∇f(x)代替∇xf(x)。假设x是一个向量，常见的梯度演算包括：

![2019-12-27 14-35-46屏幕截图
(第3个复件)](/home/jimzeus/outputs/AANN/images/media/image7.png){width="2.6416666666666666in"
height="1.63125in"}

类似地，假设X是一个矩阵，那么：

![2019-12-27 14-35-46屏幕截图
(另一个复件)](/home/jimzeus/outputs/AANN/images/media/image8.png){width="2.0215277777777776in"
height="0.6625in"}

线性代数/几何学
---------------

### 张量/标量/向量/矩阵

**张量**指的是**N轴（axis）**的数据。轴的个数也被称为**阶（order）**。

-   **标量**（Scalar）：0阶的张量，只有一个数字

-   **向量**（Vector）：1阶的张量，也叫矢量，一个数组

-   **矩阵**（Matrix）：2阶的张量，2个轴通常称为行和列

**混淆注意！**

**维（dimension）**通常用来描述向量，比如**N维向量**，比如"128维特征值"。这里的**维**和张量的**轴**定义不一样，这种N维向量描述的仍然是**向量**（即1阶张量），其中的N描述的是数组的元素个数。

但是由于2阶张量（即矩阵）形似2维平面，3阶张量形似3维空间，而N阶张量又可以用**N维数组**来表示，导致**维**有时会被拿来形容张量的**阶**。为了避免混淆，**尽量不要用维来描述张量的阶**。

我们以numpy.ndarrry类型的变量t的形状（t.shape）来说明：

-   (1,)：标量，0阶张量，1维向量，只有1个数值

-   (3,)：矢量，1阶张量，3维向量，有3个数值

-   (5,)：矢量，1阶张量，5维向量，有5个数值

-   (2,3)：矩阵，2阶张量，有6个数值，t\[0\]为3维向量

-   (3,1)：矩阵，2阶张量，有3个值，t\[0\]为1维向量

-   (3,4,5)：3阶张量，有60个数值，t\[0,0\]为5维向量，t\[0\]为矩阵

### 特征向量/特征值

对于一个n行n列的矩阵A，假设有标量λ和非零的n维向量v使**Av=λv**

那么v是矩阵A的一个**特征向量**，标量λ是v对应的**特征值**。

### 逐点操作

**Pointwise(elementwise) operation of
matrix/vector**，指的是形状一致的矩阵/向量对应的每个元素进行操作（加、乘等）。

比如逐点乘法：

a = \[\[ 1. 2. 3.\]

\[ 10. 20. 30.\]\]

b = \[\[ 2. 2. 2.\]

\[ 3. 3. 3.\]\]

c = a\*b

c == \[\[ 20. 40. 60.\]

\[ 30. 60. 90.\]\]

### 欧式距离

**欧式距离（Euclidean distance）**指的是欧式空间中的两个点之间的距离：

在二维空间（平面）中为：![C:\\Users\\AW\\AppData\\Local\\Temp\\1561690119(1).png](/home/jimzeus/outputs/AANN/images/media/image9.png){width="1.4375in"
height="0.2923611111111111in"}

在三维空间中的距离为：![C:\\Users\\AW\\AppData\\Local\\Temp\\1561690162(1).png](/home/jimzeus/outputs/AANN/images/media/image10.png){width="2.0305555555555554in"
height="0.32569444444444445in"}

在N维空间中的距离为：![C:\\Users\\AW\\AppData\\Local\\Temp\\1561690208(1).png](/home/jimzeus/outputs/AANN/images/media/image11.png){width="2.6694444444444443in"
height="0.33819444444444446in"}

在神经网络中，两个N维特征值之间的差距可以用**欧式距离**来衡量。

**欧式边缘（Euclidean
Margin）**指的是两个不同类别间的在欧式空间中的间隔区，在分类问题中，欧式边缘越大，被认为**特征判别度**越高，在面对**unseen数据**（未见过的数据）时候的鲁棒性越好。

### 余弦距离（余弦相似度）

**余弦距离（Cosine
distance）**，是用向量空间中两个向量夹角的余弦值作为衡量两个个体间差异的大小的度量。

**余弦距离**来自**余弦相似度，**余弦相似度取值范围为\[-1,
1\]，向量夹角为0的情况下取1，夹角为180度时取-1。余弦相似度定义如下，其中a,
b分别为两个向量：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561692783(1).png](/home/jimzeus/outputs/AANN/images/media/image12.png){width="1.3722222222222222in"
height="0.5631944444444444in"}

在二维空间中，展开为：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561692838(1).png](/home/jimzeus/outputs/AANN/images/media/image13.png){width="2.3854166666666665in"
height="0.5722222222222222in"}

余弦距离的定义则为：![C:\\Users\\AW\\AppData\\Local\\Temp\\1561694049(1).png](/home/jimzeus/outputs/AANN/images/media/image14.png){width="1.2756944444444445in"
height="0.1798611111111111in"}，取值范围为\[0, 2\]

### 曼哈顿距离

**Manhattan
distance**，表示两个点在标准坐标系上的绝对轴距之总和。公式为：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561959811(1).png](/home/jimzeus/outputs/AANN/images/media/image15.png){width="2.359722222222222in"
height="0.31666666666666665in"}

下图中绿色为欧式距离（长度约为8.48），红色、蓝色和绿色表示的**曼哈顿距离**长度一样，均为12。

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561959873(1).png](/home/jimzeus/outputs/AANN/images/media/image16.png){width="2.8090277777777777in"
height="2.890277777777778in"}

### 闵可夫斯基距离

**闵可夫斯基距离，Minkowski distance**，两个向量（点）的p阶距离。

公式为：![C:\\Users\\AW\\AppData\\Local\\Temp\\1561960049(1).png](/home/jimzeus/outputs/AANN/images/media/image17.png){width="2.484722222222222in"
height="0.3590277777777778in"}

当p=1时就是**曼哈顿距离**，p=2时是**欧式距离**。

### 范数

**范数（norm）**，对于**向量**来说，其公式为：

![2019-12-26
16-27-09屏幕截图](/home/jimzeus/outputs/AANN/images/media/image18.png){width="2.21875in"
height="0.8506944444444444in"}

-   当p为0时，被称为**L0范数**，表示向量x中非0元素的个数。

-   当p为1时，被称为**L1范数**，即为各个元素绝对值之和。对应到原点的**曼哈顿距离**。

-   当p为2时，被称为**L2范数**或者**欧式范数**，若x为N维向量，其L2范数为x到其所在N维空间原点的距离。对应到原点的**欧式距离**，通常‖x‖指代L2范数

对于**矩阵**来说，设X是一个m行n列矩阵。矩阵X的Frobenius范数为该矩阵元素平方和的平方根：

![2019-12-27
12-02-59屏幕截图](/home/jimzeus/outputs/AANN/images/media/image19.png){width="3.0631944444444446in"
height="1.3381944444444445in"}

其中x~ij~为矩阵X在第i行第j列的元素。

### 齐次坐标

**齐次坐标（homogeneous coordinates）**，或**投影坐标（projective
coordinates）**是指一个用于**投影几何**里的坐标系统，如同用于**欧氏几何**里的**笛卡儿坐标**一般。齐次坐标可让包括无穷远点的点坐标以有限坐标表示。使用齐次坐标的公式通常会比用笛卡儿坐标表示更为简单，且更为对称。齐次坐标有着广泛的应用，包括电脑图形及3D电脑视觉。使用齐次坐标可让电脑进行**仿射变换**，并通常，其投影变换能简单地使用矩阵来表示。

如一个点的齐次坐标乘上一个非零标量，则所得之坐标会表示同一个点。因为齐次坐标也用来表示无穷远点，为这一扩展而需用来标示坐标之数值就比投影空间之维度多一。例如，在齐次坐标里，需要两个值来表示在投影线上的一点，需要三个值来表示投影平面上的一点。

比如在欧式空间中表示一个2D的**点**需要2个值，以下为矩阵形式：

p=$\begin{bmatrix}
x \\
y \\
\end{bmatrix}$

而对于点的表示，齐次坐标系加上了一个维度，值为1，并且乘以一个系数k~p~，如下：

p = k~p~$\begin{bmatrix}
x \\
y \\
1 \\
\end{bmatrix}$ = $\begin{bmatrix}
k_{p}x \\
k_{p}y \\
k_{p} \\
\end{bmatrix}$

对于直线的表示，欧式空间中的笛卡尔坐标表示为：

y = ax + b

或者

ax + by + c = 0

而用齐次坐标系来表示，则为

l = $\begin{bmatrix}
a \\
b \\
c \\
\end{bmatrix}$= k~l~$\begin{bmatrix}
m \\
 - 1 \\
g \\
\end{bmatrix}$ = $\begin{bmatrix}
k_{l}m \\
 - k_{l} \\
k_{l}g \\
\end{bmatrix}$

实投影平面可以看作是一个具有额外点的欧氏平面，这些点称之为无穷远点，并被认为是位于一条新的线上（该线称之为无穷远线）。每一个无穷远点对应至一个方向（由一条线之斜率给出），可非正式地定义为一个点自原点朝该方向移动之极限。在欧氏平面里的平行线可看成会在对应其共同方向之无穷远点上相交。给定欧氏平面上的一点 (x, y)，对任意非零实数
Z，三元组 (xZ, yZ, Z) 即称之为该点的齐次坐标。依据定义，将齐次坐标内的数值乘上同一个非零实数，可得到同一点的另一组齐次坐标。例如，笛卡儿坐标上的点
(1,2) 在齐次坐标中即可标示成 (1,2,1) 或
(2,4,2)。原来的笛卡儿坐标可透过将前两个数值除以第三个数值取回。因此，与笛卡儿坐标不同，一个点可以有无限多个齐次坐标表示法。

一条通过原点 (0, 0) 的线之方程可写作 nx + my = 0，其中 n 及 m 不能同时为
0。以参数表示，则能写成 x = mt, y = − nt。令
Z=1/t，则线上的点之笛卡儿坐标可写作 (m/Z,
− n/Z)。在齐次坐标下，则写成 (m, − n, Z)。当 t
趋向无限大，亦即点远离原点时，Z 会趋近于 0，而该点的齐次坐标则会变成 (m,
−n, 0)。因此，可定义 (m, −n, 0) 为对应 nx + my =
0 这条线之方向的无穷远点之齐次坐标。因为欧氏平面上的每条线都会与透过原点的某一条线平行，且因为平行线会有相同的无穷远点，欧氏平面每条线上的无穷远点都有其齐次坐标。

概括来说：

-   投影平面上的任何点都可以表示成一三元组 (X, Y, Z)，称之为该点的齐次坐标或投影坐标，其中
    X、Y 及 Z 不全为 0。

-   以齐次坐标表表示的点，若该坐标内的数值全乘上一相同非零实数，仍会表示该点。

-   相反地，两个齐次坐标表示同一点，当且仅当其中一个齐次坐标可由另一个齐次坐标乘上一相同非零常数得取得。

-   当 Z 不为 0，则该点表示欧氏平面上的该 (X/Z, Y/Z)。

-   当 Z 为 0，则该点表示一无穷远点。

注意，三元组 (0, 0, 0) 不表示任何点。原点表示为 (0, 0, 1)

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image20.png){width="2.5659722222222223in"
height="2.5659722222222223in"}

齐次坐标系中的点（红色线）在投影平面（绿色，x~3~=1）上的投影（交点）

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image21.png){width="2.53125in"
height="2.53125in"}

齐次坐标系中的直线（红线(t~1~,t~2~,t~3~)及其对应的蓝色平面t~1~x~1~+t~2~x~2~+t~3~x~3~=0）在投影平面（绿色，x~3~=1）上的投影（交叉线）

**参考**

[[维基百科]{.underline}](https://zh.wikipedia.org/wiki/%E9%BD%90%E6%AC%A1%E5%9D%90%E6%A0%87)

[[https://mc.ai/part-i-projective-geometry-in-2d/]{.underline}](https://mc.ai/part-i-projective-geometry-in-2d/)

### 仿射变换（Affine Transformation）

**仿射变换**，又称**仿射映射**，是指在几何中，对一个向量空间进行一次线性变换并接上一个平移，变换为另一个向量空间。

一个对向量![](/home/jimzeus/outputs/AANN/images/media/image22.png){width="0.10902777777777778in"
height="0.1701388888888889in"}平移![](/home/jimzeus/outputs/AANN/images/media/image23.png){width="9.791666666666667e-2in"
height="0.18472222222222223in"}，与旋转放大缩小*A*的仿射映射为

![](/home/jimzeus/outputs/AANN/images/media/image24.png){width="0.7888888888888889in"
height="0.19930555555555557in"}

上式在齐次坐标上，等价于下面的式子

![](/home/jimzeus/outputs/AANN/images/media/image25.png){width="1.9305555555555556in"
height="0.5in"}

概率论
------

### 概率

**概率（Probability）**是概率论的基本概念，是一个在0到1之间的实数，是对随机发生事件之可能性的度量。。

事件A发生的概率通常用**P(A)**表示。

### 条件概率

假设事件A和事件B的**概率**分别为P(A)和P(B)，两个事件同时发生的概率记作P(A∩B)或P(A,B)。给定事件B，事件A的**条件概率（后验概率）**：

![2019-12-27
15-08-39屏幕截图](/home/jimzeus/outputs/AANN/images/media/image26.png){width="2.0215277777777776in"
height="0.7006944444444444in"}

也就是说，

**P(A∩B) = P(B)·P(A\|B) = P(A)·P(B\|A)**

当满足

**P(A∩B) = P(A)·P(B))**

时，事件A和事件B相互独立。

### 贝叶斯定理

**贝叶斯定理**（**Bayes\'
theorem**）是概率论中的一个非常重要的定理，描述在已知一些条件下，某事件的发生概率。

![2020-01-07
14-17-16屏幕截图](/home/jimzeus/outputs/AANN/images/media/image27.png){width="2.6368055555555556in"
height="0.9in"}

其中A以及B为随机事件，且P(B)不为零。P(A\|B)是指在事件B![IMG\_260](/home/jimzeus/outputs/AANN/images/media/image28.png){width="1.0416666666666666e-2in"
height="2.0833333333333332e-2in"}发生的情况下事件A发生的概率。

证明很简单，参考上条一目了然，下式都除以P(B)即可：

**P(A∩B) = P(B)·P(A\|B) = P(A)·P(B\|A)**

应用在检测中，实际阳性概率为P(A)，检测阳性的概率为P(B)，则有：

P(A)= $\frac{FN + TP}{所有样本}$， P(B)= $\frac{FP + TP}{所有样本}$ ，
P(B\|A) = $\frac{\text{TP}}{TP + FN}$ ，P(B\|A) =
$\frac{\text{TP}}{TP + FP}$

一眼便知贝叶斯定理成立。

在贝叶斯定理中，每个名词都有约定俗成的名称：

-   P(A\|B)是已知B发生后，A的**条件概率**。由于得自B的取值也叫A的**后验概率**。

-   P(A)是A的**先验概率**（或**边缘概率**）。之所以称为\"先验\"是因为它不考虑任何B方面的因素。

-   P(B\|A)是已知A发生后，B的**条件概率**。由于得自A的取值也叫B的**后验概率**。

-   P(B)是B的**先验概率**。

按这些术语，贝叶斯定理可表述为：

**后验概率 = (似然性\*先验概率)/标准化常量**

也就是说，后验概率与先验概率和相似度的乘积成正比。

另外，比例P(B\|A)/P(B)也有时被称作**标准似然度**（**standardised
likelihood**），贝叶斯定理可表述为：

**后验概率 = 标准似然度\*先验概率**

**参考**

[[https://zh.wikipedia.org/wiki/%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%AE%9A%E7%90%86]{.underline}](https://zh.wikipedia.org/wiki/%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%AE%9A%E7%90%86)

### 期望值（EV）

离散的随机变量X的**期望**（**Expected
Value，EV，数学期望、期望值、平均值**）为

![2019-12-27 15-08-39屏幕截图
(复件)](/home/jimzeus/outputs/AANN/images/media/image29.png){width="2.00625in"
height="0.5777777777777777in"}

### 概率密度函数 (PDF)

**PDF**，**Probability Density
Funtion**，用于描述一个**连续随机变量**的输出值，在某个确定的取值点附近的可能性的函数。

概率密度函数是对连续随机变量定义的，本身不是概率，只有对连续随机变量的概率密度函数在某区间内进行**积分**后才是概率。

为什么本身不是概率？因为PDF描述的是连续随机变量，因此在任意点的概率都为0（这里又引出一个反直觉的概念：不可能事件的概率为0，但概率为0并不一定不可能），举个例子：在\[0,1\]区间上任意抽取一个实数，抽到0.5的概率为0。

### 累积分布函数 (CDF)

累积分布函数（CDF，Cumulative Distribution
Function）是概率密度函数的积分：

![](/home/jimzeus/outputs/AANN/images/media/image30.png){width="1.625in"
height="0.2513888888888889in"}

累积分布函数**有界**（介于0和1之间），**单调增加**，且**右连续**。

**参考**

[[维基百科]{.underline}](https://zh.wikipedia.org/wiki/%E7%B4%AF%E7%A7%AF%E5%88%86%E5%B8%83%E5%87%BD%E6%95%B0)

### 概率质量函数

**概率质量函数（Probability Mass
Function）**，类似**概率密度函数**，用于描述一个**离散随机变量**在各特定取值上的概率。

### 连续分级概率评分（CRPS）

**Continuous Rank Probability
Score**，可以量化一个连续概率分布（理论值）和观测样本（真实值）之间的差异。可以作为概率模型的**损失函数**和**评价函数**。

CRPS在数学形式上是**累积分布函数CDF**与**阶跃函数**（Heaviside step
function）之差的平方在实数域的积分，因此可视为**平均绝对误差**（**Mean
Absolute Error, MAE**）在连续概率分布上的推广。

**参考**

[[https://www.lokad.com/continuous-ranked-probability-score]{.underline}](https://www.lokad.com/continuous-ranked-probability-score)

### 分位数 (Quantile)

分位数指的是在概率分布中（概率密度函数）的分割点，这些分割点使得相邻分割点形成的区间的概率分布一致，下图中正态分布的三个**Quantile**（Q1,Q2,Q3）使得整个正态分布被分为4个概率相同的部分（均为25%）

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image31.png){width="4.684027777777778in"
height="2.7020833333333334in"}

**参考**

[[https://en.wikipedia.org/wiki/Quantile]{.underline}](https://en.wikipedia.org/wiki/Quantile)

### 独立

在概率论里，说两个事件是独立的，直觉上是指一次实验中一事件的发生不会影响到另一事件发生的概率。例如，在一般情况下可以认为连续两次掷骰子得到的点数结果是相互独立的。类似地，两个随机变量是独立的，若其在一事件给定观测量的条件概率分布和另一事件没有被观测的概率分布是一样的。

标准的定义为：

两个事件A和B是独立的，当且仅当** P(A∩B) = P(A)·P(B)**。

**参考**

[[维基百科]{.underline}](https://zh.wikipedia.org/wiki/%E7%8B%AC%E7%AB%8B_(%E6%A6%82%E7%8E%87%E8%AE%BA))

### 独立同分布

在概率论与统计学中，**独立同分布**（**Independent and identically
distributed**，缩写为**IID**）是指一组随机变量中每个变量的概率分布都相同，且这些随机变量互相独立。

一组随机变量独立同分布并不意味着它们的样本空间中每个事件发生概率都相同。例如，投掷非均匀骰子得到的结果序列是独立同分布的，但掷出每个面朝上的概率并不相同。

### 联合分布

在概率论中, 对两个随机变量X和Y，其**联合分布**是同时对于X和Y的概率分布.

对离散随机变量而言，联合分布概率质量函数为Pr(X = x & Y = y)，即：

![](/home/jimzeus/outputs/AANN/images/media/image32.png){width="5.014583333333333in"
height="0.22916666666666666in"}{\\displaystyle P(X=x\\;\\mathrm {and}
\\;Y=y)\\;=\\;P(Y=y\|X=x)P(X=x)=P(X=x\|Y=y)P(Y=y).\\;}

因为是概率分布函数，所以必须有

![](/home/jimzeus/outputs/AANN/images/media/image33.png){width="2.160416666666667in"
height="0.3840277777777778in"}

类似地，对连续随机变量而言，联合分布概率密度函数为fX,Y(x, y)，其中fY\|X(y\|x)和fX\|Y(x\|y)分别代表X = x时Y的条件分布以及Y = y时X的条件分布；fX(x)和fY(y)分别代表X和Y的边缘分布。

同样地，因为是概率分布函数，所以必须有

{\\displaystyle \\int \_{x}\\int \_{y}f\_{X,Y}(x,y)\\;dy\\;dx=1.}

![](/home/jimzeus/outputs/AANN/images/media/image34.png){width="1.9326388888888888in"
height="0.46458333333333335in"}

### 伯努利分布（0-1分布）

**伯努利分布**（**Bernoulli
distribution**，又名**两点分布**或者**0-1分布**），是一个[离散型概率分布](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83#%E7%A6%BB%E6%95%A3%E5%88%86%E5%B8%83)。若伯努利试验成功，则伯努利随机变量取值为1。若伯努利试验失败，则伯努利随机变量取值为0。记其成功概率为![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image35.png){width="0.5541666666666667in"
height="0.15902777777777777in"}，失败概率为![IMG\_257](/home/jimzeus/outputs/AANN/images/media/image36.png){width="0.65625in"
height="0.14583333333333334in"}。

其**概率质量函数**为：

![](/home/jimzeus/outputs/AANN/images/media/image37.png){width="2.65625in"
height="0.5083333333333333in"}

### 二项分布

n次独立重复的**伯努利实验**中，设每次试验中事件A发生的概率为p。用X表示n重伯努利试验中事件A发生的次数，则X的可能取值为0，1，...，n,且对每一个k（0≤k≤n）,事件{X=k}即为"n次试验中事件A恰好发生k次"，随机变量X的离散概率分布即为**二项分布**（**Binomial
Distribution**）。

二项分布的**概率质量函数**为：

![](/home/jimzeus/outputs/AANN/images/media/image38.png){width="2.0104166666666665in"
height="0.3125in"}

二项分布的期望**E(X)=np**，方差**D(X)= np(1-p)**

![](/home/jimzeus/outputs/AANN/images/media/image39.png){width="2.6145833333333335in"
height="1.9305555555555556in"}
![](/home/jimzeus/outputs/AANN/images/media/image40.png){width="2.6881944444444446in"
height="2.1993055555555556in"}

### 泊松分布

**Poisson Distribution**是一种统计与概率学里常见到的离散概率分布。

泊松分布适合于描述单位时间内随机事件发生的次数的概率分布。如某一服务设施在一定时间内受到的服务请求的次数，汽车站台的候客人数、机器出现的故障数、自然灾害发生的次数、DNA序列的变异数、放射性原子核的衰变数、激光的光子数分布等等。

泊松分布的**概率质量函数**为：

![](/home/jimzeus/outputs/AANN/images/media/image41.png){width="1.1423611111111112in"
height="0.3159722222222222in"}

泊松分布的参数**λ**是单位时间（或单位面积）内随机事件的平均发生率。

泊松分布具有如下特性：

-   服从泊松分布的随机变量，其**数学期望**与**方差**相等，同为参数*E*(*X*)
    =*V*(*X*) =λ

-   两个独立且服从泊松分布的随机变量，其和仍然服从泊松分布。更精确地说，若 *X*\~Poisson(λ~1~)，且*Y*\~Poisson(λ~2~)
    ，则*X*+*Y*\~Poisson(λ~1~+λ~2~)

![](/home/jimzeus/outputs/AANN/images/media/image42.png){width="2.566666666666667in"
height="2.4340277777777777in"}
![](/home/jimzeus/outputs/AANN/images/media/image43.png){width="2.6041666666666665in"
height="2.6430555555555557in"}

### 指数分布

**指数分布**（**Exponential
distribution**）是一种连续概率分布。指数分布可以用来表示独立随机事件发生的时间间隔，比如旅客进入机场的时间间隔、打进客服中心电话的时间间隔、中文维基百科新条目出现的时间间隔等等。

其**PDF（概率密度函数）**为：

![](/home/jimzeus/outputs/AANN/images/media/image44.png){width="2.3402777777777777in"
height="0.5708333333333333in"}

**CDF（累积分布函数）**为**：**

![](/home/jimzeus/outputs/AANN/images/media/image45.png){width="2.11875in"
height="0.5118055555555555in"}

![](/home/jimzeus/outputs/AANN/images/media/image46.png){width="2.5805555555555557in"
height="2.092361111111111in"}
![](/home/jimzeus/outputs/AANN/images/media/image47.png){width="2.579861111111111in"
height="2.0902777777777777in"}

[[维基百科]{.underline}](https://zh.wikipedia.org/wiki/%E6%8C%87%E6%95%B0%E5%88%86%E5%B8%83)

### 伽玛分布

**Gamma Distribution**，是一个连续概率函数，有两个参数：

-   α：形状参数

-   β：尺度参数。

假设X~1~, X~2~, \...
X~n~ 为连续发生事件的等候时间，且这n次等候时间为独立的，那么这n次等候时间之和Y
(Y=X~1~+X~2~+\...+X~n~)服从**伽玛分布**，即 Y\~Gamma(α , β)，其中α = n,
β = λ。这里的 λ
是连续发生事件的平均发生频率。 **指数分布**是**伽玛分布**α =
1的特殊情况。

其概率密度函数为：

![](/home/jimzeus/outputs/AANN/images/media/image48.png){width="3.1215277777777777in"
height="1.6729166666666666in"}

![](/home/jimzeus/outputs/AANN/images/media/image49.png){width="2.5875in"
height="2.1013888888888888in"}
![](/home/jimzeus/outputs/AANN/images/media/image50.png){width="2.6215277777777777in"
height="2.1173611111111112in"}

### 正态分布（高斯分布）

**正态分布（normal distribution）**又名**高斯分布**（**Gaussian
distribution**），是一个非常常见的连续概率分布。正态分布在统计学上十分重要，经常用在自然和社会科学来代表一个**不明的随机变量**。

若随机变量X服从一个位置参数为μ、尺度参数为σ的正态分布，记为：

> {\\displaystyle X\\sim N(\\mu ,\\sigma
> \^{2}),}![IMG\_259](/home/jimzeus/outputs/AANN/images/media/image51.png){width="0.9493055555555555in"
> height="0.20347222222222222in"}

则其**概率密度函数**为 ：

{\\displaystyle f(x)={\\frac {1}{\\sigma {\\sqrt {2\\pi
}}}}\\;e\^{-{\\frac {\\left(x-\\mu \\right)\^{2}}{2\\sigma
\^{2}}}}\\!}![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image52.png){width="1.5354166666666667in"
height="0.4888888888888889in"}

正态分布的**期望值**（**EV**）或者**平均值**（**mean**）等于**位置参数**μ，决定了分布的位置；其**方差**σ^2^的开平方或（**标准差**）等于尺度参数σ，决定了分布的幅度。

正态分布的概率密度函数曲线呈钟形，因此人们又经常称之为**钟形曲线**（类似于寺庙里的大钟，因此得名）。我们通常所说的**标准正态分布**是位置参数**μ=0**，尺度参数**σ^2^=1**的正态分布。

![](/home/jimzeus/outputs/AANN/images/media/image53.png){width="2.623611111111111in"
height="1.9756944444444444in"}
![](/home/jimzeus/outputs/AANN/images/media/image54.png){width="2.629166666666667in"
height="2.0097222222222224in"}

### 学生t分布

**Student's
t-distribution**，可简称为**t分布**，常被用于根据小样本来估计呈**正态分布**且方差未知的总体的平均值。如果总体方差已知（例如在样本数量足够多时），则应该用**正态分布**来估计总体均值。

t分布并不是仅仅用于小样本（虽然小样本中用得风生水起）中，大样本依旧可以使用。t分布与正态分布相比多了**自由度参数**，在小样本中，能够更好的剔除异常值对于小样本的影响，从而能够准确的抓住数据的集中趋势和离散趋势。

**t分布是如何产生的：**

假设X~1~,\...X~n~为是正态分布（期望值为μ、方差为σ^2^）的独立的随机采样，令：

![](/home/jimzeus/outputs/AANN/images/media/image55.png){width="0.9347222222222222in"
height="0.4097222222222222in"}

为样本均值，而令

![](/home/jimzeus/outputs/AANN/images/media/image56.png){width="1.625in"
height="0.3861111111111111in"}

为**贝塞尔校正（Bessel's
Correction）**后的样本方差。（[[为什么其分母是n-1？]{.underline}](https://www.zhihu.com/question/20099757/answer/658048814)）

那么随机变量

![](/home/jimzeus/outputs/AANN/images/media/image57.png){width="0.4409722222222222in"
height="0.4097222222222222in"}

服从**标准正态分布**。并且随机变量

![](/home/jimzeus/outputs/AANN/images/media/image58.png){width="0.44513888888888886in"
height="0.3923611111111111in"}

服从**自由度（Degree of
freeom）**为n-1的**学生T分布**，其**概率密度函数**为：

![](/home/jimzeus/outputs/AANN/images/media/image59.png){width="2.3993055555555554in"
height="0.6215277777777778in"}

其中ν= n-1，表示自由度，T分布相较于**正态分布**有"厚尾（heavy
tail）"的特点，表现在**PDF函数**曲线中即峰低尾高，自由度越高，越接近正态分布。当为无穷大时，等价于**正态分布**，当ν=1的时候，就退化成了**柯西分布**，下图中蓝色为正态分布，红色为ν=3的T分布，绿色为ν=1和2的T分布。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image60.png){width="2.8715277777777777in"
height="2.8715277777777777in"}

下图为T分布的PDF和CDF

![](/home/jimzeus/outputs/AANN/images/media/image61.png){width="2.529861111111111in"
height="2.24375in"}
![](/home/jimzeus/outputs/AANN/images/media/image62.png){width="2.6041666666666665in"
height="2.75in"}

**参考**

[[维基百科]{.underline}](https://zh.wikipedia.org/wiki/%E5%AD%A6%E7%94%9Ft-%E5%88%86%E5%B8%83)

[[https://zhuanlan.zhihu.com/p/42136925]{.underline}](https://zhuanlan.zhihu.com/p/42136925)

[[https://www.zhihu.com/question/20099757/answer/658048814]{.underline}](https://www.zhihu.com/question/20099757/answer/658048814)

[[https://www.cnblogs.com/think-and-do/p/6509239.html]{.underline}](https://www.cnblogs.com/think-and-do/p/6509239.html)

### 多元正态分布

**Multivariate Normal
Distribution**，也叫**多变量高斯分布**（**Multivariate Gaussian
Distribution**），指的是N维随机变量X=\[X~1~, \...,
X~N~\]^T^满足下列**等价条件**（即任意一个）：

-   任何线性组合Y=a~1~X~1~+\...+a~N~X~N~服从正态分布

-   存在随机变量Z=\[Z~1~,\...Z~M~\]^T^（它的每个元素服从独立标准正态分布），向量u=\[u~1~,\...u~N~\]^T^及N\*M矩阵A，满足X=AZ+u

信息论
------

### 信息熵（Entropy）

即信息论中的**熵（entropy）**，是接收的每条消息中包含的信息的平均量，又被称为**信息熵、香农熵、信源熵、平均自信息量**。这里，"消息"代表来自分布或数据流中的事件、样本或特征。其定义为：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561637291(1).png](/home/jimzeus/outputs/AANN/images/media/image63.png){width="1.5305555555555554in"
height="0.50625in"}

其中Pi为每种情况发生的概率，各种情况的概率之和为1，比如，一件事情发生的三种情况的概率分别为1/2，1/4，1/4，则其信息熵为（以2为底），1/2\*log2
+ 1/4\* log4 + 1/4\*log4=0.5+0.5+0.5=1.5。

从定义中可以看出，越小概率的事情，其信息熵越高，如果某事件只有一种情况（此时其概率为100%，即1），则信息熵为
1\*log1=0。举个例子，如果有人告诉你"太阳今天从东方升起"那么你没有获得任何信息，因为这必然发生，这话没有任何信息量，等于没说。

熵最好理解为**不确定性的量度**而不是确定性的量度，因为越随机的信源的熵越大。来自信源的另一个特征是样本的概率分布。这里的想法是，比较不可能发生的事情，当它发生了，会提供更多的信息。由于一些其他的原因，把信息（熵）定义为概率分布的对数的相反数是有道理的。事件的概率分布和每个事件的信息量构成了一个随机变量，这个随机变量的均值（即期望）就是这个分布产生的信息量的平均值（即熵）。熵的单位通常为比特，但也用Sh、nat、Hart计量，取决于定义用到对数的底。

信息熵公式的来源：

[[https://zhuanlan.zhihu.com/p/26486223]{.underline}](https://zhuanlan.zhihu.com/p/26486223)

### 联合熵（Joint entropy）

联合熵用于衡量若干变量的不确定性，两个变量X 和Y的联合信息熵定义为：

> ![IMG\_258](/home/jimzeus/outputs/AANN/images/media/image64.png){width="3.2944444444444443in"
> height="0.41180555555555554in"}{\\displaystyle \\mathrm {H}
> (X,Y)=-\\sum \_{x}\\sum \_{y}P(x,y)\\log \_{2}\[P(x,y)\]\\!}

P(x,y)为X和Y分别取值x,y的概率。

-   变量的联合熵大于等于变量中任何一个的独立熵

-   变量的联合熵小于等于变量的独立熵之和

### 条件熵（Conditional entropy）

条件熵描述了在已知第二个随机变量 X的值的前提下，随机变量Y的信息熵还有多少。同其它的信息熵一样，条件熵也用Sh、nat、Hart等信息单位表示。

基于X条件的 Y的信息熵，用 **H(Y\|X) **表示。

-   条件熵的链式法则：

条件熵H(Y\|X)等于联合熵H(Y,X)减去独立熵H(X)：

![](/home/jimzeus/outputs/AANN/images/media/image65.png){width="2.1680555555555556in"
height="0.2375in"}

-   条件熵的贝叶斯规则：

![](/home/jimzeus/outputs/AANN/images/media/image66.png){width="2.673611111111111in"
height="0.22916666666666666in"}

H(Y\|X) = H(X,Y) - H(X)

H(X\|Y) = H(X,Y) - H(Y)

上式减下式，可得

H(Y\|X) - H(X\|Y) = H(Y) - H(X)

得：

H(Y\|X) = H(X\|Y) - H(X) + H(Y)

### 相对熵（Relative entropy）

也叫**KL散度**（**Kullback-Leibler
Divergence**），是两个概率分布P和Q差别的非对称性的度量。
KL散度是用来度量使用基于Q的分布来编码服从P的分布的样本所需的额外的平均比特数。典型情况下，P表示数据的真实分布，Q表示数据的理论分布、估计的模型分布、或P的近似分布。

对于离散随机变量，其概率分布P 和 Q的KL散度可按下式定义为

![](/home/jimzeus/outputs/AANN/images/media/image67.png){width="2.7534722222222223in"
height="0.5409722222222222in"}

### 交叉熵（Cross entropy）

在信息论中，基于相同事件测度的两个概率分布p和q的**交叉熵（Cross
entropy）**是指，当基于一个"非自然"（相对于"真实"分布p而言）的概率分布q进行编码时，在事件集合中唯一标识一个事件所需要的平均比特数（bit）。基于概率分布p和q的交叉熵定义为：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561637828(1).png](/home/jimzeus/outputs/AANN/images/media/image68.png){width="3.8847222222222224in"
height="0.3486111111111111in"}

其中H(p)为p的熵，也就是说交叉熵永远的大于等于真实分布的**信息熵**，![C:\\Users\\AW\\AppData\\Local\\Temp\\1561637913(1).png](/home/jimzeus/outputs/AANN/images/media/image69.png){width="0.59375in"
height="0.21388888888888888in"}被称为p到q的**相对熵**。

对于一个离散分布的p和q，**交叉熵**为：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561637999(1).png](/home/jimzeus/outputs/AANN/images/media/image70.png){width="2.660416666666667in"
height="0.5270833333333333in"}

例如，一件事情发生的三种情况的真实分布分别为1/2，1/4，1/4，预测分布则为1/4，1/4，1/2，

则交叉熵为（以2为底）：1/2\*log4 + 1/4\* log4 + 1/4\*log2= 1+0.5+0.25 =
1.75。

### 汉明距离（Hamming Distance）

**汉明距离（Hamming
Distance）**表示两个等长字符串在对应位置上不同字符的数目，我们以d(x,
y)表示字符串x和y之间的汉明距离。

对于二进制串a和b来说，汉明距离等于a XOR
b中１的数目，我们又称其为汉明权重，也叫做population count或popcount。

**参考**

汉明码（Hamming Code）汇总

[[https://zhuanlan.zhihu.com/p/84614845]{.underline}](https://zhuanlan.zhihu.com/p/84614845)

统计学
------

### 方差（Variance）

**方差，Variance**，在概率论和统计学中，一个随机变量的方差描述的是它的离散程度，也就是该变量离其期望值的距离。一个实随机变量的方差也称为它的二阶矩或二阶中心动差，恰巧也是它的二阶累积量。

简单来说，就是将各个误差将之平方（而非取绝对值，使之肯定为正数），相加之后再除以总数，透过这样的方式来算出各个数据分布、零散（相对中心点）的程度。继续延伸的话，方差的正平方根称为该随机变量的**标准差**（此为相对各个数据点间）。

![](/home/jimzeus/outputs/AANN/images/media/image71.png){width="5.061805555555556in"
height="0.6097222222222223in"}

设X为服从分布F的随机变量，
如果E\[X\]是随机变数*X*的期望值（平均数μ=E\[*X*\]），随机变量X或者分布F的方差为：**Var(X)
= E\[(X-μ)^2^\]**

### 标准差（Standard Deviation）

**标准差（标准偏差、均方差，Standard Deviation，SD）**，数学符号σ（sigma），在概率统计中最常使用作为测量一组数值的离散程度之用。标准差定义：为**方差**开算术平方根，反映组内个体间的离散程度；标准差与期望值之比为标准离差率。

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562381701(1).png](/home/jimzeus/outputs/AANN/images/media/image72.png){width="2.3333333333333335in"
height="0.7152777777777778in"}

### 协方差（Covariance）

**协方差（Covariance）**在概率论和统计学中用于衡量两个变量的总体误差。而**方差**是协方差的一种特殊情况，即当两个变量是相同的情况。

期望值分别为**E(X)=μ**与**E(Y)=ν**的两个具有有限二阶矩的实数随机变量X 与Y 之间的协方差定义为：

**cov(X,Y) = E((X-μ)(Y-ν)) = E(X·Y)-μν**

其推导过程为：

![](/home/jimzeus/outputs/AANN/images/media/image73.png){width="3.4034722222222222in"
height="0.6888888888888889in"}

如果两个变量的变化趋势一致，也就是说如果其中一个大于自身的期望值时另外一个也大于自身的期望值，那么两个变量之间的协方差就是正值；如果两个变量的变化趋势相反，即其中一个变量大于自身的期望值时另外一个却小于自身的期望值，那么两个变量之间的协方差就是负值。

如果X与Y是统计独立的，那么二者之间的协方差就是0，因为两个独立的随机变量满足**E\[XY\]=E\[X\]E\[Y\]**。但是反过来并不成立，即如果X与Y的协方差为0，二者并不一定是统计独立的。

### 自相关（Auto correlation）

**自相关**（**Autocorrelation**），也叫**序列相关**，是一个信号于其自身在不同时间点的互相关。非正式地来说，它就是两次观察之间的相似度对它们之间的时间差的函数。它是找出重复模式（如被噪声掩盖的周期信号），或识别隐含在信号谐波频率中消失的基频的数学工具。它常用于信号处理中，用来分析函数或一系列值，如时域信号。

自相关函数在不同的领域，定义不完全等效。在某些领域，自相关函数等同于**自协方差**（信号与自身经过时间平移得到的信号之间的协方差）。比如在统计学中：

![](/home/jimzeus/outputs/AANN/images/media/image74.png){width="2.3965277777777776in"
height="0.46319444444444446in"}

**E**：期望值

**X~i~**：在t(i)时的随机变量值。

**μ~i~**：在t(i)时的预期值。

**X~i+k~**：在t(i+k)时的随机变量值。

**μ~i+k~**：在t(i+k)时的预期值。

**σ^2^**：为方差。

### 互相关（Correlation）

在统计学中，**互相关（Correlation）**有时用来表示**两个随机矢量 X 和 Y 之间**的协方差**cov（X, Y）**，以与矢量 X 的**协方差**概念相区分，矢量 X 的**协方差**是 X 的各标量成分之间的协方差矩阵。

### 蒙特卡洛法 / 拉斯维加斯法

**蒙特卡罗算法**（**Monte Carlo
Method**，**统计模拟方法**）和**拉斯维加斯算法**（**Las Vegas
Algorithm**）并不是一种算法的名称，而是对一类随机算法的特性的概括。具有如下特点：

-   蒙特卡罗算法：采样越多，越近似最优解；

-   拉斯维加斯算法：采样越多，越有机会找到最优解

**参考**

[[https://www.zhihu.com/question/20254139/answer/33572009]{.underline}](https://www.zhihu.com/question/20254139/answer/33572009)

### 马尔可夫链（Markov Chain）

**马尔科夫链（MC，Markov
Chain）**定义本身比较简单，它假设某一时刻状态转移的概率只依赖于它的前一个状态。举个形象的比喻，假如每天的天气是一个状态的话，那个今天是不是晴天只依赖于昨天的天气，而和前天的天气没有任何关系。

当然这么说可能有些武断，但是这样做可以大大简化模型的复杂度，因此马尔科夫链在很多时间序列模型中得到广泛的应用，比如循环神经网络**RNN**，**隐式马尔科夫模型HMM**等，当然**MCMC（马尔科夫链-蒙特卡洛方法）**也需要它。

**参考**

[[https://www.cntofu.com/book/85/math/probability/markov-chain.md]{.underline}](https://www.cntofu.com/book/85/math/probability/markov-chain.md)

### 回归分析

**回归分析（Regression
Analysis）**是一种统计学上分析数据的方法，目的在于了解两个或多个变量间是否相关、相关方向与强度，并建立数学模型以便观察特定变量来预测研究者感兴趣的变量。更具体的来说，回归分析可以帮助人们了解在只有一个自变量变化时因变量的变化量。一般来说，通过回归分析我们可以由给出的自变量估计因变量的条件期望

回归分析是建立因变量Y（或称依变量，反因变量）与自变量X（或称独变量，解释变量）之间关系的模型。简单线性回归使用一个自变量X，复回归使用超过一个自变量（X~1~,
X~2~ \... X~i~）。

**参考**

[[维基百科]{.underline}](https://zh.wikipedia.org/wiki/%E8%BF%B4%E6%AD%B8%E5%88%86%E6%9E%90)

### 线性回归（Linear Regression）

**Linear
Regression，线性回归**输出是一个连续值，因此适用于回归问题。回归问题在实际中很常见，如预测房屋价格、气温、销售额等连续值的问题。

与回归问题不同，分类问题中模型的最终输出是一个离散值。我们所说的图像分类、垃圾邮件识别、疾病检测等输出为离散值的问题都属于分类问题的范畴。**softmax回归**则适用于分类问题。

### 逻辑回归（Logistic Regression）

统计学中，**逻辑回归（Logistic
Regreession，LR）**用于给分类问题的可能性建模，通常用于**二分类**，比如胜负、生死、健康/生病等等。但也可以用于**多分类**，在多分类的逻辑回归中，每个分类的可能性被赋予一个0到1之间的值，所有分类的值和为1。

**混淆注意！**

逻辑回归虽然名字叫回归，但却是一种分类模型

**参考**

[[https://en.wikipedia.org/wiki/Logistic\_regression]{.underline}](https://en.wikipedia.org/wiki/Logistic_regression)

### softmax回归

**Softmax Regression**，又叫**多项式回归（Multinomial
Regression，Multinomial Logistic
Regression）**。用于分类问题，参考[[softmax激活函数]{.underline}](\l)。

### 最大似然估计（MLE）

**最大似然估计（MLE**，**Maximum Likelihood
Estimation）**，用于估计一个概率模型的参数的方法。

给定一个概率分布**D**，已知其概率密度函数（连续分布）或概率质量函数（离散分布）为**f~D~**，以及一个分布参数**θ**，我们可以从这个分布中抽出一个具有n个值的采样X~1~,
X~2~, \..., X~n~，利用**f~D~**计算出其似然函数：

**L(θ\|x~1~,\...,x~n~) = f~θ~(x~1~,\...,x~n~) **

若**D**是离散分布，**f~θ~**即是在参数为**θ**时观测到这一采样的概率。若其是连续分布，**f~θ~**则为**X~1~,X~2~,\...,X~n~**联合分布的概率密度函数在观测值处的取值。一旦我们获得**X~1~,X~2~,\...,X~n~**，我们就能求得一个关于**θ**的估计。最大似然估计会寻找关于**θ**的最可能的值（即，在所有可能的**θ**取值中，寻找一个值使这个采样的"可能性"最大化）。从数学上来说，我们可以在**θ**的所有可能取值中寻找一个值使得似然函数取到最大值。这个使可能性最大的**θ**值即称为**θ**的**最大似然估计**。由定义，最大似然估计是样本的函数。

**参考**

[[维基百科]{.underline}](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1)

数字信号处理
------------

### 傅立叶变换

**Fourier
Transform**，一种线性积分变换，用于信号在**时域**（或**空域**）和**频域**之间的变换。

### 离散傅立叶变换 

**DFT，Discrete Fourier
Transform**，是**傅立叶变换**在时域和频域上都呈离散的形式，将信号的时域采样变换为其**DTFT**（离散时间傅立叶变换）的频域采样。

### 快速傅立叶变换

**Fast Fourier
Transform，FFT**，是快速计算**离散傅立叶变换（DFT）**的方法

### 基波和谐波

在复杂的周期性震荡中，包含**基波**和**谐波**，和该振荡最长周期相等的正弦波分量称为基波。相应于这个周期的频率称为**基波频率**。频率等于基波频率的**整倍数**的正弦波分量称为**谐波**。

### 滤波器

**filter**，过滤特定频率信号的器件/装置/程序/函数，根据其作用大致分为

-   **低通滤波器**：Low-Pass
    Filter，LPF，允许低频（低于某个频率）信号通过

-   **高通滤波器**：High-Pass
    Filter，HPF允许高频（高于某个频率）信号通过

-   **带通滤波器**：Bandwidth-Pass
    Filter，BPF，允许某个频带（bandwidth，某两个频率之间）信号通过，通常是高通滤波器和低通滤波器的级联

-   **带阻滤波器**：减弱某个频带（bandwidth，某两个频率之间）信号通过，通常是高通滤波器和低通滤波器的级联

### FIR滤波器

**Finite Impulse Response
Filter，有限脉冲响应滤波器**，数字滤波器的一种。这类滤波器对于脉冲输入信号的响应最终趋向于0，因此是有限的，从而得名。

FIR滤波器是一线性系统，输入信号x(0), x(1), \... ,
x(n)，经过该系统后的输出信号y(n)可表示为：

![](/home/jimzeus/outputs/AANN/images/media/image75.png){width="3.652083333333333in"
height="0.20833333333333334in"}

其中h~0~, h~1~, \...,
h~N~是滤波器的脉冲相应，通常称为**滤波器的系数**，N是**滤波器的阶数**。

### IIR滤波器

**Infinite Impulse Response
Filter，无限脉冲响应滤波器**，数字滤波器的一种。

### 小波变换

**WT，Wavelet
Transform**，跟傅立叶变换一样分为**连续小波变换**（**CWT**）和**离散小波变换**（**DWT**）。

### ACF（自相关函数）

**自相关函数（Auto-correlation
function）**，用于衡量信号在不同时期之间的相关关系。

### XCF（互相关函数）

**互相关函数（Cross-correlation
function）**，是用来表示两个信号之间相似性的一个度量。

对于离散函数**f~i~**和**g~i~**来说，互相关定义为：

![](/home/jimzeus/outputs/AANN/images/media/image76.png){width="1.775in"
height="0.42291666666666666in"}

互相关实际上类似于两个函数的卷积。

NN相关
------

神经网络相关的概念

### 数据集

**数据集（Dataset）**，机器学习模型的输入数据，通常分为三部分：

-   **训练集：Training Set**，用于训练模型，调整权重参数

-   **验证集：Validation Set**，也叫**开发集（Dev
    Set）**用于模型的选择，超参数的选择

-   **测试集：Test Set**，为了最终测试选择并训练好的模型（的泛化能力）。

**训练集**是模型学习训练的数据集，
不同的模型（神经网络结构、超参数）在这个数据集上调整权重参数，以求（在训练集上）达到更高的准确率。而**验证集**提供了一个统一的衡量标准，以便调整模型（修改超参数、网络结构，或者直接放弃某个模型），某些训练过程不使用验证集。**测试集**则是在模型训练完成之后，用于测试模型的准确性（测试集之前从没喂给过模型，以测试泛化能力）。

打个比方，训练集就像平时作业，验证集像平时小考，测试集则是决定你升学方向的毕业考试。

数据集是神经网络的研究中非常重要的一个部分，甚至有有独有的数据集就可以出论文的说法。

CV数据集大全：

[[http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm]{.underline}](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm)

数据集列表：

[https://en.wikipedia.org/wiki/List\_of\_datasets\_for\_machine-learning\_research]{.underline}

### 层

神经网络中的层通常可以分为输入层、输出层和隐藏层三大类。层是一个略微模糊的概念，举一个最简单的例子，下图是一个2个输入（i1和i2），2个输出（o1和o2），2个隐藏神经元（h1和h2）的神经网络。

![](/home/jimzeus/outputs/AANN/images/media/image77.png){width="2.58125in"
height="1.8798611111111112in"}

那么问题来了，这个网络有几层？

从结构图上看，可以认为节点（即张量）代表层，因此可以说它有**三层**（输入层i、输出层o和隐藏层h）。而从计算的角度来讲，连接线代表层，因此其实只有**两层**全连接层（输入-\>隐藏，隐藏-\>输出），所谓的输入层只是一个没有计算量的向量而已。

此外在某些环境里，只有**卷积**、**全连接**等这些层被视为层，而在某些环境里，**BN**、**激活函数**也被视为BN层、激活层。

### 参数和FLOPs

有两个值可以用于描述神经网络的大小，层的结构直接决定了这两个值：

-   **参数数量**

可学习参数（即权重weight、偏置bias等）的数量，这个值的大小决定了神经网络在空间上的大小（即所使用的内存和硬盘的大小）。

-   **FLOPs**

**FLoat
OPerations**，描述一次计算的**浮点操作**数量（注意不要和**FLOPS/FLoat
Operations Per
Second**混淆，FLOPS是用来描述计算机性能的），这个值的大小决定了计算量，也就是神经网络在时间上的大小。

**FLOPs**通常是"加"和"乘"两种操作，有两种计算方法，一种是将加和乘分别算作一次，还有一种是将一次加和一次乘合起来算作一次FLOP（因为加和乘操作是一一对应的，这也是通常的计算方式）。下面的计算中我们采用第二种。

**参数**和**FLOPs**正相关，但不成严格正比。比如同样FLOPs的情况下，全连接层的参数数量就要远大于卷积层，但可能是由于网络中大部分计算量都集中在卷积层的原因，很多文章中直接用参数数量来衡量计算量。

### Ground Truth

**Ground Truth**（**GT**）表示**真实值**。

**BTW**：至于为什么叫Ground，据说这个词的起源是**遥感领域**，相对于在天上的遥感卫星所测量到的数据来说，地面上的真实数据就是Ground
Truth。

### 置信度

**Confidence**，表示对某个预测的信心，通常取值范围从0到1。

### One-hot标签

**One-hot标签**（中文通常翻译为**独热标签**，但很少用，一般直接用英文）用于描述分类任务的**ground
truth**，一个描述N个分类的N维向量，其中正确分类对应的标签值为1，其余都为0，没有其它取值。

### 降维/升维

**降维**指的是减少特征维度（比如CNN中特征图的通道数量），实现方法有1\*1卷积等。

**升维**指的是增加特征维度，实现方法有1\*1卷积等。

### 上采样/下采样

**上采样（Upsampling）**，目的是放大图像，采用的具体方法有插值、反卷积、反池化等。

下采样又叫**降采样**（**Subsampling**，**Downsampling**），目的是缩小图像，具体采用的方法主要是池化，步长（stride）大于1的卷积也可以实现下采样。

### 编码器/解码器

**编码器（encoder）**和**解码器（decoder）**，在深度学习中，编码器的概念是指一个将输入转化为特征图作为输出的网络（可以是FC、CNN、RNN），特征图中包含了输入的信息。而解码器则是一个完全相反的网络，以特征图作为输入，而试图给出期望的输出。

通常**编码**是个**降采样**的过程，而**解码**是个**上采样**的过程。编码器和解码器成对出现（基础的CNN比如VGG等就可以看作一个编码器，但是在没有对应的解码器网络的时候它不会被叫做编码器），先编码，再解码。**编码器/解码器**在**图像语义分割、机器翻译、GAN**等等中都有应用。

### 开集/闭集

**闭集（close-set）**指的是在分类问题中，所有的输入数据都属于某一类别，不会出现没有见过的unknown数据。而**开集（open-set）**正好相反。比如在人脸识别中，如果会有不在数据库中的人脸作为输入，则为open-set，否则为close-set。

闭集对类别间的**判别度**（discriminative）要求更低，因为数据已知，而开集则因为运行中的数据未知，因此对判别度的要求更高，这样才好更清晰的区别于各种未知数据。

### 类内/类间

**类内（intra-class）**指的是分类问题中属于同一类别的不同样本，**类间（inter-class）**指的是属于不同类别的样本。

### 二分类任务衡量标准

在图像分类和目标检测等**二分类任务**中，检测结果可分为四类，即

-   **TP：True Positive**，**真阳性**，阳性样本，检测为阳性

-   **FP：False Positive**，**假阳性**，阴性样本，检测为阳性

-   **TN：True Negative，TN**，**真阴性**，阴性样本，检测为阴性

-   **FN：False Negative，FN**，**假阴性**，阳性样本，检测为阴性

其中TP和FP是检测结果为阳性，TN和FN是检测结果为阴性；TP和FN实际为阳性，TN和FP实际为阴性。下面介绍由这四个值引申出几个"率"：

-   **Accuracy：准确率**，最直观的衡量的标准，等于(TP+TN)/(TP+TN+FP+FN)，所有样本的检测准确性。

-   **TPR（True Positive
    Ratio）**，也叫**查全率**，**召回率（Recall）**，hit
    rate，或者Sensitivity，等于TP/(TP+FN)，用于衡量所有实际为阳性的样本的检测准确率。

-   TNR（True Negative
    Ratio），又叫Specificity，等于TN/(TN+FP)，用于衡量所有实际为阴性的样本的检测准确率，不太常用。

-   FNR（False Negative
    Ratio），等于FN/(TP+FN)，等于1-TPR，用于衡量所有实际为阳性样本的检测错误率，不太常用。

-   FPR（False Positive
    Ratio），等于FP/(FP+TN)，等于1-TNR，用于衡量所有实际为阴性样本的检测错误率，不太常用。

-   **查准率（准确率，Precision）**，**PPV（Positive Predictive
    Value）**，等于TP/(TP+FP)，表示所有**被检测为**阳性的样本中，检测正确的比例。

-   NPV（Negative Predictive
    Value），等于TN/(TN+FN)，表示所有被检测为阴性的样本中，检测正确的比例，不太常用。

**混淆注意！**

Accuracy和Precision这两个词中文都可翻译为准确率，但在这里是两个概念，因此建议使用**查准率**表示Precision，或者直接用英文。

-   **F1-Score**：也叫**F1-measure**，是β=1时**F-score**的一个特例，F-score同时考虑了查准率和查全率，是这两者的一个加权平均，**F1-Score**的公式为：**2×**$\frac{\mathbf{\text{precision}}\mathbf{\  \times \ }\mathbf{\text{recall}}}{\mathbf{precision\  + \ recall}}$

不同的衡量标准在不同的情况下有意义，要根据具体的工作场景来决定用什么衡量标准。

场景A：

对医学影像的癌症检测，可能在在所有样本中，实际为阳性的只有0.1%，这时候如果算法将所有样本直接判定为阴性，则Accuracy也可以达到99.9%，但这毫无意义。

场景B：

PPV和NPV这两个值，在二元图像分类任务中，仅在测试数据集中阳性/阴性样本的比例和真实环境差不多的时候有意义。举个例子，在鉴定枪械图片的任务中，测试数据集包括100张枪械图片和100张非枪械图片，枪械图片被检测为枪械图片有95张（TP），5张没检测出来（FN）；非枪械图片被检测为枪械图片有10张（FP），被检测为非枪械图片为90张（TN）。

那么，TPR为95%，TNR为90%，PPV=95/(95+10)约为90.5%；NPV=90/(90+5)约为94.7%。仿佛挺美好，但是现实生活中的绝大部分图片都不是枪械图片，让我们假设每10000张非枪械图片里搭配100张枪械图片，假设TPR和TNR不变，则TP和FN数量不变，TN为9000张，FP为1000张，于是有：PPV=95/(95+1000)，约为8.7%。或者说，每100张被识别为枪械的图片中，只有9张不到是真的枪械图片，另外90多张都是错误识别的结果。

### 感受野

感受野指的是在图像相关的任务中，卷积核能"看到"的范围，和卷积核的大小，以及卷积方式有关，比如7\*7卷积核的感受野大于5\*5卷积核，两次3\*3卷积核的感受野大于一次，带孔卷积的感受野大于普通卷积。

### 表现力

表现力指的是网络对输入数据描述的能力，表现力过弱会导致准确性不高，这就像你无法用8个bit来描述足够鲜艳的颜色一样。而表现力过强也可能会带来过拟合之类的问题（在数据集不够大的情况下），这种情况下网络表现力会过多的描述一些事实上和结果无关的特征。

### 嵌入

**embedding**，通常为一个N维的向量，其概念有点类似特征值，指的是某种高维空间的向量在低维空间的表示。

比如词嵌入（对应的原始高维度为所有词的个数，One-hot格式），或者人脸识别中的脸部特征的嵌入（对应的原始高维度为图片长×宽×3，3为RGB）。

可以把**嵌入**视为一种描述，对于词来说，类似一个对该词的释义，对脸图来说，就是对脸的描述，比如眼睛大小用一个值表示，鼻子长短用一个值表示等，这些值组成了一个若干维的向量。只不过这种描述更加精确，且无法为人类所理解。

有一个例子可以阐述嵌入的意思，词嵌入中，比如我们用**E~国王~**表示"国王"的词嵌入，可以发现**E~国王~-E~男人~+E~女人~**得出的结果，和**E~女王~**很接近。

### 二分类/多分类/多标签分类

**二分类（binary-class
classification）**指的是预测结果只有两种可能的类别，比如**是**或者**否**。

**多分类（multi-class
classification）**指的是预测结果有若干种可能的类别，比如一个通常的图像分类问题，预测结果可能是人、车、树等等。

**多标签分类（multi-label
classification）**指的是在多分类的基础上，一个实例预测的结果可能多个标签，而非仅属于一个类别，比如一张图片中既包含人，也包含车。图片多分类不是一个显学，因为类似的功能通常在目标检测任务（Object
Detection）里实现。

### 骨干网络（Backbone）

**骨干网络（Backbone）**，也叫**主干网络**，**基础网络**，通常指的是图像/视频处理任务中，用于特征提取的核心部分，这种网络本身通常用于图像分类任务，而在更复杂的图像/视频处理中（目标检测、图像分割、行为识别等等），作为一个子部分出现。

### 分布（Distribution）

这里的**Distribution**，是指各种深度学习框架中，用于描述**可能性分布**（**Probability
Distribution**）的数据结构，在[[Tensorflow]{.underline}](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution?version=nightly)、[[Pytorch]{.underline}](http://docs.pyro.ai/en/stable/distributions.html)、[[Gluonts]{.underline}](https://gluon-ts.mxnet.io/api/gluonts/gluonts.distribution.html)中都有出现。

Distribution相关的shape（张量的维度）分为三个级别，**从小到大**分别是：

-   **Event Shape**：一个Event
    shape的张量描述了**单个分布中**的一个事件，如果这个分布是个单元分布（Univariate
    Distribution），则event
    shape为标量；如果该分布是一个N元的多元分布（Multivariate
    distribution，比如多元正态分布），则event shape是一个**N维向量**。

-   **Batch Shape**：Batch
    描述了一次采样中的若干分布（样本空间）的事件，batch
    shape维度描述了这些相互独立的分布的数量。

-   **Sample
    Shape**：描述了采样（sample）次数，这个shape中的每个元素都是同样的若干个（单元或者多元）分布的事件。

举个例子，让5个人每人掷2个骰子，两个骰子点数各为一个变量，两个骰子都为3是一个事件。让这5个人掷4次。则：

-   event shape为2维，两个元素都为3，单个event的概率是1/36

-   batch shape为5维

-   sample shape为4维

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image78.png){width="4.58125in"
height="6.157638888888889in"}

**混淆注意！**

**draw**这个词在不同语境下会有歧义，可以把包含单个分布的一次抽签视为一个draw（仅包括event
shape），也可以把所有分布的一次抽签视为一个draw（包括batch shape和event
shape）。

由以上可知，对于单个sample，变量的维度为Batch shape \* event
shape，而其概率密度的维度为Batch shape。

总结起来：

-   Event中的元素（变量）**可能**是相互依赖的（dependent），产生单个概率密度

-   Batch中的各个元素（Event，事件）互相独立（independent），但并不相同（identical），产生batch
    shape个概率密度。

-   Sample中的各元素（Sample）相互独立且相同（即独立同分布，Independent
    Identical Distribution）

**参考**

[[https://www.tensorflow.org/probability/examples/Understanding\_TensorFlow\_Distributions\_Shapes]{.underline}](https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes)

[[https://bochang.me/blog/posts/pytorch-distributions/]{.underline}](https://bochang.me/blog/posts/pytorch-distributions/)

[[https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/]{.underline}](https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/)

超参数
------

超参数是在网络构型确定的情况下，需要人为确定的，无法学习的参数。对于一个网络来说，构型+超参数的组合决定了这个网络（以及它是否优秀）。

### 学习率

**学习率（Learning rate）**，学习的速度，通常被设置为0.001。

### Dropout Ratio

Dropout层每次随机关闭的神经元的比率。

### Batch size

单批次训练样本的数量。

NN训练相关
----------

### Epoch / Batch

在进行预测-反向传播-优化这个过程的时候，鉴于数据集的尺度，每次优化不可能对所有的数据都进行一遍计算，目前采用的方法是随机选择若干个数据元素，成为一个batch，在此基础上进行一次训练。

而一个epoch指的是将数据集中所有的数据都训练一遍，一个epoch中包含的训练次数（batch
number）为n/bs，n为数据集大小（比如图片集中总的图片数量），bs为batch
size（每个batch中包含的图片数量）。

### 过拟合（Overfit）

**过拟合（Overfitting）**指的是训练出来的网络过于精密的拟合训练数据，但不能很好的拟合不包含在训练数据中的其他数据（unseen
data），其表现是在训练集上的成绩远好于在测试集上的成绩。

过拟合的原因是因为相较于**有限的训练数据**（如果训练数据足够多，多到可以覆盖所有可能的数据，也不会出现过拟合），网络的**参数过多**或者结构过于复杂，从而使得过多的参数从统计噪声中获取了信息表达。

防止过拟合的方法包括模型选择、正则化、dropout、权值衰减、剪枝等等。

过拟合本身是个来自于统计学的概念。

### 学习率衰减

**学习率衰减（Learning Rate
Decay）**是神经网络训练中的一种技巧，随着学习的进行使学习率逐渐减小。在训练的初期快速的学习，使得模型更快速的接近最优解，而随着迭代的次数增加，学习率逐渐减小，使得模型在训练后期不会有太大波动。

### 权值衰减

**权值衰减（Weight
Decay）**是神经网络训练中的一种技巧，以减小权重参数的值为目的进行学习的方法，通过减小权重参数的值来抑制过拟合的发生。

### 梯度消失

**梯度消失（Gradient
Vanishing）**，是在梯度下降法中会出现的问题，由于输出不断靠近0或者1，使得其导数逐渐接近0，因此在反向传播中的梯度不断变小，最后消失。由于梯度的消失，导致学习速度降低，权重无法得到有效的更新，甚至神经网络完全无法继续训练。

### 超参数

**超参数（Hyper
Parameter）**：指的是各层的神经元数量、batch大小、学习率、权值衰减等神经网络学习的过程中不会更新，而需要人为设定的参数。

### 标准化（Normalization）

**标准化（Normalization）**也叫**正规化（Standardization），**在统计学中通常指的是将不同衡量标准下的值调整到一个统一的标准下，最常见的是0到1之间。

### 泛化（Generalization）

**泛化（Generalization）**这个词在机器学习中指的是消除过拟合，将模型应用于非训练数据。

CNN相关
-------

### 全连接层

**全连接层（Fully-connect
Layer）**，也叫Dense层，相邻两层的所有神经元都有连接。

### 卷积层

**卷积层（Convolution Layer）**，进行卷积运算的层，由卷积核组成。

### 池化层

**池化层（Pooling
Layer）**，进行缩小长和高方向上的空间的运算的层，没有可学习的参数，通常是Max池化，也有Average池化等。

### 卷积核

**卷积核（Convolution
Kernel）**，也叫过滤器（Filter），进行卷积运算的滤波器。

### 特征图

**特征图（Feature Map）**，卷积层的输入/输出数据

### 填充（Padding）

**填充（Padding）**，当需要输出特征图和输入特征图的大小（H\*W）保持一致的时候，需要向输入特征图的周围填入固定的数据（比如0）以让卷积核在边缘也可以正常工作

### 步幅（Stride）

**步幅（Stride）**：应用滤波器的位置间隔，步幅越大，输出特征图越小

### 通道（Channel）

**通道（Channel），**卷积运算输入数据的纵深维度。用于处理图像的卷积层是2维的，通道是除了长和高外的第三个维度，此外批量训练时还有第四个维度N（即batch
size）

Python相关
==========

Python是当前深度学习的主流语言，这里介绍了一些Python中难懂或易混淆的概念。而Python的基础的简单的语法，并不包含在这里。

完整的Python官方文档在这里：

[[https://docs.python.org/3/tutorial/index.html]{.underline}](https://docs.python.org/3/tutorial/index.html)

如果英文吃力或者只想快速入门Python：

[[https://www.runoob.com/python/python-tutorial.html]{.underline}](https://www.runoob.com/python/python-tutorial.html)

下文中：

蓝色表示package（或者目录）

红色表示module（或者文件）

绿色表示class

黑色表示其他（包括各种属性、函数、变量等）

青底黑字表示代码、命令行

包、模块、属性
--------------

关于Python的包、模块、属性的介绍。

-   **包**：python中的**包（package）**就是同名目录，且需要在该目录下有**\_\_init\_\_.py**表示该目录是一个package（其内容可以为空）。

-   **模块**：python中的**模块（module）**就是同名的py文件。

-   **属性**：module的**属性（attribute）**可以是函数、类、变量等。

    package的拓扑结构相当于目录的结构，即对应package1.package2（package2是package1的子package），目录package1有子目录package2。在\_\_init\_\_.py中可以import其他的package/module/attribute，从用户角度来看，类似于文件系统中的link或快捷方式。

package也是module，其内容是package对应目录中的\_\_init\_\_.py文件(你可以通过打印module.\_\_file\_\_来确认这点)。如果其中import了下属module的attribute，则相当于该package对应的module也有了这个attribute（就像快捷方式）。

模块文件（xxx.py或者\_\_init\_\_.py）中的**\_\_all\_\_**属性（是一个list类型的变量）用于显式定义from
module import \*时暴露出来的接口。如果没有\_\_all\_\_，则:

-   对于module来说，其中所有非下划线开头的成员都会被暴露出来

-   对于package来说，其所有非下划线开头的成员，和下级模块都会被暴露出来

from package.module import \*
受\_\_all\_\_的约束，只会导入被\_\_all\_\_暴露出来的attribute和子module，但是在直接from
package.module import xxx时,xxx不受\_\_all\_\_约束。

**import和from import**：

-   from A.B import
    C：A必须是package，B可以是package/module，C可以是package/module/attribute/\*，用时直接用C

-   import
    A.B.C：A、B必须是package，C可以是package/module，用时需要写全A.B.C（这个操作实际上相当于import
    A，因为之后也可以使用A的其他sub-package或者module）

**导入package:**

-   import package :导入package（作为module），之后可以直接用package

-   import package1.package2
    ：导入package1.package2进入命名空间，之后可以通过package1.package2使用package2。该操作实际是导入整个package1进入命名空间，也可以通过package1.xxx来使用package1的其他成员。

-   from package1 import package2
    ：直接导入package2进入命名空间，之后可直接使用package2

**导入module：**

-   import package.module
    ：导入package.module进入命名空间（其实是导入整个package进入命名空间），之后可以通过package.module使用module

-   from package import module
    ：直接导入module进入命名空间，之后可直接使用module

**导入attribute：**

-   from package.module import attribute：导入attribute，之后可以直接用

-   from package import
    attribute：同上，因为在package的\_\_init\_\_.py中导入module，因此此处的module可以忽略

**导入所有成员：**

-   from package1.package2 import
    \*：导入package2下所有的module和attribute，通过package2下\_\_init\_\_.py文件中的\_\_all\_\_变量指定，如果没有则导入所有非下划线开头的成员

-   from package.module import \*：导入module下所有的attribute

**错误格式：**

-   import package.attribute：错误，import a.b.c中c只能是module/package

-   import package.module.attribute ：错误，只有package之后才能跟"."

-   from package import module.attribute：错误，import之后不能有"."

总结一下就是：

-   可以import package/module

-   可以from package/module import package/module/attribute

-   **"."只能在package之后**

-   from xxx import yyy中，import之后不能有"."

类和对象
--------

Python中，类（Class）也是一个对象（Object）

### 对象

对象是Python组织数据的形式，所有的数据都是**对象（object）**，即某个**类（Class）**的instance。即便是整数，甚至整数常量这种简单的数据类型（其类为\<class
'int'\>)。每个对象都有**ID（identity）**，**类型（type）**和**值（value）**。这三者中，只有value是可以变化的，另外两个都是不可变的。

ID可以被视为对象在内存中的位置，内嵌函数**id()**返回了对象的ID，而**is**操作符则比较了两个对象的ID是否一致。

-   **type()**返回了对象的类型（即所属的类Class）：type(3)==int，除了常量之外（变量、类、包等）也可以用xxx.\_\_class\_\_得到同样结果

-   而类型本身也是一个对象，其类型是"type"：type(type(3))==type

-   而type的所属类型也是type：type(type(type(3)))==type

某些对象包含了其他对象（的引用），它们被称为**容器**（**Container**），比如list，tuple，dict等，这些引用是容器的值的一部分。

下文中：

-   对象（object）、实例（instance）基本同义

-   类型（type）、类（class）基本同义，在描述元类的时候略有区别：类型可以用来形容元类，而类不行

### 类=对象

定义某个类的一个对象，可以用如下语句：

object = class(args\...)

比如：

a = int(4)

或者

b = list(\[1,2,3\])

由于Python中所有的**类**也都是**对象**，因此这里的class()相当于class.\_\_call\_\_()，即一个object的可调用函数：

a = int.\_\_call\_\_(4)

b = list.\_\_call\_\_(\[1,2,3\])

Python中所有的**类**也都是**对象**，而这些对象的类型是'type'，或者说，所有类都是'type'的实例（包括'type'本身也是'type'的实例），如此一来，定义一个新的类，相当于：

class = type(classname, superclass, attributedict)

或者

class = type.\_\_call\_\_(classname, superclass, attributedict)

例如：

class NewClass:

data = 1

相当于：

NewClass = type("NewClass", (), {'data':1})

相当于：

NewClass = type.\_\_call\_\_("NewClass", (), {'data':1})

type是一个**metaclass**，即**元类**，元类是类的类型，元类和类的关系，一如类和对象的关系。

### metaclass

**元类（metaclass）**的对象是**普通类**，但是普通类的类可以不是'type'，也就时说除了'type'还可以定义其他的metaclass（**通常**这些metaclass的类型也是'type'）。

定义新的**metaclass**，并将**类**的元类设置为这个新**metaclass**的目的，通常是为了修改创建类的**对象**时的过程。

在类的定义中，通过声明metaclass=MyMeta，会使得：

-   在生成**类**的时候，如果有，MyMeta的\_\_new\_\_()和\_\_init\_\_()会被调用

-   在生成**类的对象**时，如果有，MyMeta类的\_\_call\_\_()替代type.\_\_call\_\_()被调用：

例如：

class Foo(metaclass=MyMeta)

会依次调用以下函数来创建Foo：

1.  type.\_\_call\_\_()：MyMeta的类型是'type'

    1.  MyMeta.\_\_new\_\_()

    2.  MyMeta.\_\_init\_\_()

foo = Foo()

会依次调用如下函数来创建Foo的实例foo：

1.  MyMeta.\_\_call\_\_()

    1.  Foo.\_\_new\_\_()

    2.  Foo.\_\_init\_\_()

**混淆注意**：

声明某个类的元类在Python3的用法是：

class Foo(metaclass=MyMeta)

在python2.7的用法则是如下：

class Foo:

\_\_metaclass\_\_=MyMeta

**混淆注意**：

注意区分"父类-子类"和"类型-对象"的关系，父类-子类能构成若干层的继承结构，但是（在不指定metaclass的情况下）所有这些类的类型都是'type'，而子类缺省和父类的元类一致，但可以通过声明metaclass来改变，简单来说，这是两个不同维度的概念。

**参考**

[[https://zhuanlan.zhihu.com/p/98440398]{.underline}](https://zhuanlan.zhihu.com/p/98440398)

[[https://lotabout.me/2018/Understanding-Python-MetaClass/]{.underline}](https://lotabout.me/2018/Understanding-Python-MetaClass/)

[[https://www.cnblogs.com/chengege/p/11102802.html]{.underline}](https://www.cnblogs.com/chengege/p/11102802.html)

### 实例化的过程

我们知道，如果类Foo的定义中有\_\_call\_\_函数的实现，则Foo的对象foo是可以调用的，即：

foo()

相当于调用foo所属类Foo的\_\_call\_\_函数：

Foo.\_\_call\_\_()

而当一个类Foo的对象foo生成的时候，发生的函数调用是这样的：

foo = Foo()

1.  调用Foo的所属类型的\_\_call\_\_：**type.\_\_call\_\_()**

    1.  调用Foo的\_\_new\_\_函数：**Foo.\_\_new\_\_()**

    2.  调用Foo的\_\_init\_\_函数：**Foo.\_\_init\_\_()**

当定义一个metaclass MyMeta的时候：

class MyMeta(type)：

相当于：

MyMeta = type("MyMeta", \...)

1.  调用type所属类型（仍是type）的\_\_call\_\_：**type.\_\_call\_\_()**

    1.  调用type的\_\_new\_\_函数：**type.\_\_new\_\_()**

    2.  调用type的\_\_init\_\_函数：**type.\_\_init\_\_()**

当定义一个metaclass为MyMeta的类的时候：

class Foo(metaclass=MyMeta):

相当于：

Bar = MyMeta("Bar", \...)

1.  调用MyMeta所属类型（type）的\_\_call\_\_函数：**type.\_\_call\_\_()**

    1.  调用MyMeta的\_\_new\_\_函数：**MyMeta.\_\_new\_\_()**

    2.  调用MyMeta的\_\_init\_\_函数：**MyMeta.\_\_init\_\_()**

        当生成Bar的对象bar的时候：

bar = Bar()

1.  如果Bar的所属类型MyMeta有定义\_\_call\_\_，则调用：**MyMeta.\_\_call\_\_()**

    1.  如果其中调用了self.\_\_new\_\_函数，则调用：**Bar.\_\_new\_\_()**

    2.  如果其中调用了self.\_\_init\_\_函数，则调用：**Bar.\_\_init\_\_()**

2.  否则，则调用**type.\_\_call\_\_()**

    3.  调用type的\_\_new\_\_函数：**type.\_\_new\_\_()**

    4.  调用type的\_\_init\_\_函数：**type.\_\_init\_\_()**

类
--

### Final和Virtual

先来看看C++和Java中的关于类成员函数的特点，并同时对Python中的情况做比较，下文混合使用两种语言中的名称，以便强调其特点（Final函数，虚函数，纯虚函数）：

-   **Final函数**：指的是函数不会被子类的继承覆盖

    -   C++的成员函数**缺省**就是这种函数，没有特别的名称

    -   Java中被称为**final method**，需要加**final前缀**来定义。

    -   Python中没有**Final函数**，但可以通过metaclass实现（比较复杂）

-   **虚函数**：函数会被子类的继承覆盖

    -   C++需要加**virtual前缀**来定义，被称为**virtual
        > function（虚函数）**

    -   Java中的成员函数**缺省**就是这种函数，没有特别的名称。

    -   Python中类的成员函数缺省就是**虚函数**

-   **纯虚函数**：函数只有声明没有实现，需要子类实现

    -   C++中需要加**virtual前缀**和**=0后缀**来定义，被称为**pure
        > virtual function（纯虚函数）**

    -   Java中称为**abstract
        > method（抽象方法）**，需要加**abstract前缀**来定义。

    -   **纯虚函数**在Python中被称为**abstract
        > method（抽象方法）**，可以通过**\@abstractmethod**修饰符来修饰函数，但仅仅如此无法阻止包含这种函数的类被直接实例化（需配合ABCMeta来使用）。

        再来看看C++和Java中关于类的特点，以及Python的比对：

<!-- -->

-   **普通类**：可以被继承，可以被实例化

    -   C++中的类缺省就是这种类

    -   Java中的类缺省就是这种类

    -   Python中的类缺省就是这种类

-   **Final类**：可以被实例化，但是不能被继承

    -   C++中直到C++11标准之后，才有关键字**final后缀**来定义

    -   Java中通过**final前缀**来定义这种类，被称为**final
        > class（final类）**

    -   Python中没有**Final类**，但可以通过metaclass实现

-   **抽象类**：**可能**包含**纯虚函数**，不能被实例化

    -   C++中被称为**abstract base
        > class（抽象基类）**，没有专门的关键字描述，类中只要包含**纯虚函数**就成为**抽象基类**，子类只有实现了所有**纯虚函数**才可以实例化

    -   Java中被称为**abstract
        > class（抽象类）**，通过**abstract前缀**来定义。且任何包含**纯虚函数（抽象方法）**的类都自动成为**抽象类**。不包含纯虚函数但有abstract修饰的类也不可以被实例化。

    -   Python中被称为**abstract base
        > class（抽象基类）**，没有专门的关键字来声明，但是可以通过以下方法来实现抽象类，注意这两者要同时使用，方可阻止类的实例化：

        -   类的声明中设置元类：metaclass=abc.ABCMeta

        -   并且用**\@abstractmethod**修饰**纯虚函数（抽象方法）**

-   **接口**：类似抽象类，但定义中仅包括**纯虚函数**，解决了单继承中的多继承问题。

    -   C++中是多继承的，没有接口的概念，不过有个类似的东西被称为**virtual
        > base
        > class（虚基类）**，本身的定义没有关键字描述，通过在子类继承的时候加上**virtual关键字**实现。虚基类可以解决**基类共享**的问题，这个问题在单继承的Java中不存在。

    -   Java中使用**interface关键字**定义接口，成员函数（可以不用显式指定）都是**纯虚函数（抽象方法）**，成员变量只能是static和final的。Java是单继承的，因此通过接口可以实现类似于C++多继承的特性。

    -   Python中没有**接口**，不过Python是多继承的，接口意义不大

        简化为下表：

  ----------- ---------------------------- ------------------------------------------ ------------------- -------------------------------------------------------------------------------------
  名称        特点                         C++                                        java                python

  Final函数   不会被子类覆盖               缺省                                       final method\       无
                                                                                      (前缀 final)        

  虚函数      会被子类覆盖                 虚函数\                                    缺省                缺省
                                           （前缀virtual关键字）                                          

  纯虚函数    只有声明没有实现，\          纯虚函数\                                  抽象函数\           抽象方法\
              必须被子类实现才可以实例化   (前缀virtual，后缀 =0)                     (前缀abstract)      （用\@abstractmethod 修饰的函数，但仅仅如此无法阻止实例化）

  普通类      可以被实例化\                缺省                                       缺省                缺省
              可以被继承                                                                                  

  Final类     不能被继承                   后缀final (C++11之后)                      前缀final           无（可以通过metaclass实现）

  抽象类      带有纯虚函数的类，\          抽象类（包含纯虚函数，没有特别的关键字）   抽象类\             抽象基类，实现如下：metaclass=abc.ABCMeta给需要抽象的方法加上\@abstractmethod修饰符
              不能被实例化                                                            (前缀abstract)      

  接口        只包含纯虚函数               无，但有类似的虚基类\                      接口\               无
                                           (继承时加virtual前缀)                      (interface关键字)   
  ----------- ---------------------------- ------------------------------------------ ------------------- -------------------------------------------------------------------------------------

### 抽象基类

Python中定义一个**抽象基类（Abstract Base
Class）**没有关键字来实现，需要通过设置类的元类来实现：

-   设置metaclass为abc.ABCMeta

-   函数要加上\@abstractmethod修饰符

    注意这两个操作都需要，只采取其中任意一种，类都可以被实例化。

from abc import ABCMeta, abstractmethod

class C(metaclass=ABCMeta):

\@abstractmethod

def func(self):

> pass

或者可以**继承ABC**作为其父类，这是个语法糖，ABC是个元类已经设置为ABCMeta的类，因此继承ABC相当于设置metaclass为ABCMeta：

from abc import ABC, abstractmethod

class C(ABC):

\@abstractmethod

def func(self):

> pass

对应文件为abc.py，给出了**抽象基类**（**ABC，abstract base
classes**）的定义，collections.abc中的各种具体的ABC均来源于此。这其中主要包含了：

-   **abc.ABCMeta**：ABC的元类（metaclass），定一个ABC需要通过如下格式实现：

    class MyABC(metaclass=ABCMeta)

-   **abc.ABC**：ABC的助手类（helper
    class），定义一个ABC也可以通过继承该类实现：class MyABC(ABC)

-   **abc.abstractmethod()**：函数修饰符，用于定义一个函数为抽象方法

    **参考**

[[https://docs.python.org/3/library/abc.html]{.underline}](https://docs.python.org/3/library/abc.html)

[[https://docs.python.org/3/glossary.html\#term-abstract-base-class]{.underline}](https://docs.python.org/3/glossary.html#term-abstract-base-class)

除了使用**ABCMeta**之外，还有两种方法可以近似模拟抽象基类------在基类创建的对象调用函数的时候报错，而不是在创建对象的时候。

1.  在基类的方法中使用断言assert，使得基类的对象调用函数的时候报错

    class BaseClass(object):

    def func(self):

    assert False, "Abstract class!"

2.  使用NotImplementedError异常，使得基类的对象调用函数的时候报错

    class BaseClass(object):

    def func(self):

    raise NotImplementedError("Not implemented!")

### 内嵌抽象类（Collections ABC）

**内嵌抽象基类（Collections
ABC）**，是系统中自带的一些**抽象基类（ABC，Abstract Base Class）**。

对应文件为\_collections\_abc.py，这个模块提供了一系列的**抽象基类**，这些抽象基类用于判断某个类是否实现了一个特定的**抽象方法**。比如Hashable类可用于判断某个类是否实现了\_\_hash\_\_()函数：

from typing import Hashable

isinstance(obj, Hashable)

换句话说，**只要**自定义的类实现了这些特定的抽象方法（比如\_\_hash\_\_），无须显式的继承这些类（比如Hashable），isinstance()函数就会返回True，这些自定义的类被称为**虚拟子类**（virtual
subclass）。

**参考**

[[https://docs.python.org/3/glossary.html\#term-abstract-base-class]{.underline}](https://docs.python.org/3/glossary.html#term-abstract-base-class)

下面为内嵌ABC的列表：

  ----------------- ------------------- ------------------------------------ -------------------------------------------------------------------------------------------------------------------------------------------
  **ABC**           **Inherits from**   **Abstract Methods**                 **Mixin Methods**

  Container                             \_\_contains\_\_                     

  Hashable                              \_\_hash\_\_                         

  Iterable                              \_\_iter\_\_                         

  Iterator          Iterable            \_\_next\_\_                         \_\_iter\_\_

  Reversible        Iterable            \_\_reversed\_\_                     

  Generator         Iterator            send, throw                          close, \_\_iter\_\_, \_\_next\_\_

  Sized                                 \_\_len\_\_                          

  Callable                              \_\_call\_\_                         

  Collection        Sized,\             \_\_contains\_\_,\                   
                    Iterable,\          \_\_iter\_\_,\                       
                    Container           \_\_len\_\_                          

  Sequence          Reversible,\        \_\_getitem\_\_,\                    \_\_contains\_\_, \_\_iter\_\_, \_\_reversed\_\_, index, and count
                    Collection          \_\_len\_\_                          

  MutableSequence   Sequence            \_\_getitem\_\_, \_\_setitem\_\_,\   Inherited Sequence methods and append, reverse, extend, pop, remove, and \_\_iadd\_\_
                                        \_\_delitem\_\_, \_\_len\_\_,\       
                                        insert                               

  ByteString        Sequence            \_\_getitem\_\_, \_\_len\_\_         Inherited Sequence methods

  Set               Collection          \_\_contains\_\_, \_\_iter\_\_,\     \_\_le\_\_, \_\_lt\_\_, \_\_eq\_\_, \_\_ne\_\_, \_\_gt\_\_, \_\_ge\_\_, \_\_and\_\_, \_\_or\_\_, \_\_sub\_\_, \_\_xor\_\_, and isdisjoint
                                        \_\_len\_\_                          

  MutableSet        Set                 \_\_contains\_\_, \_\_iter\_\_,\     Inherited Set methods and clear, pop, remove, \_\_ior\_\_, \_\_iand\_\_, \_\_ixor\_\_, and \_\_isub\_\_
                                        \_\_len\_\_, add, discard            

  Mapping           Collection          \_\_getitem\_\_, \_\_iter\_\_,\      \_\_contains\_\_, keys, items, values, get, \_\_eq\_\_, and \_\_ne\_\_
                                        \_\_len\_\_                          

  MutableMapping    Mapping             \_\_getitem\_\_, \_\_setitem\_\_,\   Inherited Mapping methods and pop, popitem, clear, update, and setdefault
                                        \_\_delitem\_\_, \_\_iter\_\_,\      
                                        \_\_len\_\_                          

  MappingView       Sized                                                    \_\_len\_\_

  ItemsView         MappingView,\                                            \_\_contains\_\_, \_\_iter\_\_
                    Set                                                      

  KeysView          MappingView,\                                            \_\_contains\_\_, \_\_iter\_\_
                    Set                                                      

  ValuesView        MappingView,\                                            \_\_contains\_\_, \_\_iter\_\_
                    Collection                                               

  Awaitable                             \_\_await\_\_                        

  Coroutine         Awaitable           send, throw                          close

  AsyncIterable                         \_\_aiter\_\_                        

  AsyncIterator     AsyncIterable       \_\_anext\_\_                        \_\_aiter\_\_

  AsyncGenerator    AsyncIterator       asend, athrow                        aclose, \_\_aiter\_\_, \_\_anext\_\_
  ----------------- ------------------- ------------------------------------ -------------------------------------------------------------------------------------------------------------------------------------------

**参考**

[[https://docs.python.org/3/glossary.html\#term-abstract-base-class]{.underline}](https://docs.python.org/3/glossary.html#term-abstract-base-class)

[[https://docs.python.org/3/library/collections.abc.html]{.underline}](https://docs.python.org/3/library/collections.abc.html)

**混淆注意！**

跟**抽象基类**相关的定义的包括：

-   **抽象基类**（即不能被实例化的抽象类）这个定义，又分为：

    -   用ABCMeta等手段实现，无法实例化

    -   虽然编程中仍然可以实例化，但在编码的架构设计中被定义为抽象基类

-   内嵌抽象类（Collections ABC）

-   abc中的ABC, ABCMeta这些用于实现抽象基类的工具类/元类

### 内嵌类型（built-in types）

[[https://docs.python.org/3/library/stdtypes.html\#]{.underline}](https://docs.python.org/3/library/stdtypes.html#)

### 类的层次

对象的**类型（type）**即其所属的**类（class）**，这也决定了该对象所能支持的操作。内嵌函数type(object)可以得到对象所属的类。

类的层次关系：

-   NoneType：只有一个对象，值唯一，即**None**

-   NotImplementedType：只有一个对象，值唯一，即**NotImplemented**

-   ellipsis：

-   numbers.Number：

    -   numbers.Integral：整数

        -   Integers：即**int**类型

        -   Booleans：即**bool**类型

    -   numbers.Real:即**float**类型

    -   numbers.Complex：即**complex**类型

-   Sequences：有序的有限元素的集合，其元素可以用非负整数索引（即从0开始）

    -   不可变Sequences：创建之后不可修改

        -   Strings：即**str**类型，可通过str.encode()函数编码为bytes类型

        -   Tuples：即**tuple**，元组类型

        -   Bytes：即**bytes**类型，可通过bytes.decode()函数变为str类型

    -   可变Sequences：创建之后可以修改

        -   Lists：即**list**类型

        -   Byte
            > Arrays：即**bytearray**类型，与**bytes**类型的区别在于可以修改

-   Set类型：无序的，有限元素的集合，可len()，可iter()

    -   Sets：可变集合，通过**set()**创建

    -   Frozen Sets：不可变的集合，通过**frozenset()**创建，可hash()

-   Mappings：映射，目前只有字典一个子类

    -   字典类：即dict类型

-   Callable类型：可调用的类型

    -   自定义函数：用户通过**def**定义

    -   实例方法：

    -   generator函数：使用**yield**语句的函数

    -   Coroutine函数：协程函数

    -   异步generator函数：

    -   built-in function：内嵌函数

    -   built-in method：内嵌方法

    -   Class：就是类，所有的类都是可调用的，返回该类的对象

    -   class instance：类的实例，可以通过实现\_\_call\_\_()来变得可调用

-   模块：Python中的模块都是**module**类的对象

-   Custom classes：

-   Class instance：

-   I/O objects：

-   Internal types：

    -   Code objects：

    -   Frame objects：

    -   Traceback objects：

    -   Slice objects：

    -   Static method objects：

    -   Class method objects：

**参考**

[https://docs.python.org/3/reference/datamodel.html]{.underline}

泛型别名
--------

**泛型别名（Generic
Alias）**通过给一个类（通常是容器）做下标创建，比如list\[int\]。泛型别名没有强制性。

通常来说，给容器类的对象下标会调用**类特殊方法**\_\_getitem\_\_()函数，而给容器类做下标，则会调用**类特殊方法**\_\_class\_getitem\_\_()，这会返回一个**泛型别名**对象。

**T\[X, Y, \...\]**

表示了包含了X,
Y等类型的容器T，例如（List和Dict为list和dict类型的别名）：

from typing import List, Dict

List\[float\]

Dict\[int, str\]

以上方式在python3中通用，下面不用类型别名直接用类型名的方法在**3.9之后**才合法：

list\[float\]

dict\[int, str\]

Python并不做类型检查，因此下列语句是合法的：

\>\>\> t = list\[str\]

\>\>\> t(\[1, 2, 3\])

\[1, 2, 3\]

而在生成实例的时候会消除类型参数的信息：

\>\>\> t = list\[str\]

\>\>\> type(t)

\<class \'types.GenericAlias\'\>

\>\>\> l = t()

\>\>\> type(l)

\<class \'list\'\>

泛型别名有如下一些属性：

-   generalalias.\_\_origin\_\_：指代不包含参数的原泛型类型

\>\>\> list\[int\].\_\_origin\_\_

\<class \'list\'\>

-   generalalias.\_\_args\_\_：元组类型，泛型类的参数类

\>\>\> dict\[str, list\[int\]\].\_\_args\_\_

(\<class \'str\'\>, list\[int\])

-   generalalias.\_\_parameters\_\_：

**参考**

[[https://docs.python.org/3/library/stdtypes.html\#generic-alias-type]{.underline}](https://docs.python.org/3/library/stdtypes.html#generic-alias-type)

方法
----

### 标准库抽象类、内嵌函数、类特殊方法

-   **标准库抽象类（standard library abstract
    class）**，指的是标准库中定义的一系列**抽象基类**。

-   **内嵌函数（built-in
    function）**，指的是python的内嵌命名空间（built-in
    namespace）中自带的函数，不属于任何类

-   **类特殊方法**，指的是具有特殊名字的类，通常以"\_\_"（两个下划线）开头和结尾，通常是**标准库抽象类**中的**抽象方法**。

这三者在Python中有相关性，某个类对**内嵌函数**的支持需要类实现某个**内置虚函数**，而某个**标准库抽象类**可以用于判断是否实现了这个虚函数。

举个例子，关于哈希，以下三者等价：

-   系统的**内嵌函数**hash(object)能运行成功

-   object所属的类实现了\_\_hash\_\_()这个**内置虚函数**（或者叫**类特殊方法**）

-   该类是Hashable这个**标准库抽象类**的子类（或者object是Hashable的实例）

### 内嵌函数（Built-in Functions）

**内嵌函数**是**内嵌命名空间**（**built-in
namespace**）中自带的函数，不属于任何类。

Python有一系列内嵌函数，包括：

-   abs(x)：返回一个数字的绝对值，如果x定义了\_\_abs\_\_()，则返回x.\_\_abs\_\_()

-   all(iterable)：参数为迭代器，如果该迭代器产生的所有元素都为True则返回True，否则返回False

-   any(iterable)：参数为迭代器，如果迭代器的任意元素为True则返回True，否则返回False

-   ascii()：

-   bin()

-   bool()

-   breakpoint()

-   bytearray()

-   bytes()

-   callable()：返回布尔值，表示是否可以调用。所有类都是可以调用的，调用类返回该类的对象。对象是否可以被调用基于该类是否实现了\_\_callable\_\_()成员函数。（也可以通过判断该类是否Callable的子类来判断）

-   chr()

-   \@classmethod：函数修饰符，用于将类的成员函数声明为类函数，不跟对象绑定，隐性的第一个参数也不是对象（self），而是类（cls）

-   compile()

-   complex()

-   delattr()

-   dict()

-   dir()：不带参数则显示当前scope中的内容，带参数则返回给定object的属性

-   divmod()

-   enumerate()

-   eval()

-   exec()

-   filter()

-   float()

-   format()

-   frozenset()

-   getattr()

-   globals()

-   hasattr()

-   hash()

-   help()

-   hex()

-   id()

-   input()

-   int()

-   isinstance(object, class)：判断object是否是class的实例，返回bool值

-   issubclass(class1, class2)：判断class1是否是class2的子类，返回bool值

-   iter(object)：返回该object的迭代器，object需满足迭代协议（由\_\_iter\_\_()函数实现），或者满足序列协议（由\_\_getitem\_\_()实现）

-   len()

-   list()

-   locals()

-   map()

-   max()

-   memoryview()

-   min()

-   next()

-   object()

-   oct()

-   open()

-   ord()

-   pow()

-   print()

-   property()

-   range()

-   repr()：

-   reversed()

-   round()

-   set()

-   setattr()

-   slice()

-   sorted()

-   staticmethod()

-   str()

-   sum()

-   super()

-   tuple()

-   type()

-   vars()

-   zip()

-   \_\_import\_\_()

**参考**

[[https://docs.python.org/3/library/functions.html]{.underline}](https://docs.python.org/3/library/functions.html)

### 类特殊方法（Special method）

类可以通过实现特别的类成员方法（special
method，这些方法的方法名前后有两个下划线，Python中此类名称经常有特殊意义），使得这些方法在某些特殊时候被回调，比如用于支持某些**内嵌函数**（len(),
dir()等），或者支持某些**语句**（with,if等），这也是Python实现**运算符重载**的方式。

举个例子，如果对象x的类实现了\_\_getitem\_\_()方法，则x\[i\]相当于x.\_\_getitem\_\_(i)，即
type(x).\_\_getitem\_\_(x,i)。（可以用list类型验证此方法）

**参考**

[[https://docs.python.org/3/reference/datamodel.html\#special-method-names]{.underline}](https://docs.python.org/3/reference/datamodel.html#special-method-names)

### 基本方法

-   object.**\_\_new\_\_**(cls,\...)：静态方法，创建一个新的实例，第一个参数是类名，构造一个新的对象的时候，在\_\_init\_\_()之前被调用。其常用的一个用法是，子类的新建对象调用父类的该方法，即super().\_\_new\_\_()。

-   object.**\_\_init\_\_**(self,\...)：构造函数，被调用的时候对象已经创建（通过\_\_new\_\_()函数），即第一个参数self。常用的用法是在子类的构造函数中调用，即super().\_\_init\_\_()。

-   object.**\_\_del\_\_**(self)：对象被销毁之前调用，类似于解构函数

-   object.**\_\_repr\_\_**(self)：调用内嵌函数repr(object)时被调用，用于返回该对象的"官方"字符串表示。

-   object.**\_\_str\_\_**(self)：调用内嵌函数str(object)和print(object)的时候被调用，用于返回该对象的"非正式"或者"打印友好"的字符串表示。

-   object.**\_\_bytes\_\_**(self)：字节序化，调用内嵌函数bytes(object)的时候被调用，返回对象的字节串（byte-string）表示（bytes类型）。

-   object.**\_\_format\_\_**(self,
    format\_spec)：调用内嵌函数format(object)的时候被调用，目前等同于str(object)

-   object.**\_\_lt\_\_**(self, other)：对"\<"操作符的重载

    object.**\_\_le\_\_**(self, other)：对"\<="操作符的重载

    object.**\_\_eq\_\_**(self, other)：对"=="操作符的重载

    object.**\_\_ne\_\_**(self, other)：对"!="操作符的重载

    object.**\_\_gt\_\_**(self, other)：对"\>"操作符的重载

    object.**\_\_ge\_\_**(self, other)：对"\>="操作符的重载

-   object.**\_\_hash\_\_**(self)：哈希值的计算，调用内嵌函数hash(object)的时候被调用，返回对象的哈希值

-   object.**\_\_bool\_\_**(self)：判断语句的时候（比如if），以及调用内嵌函数bool(object)的时候被调用（如果\_\_bool\_\_()未被定义，则\_\_len\_\_()被调用）

    2.  ### 属性访问

-   object.**\_\_getattr\_\_**(self,
    name)：当\_\_getattribute\_\_()或者\_\_get\_\_()调用出现AttributeError异常时被调用

-   object.**\_\_getattribute\_\_**(self, name)：返回对象的某个属性

-   object.**\_\_setattr\_\_**(self, name, value)：设置对象的某个属性

-   object.**\_\_delattr\_\_**(self, name)：当del
    object.name的时候被调用，删除对象的某个属性

-   object.**\_\_dir\_\_**(self)：调用内嵌函数dir(object)的时候被调用，返回对象的所有属性

-   object.**\_\_get\_\_**(self, instance)：

-   object.**\_\_set\_\_**(self, instance, value)：

-   object.**\_\_delete\_\_**(self, instance)：

-   object.**\_\_set\_name\_\_**(self, owner, name)：

    3.  ### 其他

-   object.**\_\_call\_\_**(self,
    args\...)：实现此方法则对象可被调用，当实例被作为一个函数调用的时候，\_\_call\_\_()被调用，即object(args\...)。

-   object.**\_\_len\_\_**(self,
    name)：调用内嵌函数len(object)的时候被调用，bool判断的时候如果没有\_\_bool\_\_函数而有\_\_len\_\_函数，则会调用该函数判断bool值，0返回False。

-   object.**\_\_getitem\_\_**(self, key)：实现了object\[key\]的重载

-   object.**\_\_setitem\_\_**(self, name,
    value)：实现了object\[key\]=value的重载

-   object.**\_\_iter\_\_**(self)：参考[[迭代器]{.underline}](\l)

-   object.**\_\_contains\_\_**(self, item)：实现了item in
    object的重载，返回布尔值

    4.  ### with语句上下文

-   object.**\_\_enter\_\_**(self,)：进入object的with上下文环境时被调用

-   object.**\_\_exit\_\_**(self, exc\_type, exc\_value,
    traceback)：退出object的with上下文环境时被调用

    5.  ### 协程相关

-   object.**\_\_await\_\_**(self)：

-   object.**\_\_aiter\_\_**(self)：

-   object.**\_\_anext\_\_**(self)：

-   object.**\_\_aenter\_\_**(self)：

-   object.**\_\_aexit\_\_**(self, exc\_type, exc\_value, traceback)

hash()，memoryview()，set()，all()，

对于类CLASS，可以实现如下虚函数，这些虚函数会用于对于CLASS类型的对象object

\_\_len\_\_()：对象的长度，len(object)时候调用

\_\_bool\_\_()：判断object是否为True时候调用（if
object），如果没有实现返回True

### 修饰符

**修饰符（decorator）**其实是个语法糖，**修饰符**本质上也是个函数de，这个函数d的参数是个函数func（即被修饰的函数），返回另一个函数wrapper，这个函数会被用于替代原来的func，使得每次调用func的时候都实际是在调用wrapper，而在使用修饰符的时候都相当于调用了一次de，即使用的时候：

\@de

def func():

\...

相当于：

def func():

\...

func = de(func)

即用de函数返回的函数替代func。

举个简单的例子：

def log(f): \#定义修饰符函数

def wrapper():

> print("start")
>
> return f() \#当然你也可以完全不调用原函数f，不过那就失去意义了

return wrapper \#wrapper用于替换被修饰函数

\@log

def func():

print("function")

func()

运行后打印：

start

function

**参考**

[[https://docs.python.org/3/glossary.html\#term-decorator]{.underline}](https://docs.python.org/3/glossary.html#term-decorator)

迭代器
------

迭代器这玩意单独拿出来说，是因为有点绕。简单来说如下：

有2个和迭代相关的概念，分别是Iterator（迭代器）和Iterable（迭代器的容器）：

-   Iterable：任何类实现了\_\_iter\_\_()方法即是**内嵌抽象类**Iterable的**虚拟子类**，该方法返回一个Iterator，检查isinstance(obj,
    Iterable)可以判断obj是否有\_\_iter\_\_方法，但不能检查通过\_\_getitem\_\_方法迭代的类，唯一可靠判断类是否iterable的方式是通过iter()

-   Iterator：任何类实现了\_\_iter\_\_()和\_\_next\_\_()方法，即是**内嵌抽象类**Iterator的**虚拟子类**（从而必然是Iterable的虚拟子类）：

    -   \_\_iter\_\_()：Iterator的该方法返回自身

    -   \_\_next\_\_()：返回当前元素，并将Iterator指向下一个元素

如果你仍然觉得不满足，可以继续往下看。

### 定义

python中迭代器相关的一些定义。

### iterable

可以每次返回其一个成员的对象，比如所有的sequence类型（list, tuple,
str），某些非sequence类型（比如dict），或者任何定义了\_\_iter\_\_()方法或者\_\_getitem\_\_()方法的。

iterable可以被用于for循环。当一个iterable对象被作为参数传递给内嵌函数iter()时，返回该对象的iterator。当使用iterable的时候，通常无须调用iter()或者自己来处理iterator，for语句会自动处理这些：创建一个临时无名变量来表示iterator，直到循环结束。

**参考**

[[https://docs.python.org/3/glossary.html\#term-iterable]{.underline}](https://docs.python.org/3/glossary.html#term-iterable)

### iterator

一个表示数据流的对象，持续的调用iterator的**类特殊方法**\_\_next\_\_()，或者将iterator传递给**内嵌函数**next()，会持续的返回流中的数据。当没有后续的数据时，会引起StopIteration异常。

iterator必须要实现\_\_iter\_\_()方法，以便返回自身，因此所有的iterator也是iterable。可以被用在所有iterable的场景，但也有例外，当一个容器对象（比如list）每次被传递给iter()的时候（或者在for循环中），都会返回一个新的iterator，如果此时用该iterator试图继续，则会使得其看上去像一个空的容器。（FIXME：存疑，测试没有此问题）

**参考**

[[https://docs.python.org/3/glossary.html\#term-iterator]{.underline}](https://docs.python.org/3/glossary.html#term-iterator)

### generator

一个返回genertor
iterator的函数。跟普通函数的区别在于包含了yield语句，yield语句产生一系列数值，可以用于for循环，或者通过内嵌函数next()获得。

通常指代一个generator函数，但是在某些情况下也可以指代generator
iterator。

**参考**

[[https://docs.python.org/3/glossary.html\#term-generator]{.underline}](https://docs.python.org/3/glossary.html#term-generator)

### generator iterator

generator函数产生的对象。每次执行到yield语句，会暂停执行绪列，保留当前的上下文，直到下次执行的时候恢复（相较于此，函数的每次执行都会从头开始）。

**参考**

[[https://docs.python.org/3/glossary.html\#term-generator-iterator]{.underline}](https://docs.python.org/3/glossary.html#term-generator-iterator)

### 内嵌抽象基类

### Iterable

描述iterable的**内嵌抽象基类**，调用isinstance(obj,
Iterable)可以用于验证obj是否实现了**类特殊方法**\_\_iter\_\_()，但不能检查通过\_\_getitem\_\_()迭代的类，唯一可靠判断类是否iterable的方式是调用iter()。

Iterable可以被视为是Iterator的容器，通过将Iterable作为参数传给iter()，返回一个Iterator。

**参考**

[[https://docs.python.org/3/library/collections.abc.html\#collections.abc.Iterable]{.underline}](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)

### Iterator

描述iterator的**内嵌抽象基类**，调用isinstance(obj,
Iterator)可以用于验证obj是否实现了**类特殊方法**\_\_iter\_\_()和\_\_next\_\_()

**参考**

[[https://docs.python.org/3/library/collections.abc.html\#collections.abc.Iterator]{.underline}](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)

### Reversible

Reversible实现了\_\_reversed\_\_()的iterable的**内嵌抽象基类**，调用isinstance(obj,
Reversible)可以用于验证obj是否实现了**类特殊方法**\_\_iter\_\_()和\_\_reversed\_\_()。

**参考**

[[https://docs.python.org/3/library/collections.abc.html\#collections.abc.Reversible]{.underline}](https://docs.python.org/3/library/collections.abc.html#collections.abc.Reversible)

### Generator

描述generator的**内嵌抽象基类**，它实现了PEP
342中定义的协议：在iterator的基础上实现了send(),throw()和close()方法。

**参考**

[[https://docs.python.org/3/library/collections.abc.html\#collections.abc.Generator]{.underline}](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)

### 类特殊方法

-   **\_\_iter\_\_()**：返回一个迭代器Iterator，实现该方法的类是Iterable的**虚拟子类**，Iterator的该方法返回自身。

-   **\_\_next\_\_()**：返回当前元素，并将Iterator指向下一个元素

多进程
------

**参考**

[【Multiprocessing系列】Multiprocessing基础]{.underline}

[【Multiprocessing系列】Process]{.underline}

[【Multiprocessing系列】Pool]{.underline}

[【Multiprocessing系列】子进程返回值]{.underline}

[[【Multiprocessing系列】共享资源]{.underline}](https://thief.one/2016/11/24/Multiprocessing%E5%85%B1%E4%BA%AB%E8%B5%84%E6%BA%90/)

框架&接口
=========

接下来几章介绍了和机器学习相关的库和框架，基本都是基于Python的。本章主要介绍的是矩阵相关的python库。

下一章则单独列出了当前比较热门的一些深度学习框架。

框架简介
--------

本章之后的小节列出了一些传统库和DL框架的接口，这其中绝大部分（除了**Darknet**）均为Python接口。

当前最热门的三个深度学习框架为Facebook的**Pytorch**和Google的**Tensorflow**/**Keras**，以及Amazon的**MXNet/Gluon**，从普及程度上来讲，此三者为第一集团，其中**MXNet/Gluon**目前略微落后，而**Pytorch**有反超**Tensorflow**的趋势。

此外还有**CNTK**（微软），**sklearn**（并非是个纯DL框架，而是个ML框架），**Theano**（已经停止更新）等。

而**Darknet**作为一个YOLO作者的独立作品，在YOLO用户中被使用。

框架按照其作用，可以大致分为通用框架、专用框架、边缘计算库和快捷接口，这其中除了边缘计算库之外，其他三种之间都略有交叉。

+-------------+-------------+-------------+-------------+-------------+
| 出品方      | 通用框架    | 快捷接口    | 边缘计算    | 专用框架    |
+-------------+-------------+-------------+-------------+-------------+
| Google      | Tensorflow  | Keras       | tflite      |             |
+-------------+-------------+-------------+-------------+-------------+
| Facebook    | Pytorch(tor | Fast.ai?    | Pytorch     | torchvision |
|             | ch)         |             | Mobile      | （图像处理） |
|             |             |             |             |             |
|             | Caffe/Caffe |             |             |             |
|             | 2(并入Pytorch |           |             | Detectron（已 |
|             | )           |             |             | 废弃）      |
|             |             |             |             |             |
|             |             |             |             | Maskrcnn-be |
|             |             |             |             | nchmark（已废弃 |
|             |             |             |             | ）          |
|             |             |             |             |             |
|             |             |             |             | Detectron2  |
|             |             |             |             | (CV)        |
|             |             |             |             |             |
|             |             |             |             | PySlowFast（ |
|             |             |             |             | 视频理解）  |
|             |             |             |             |             |
|             |             |             |             | Prophet（时间序 |
|             |             |             |             | 列）        |
+-------------+-------------+-------------+-------------+-------------+
| Amazon      | MXNet       | gluon       |             | gluoncv（CV） |
|             |             |             |             | \           |
|             |             |             |             | gluonnlp（NL |
|             |             |             |             | P）         |
|             |             |             |             |             |
|             |             |             |             | gluon-ts（时间 |
|             |             |             |             | 序列）      |
+-------------+-------------+-------------+-------------+-------------+
| 腾讯        |             |             | ncnn (C++)  |             |
|             |             |             |             |             |
|             |             |             | TNN         |             |
+-------------+-------------+-------------+-------------+-------------+
| 阿里        |             |             | MNN (C++)   |             |
+-------------+-------------+-------------+-------------+-------------+
| 百度        | PaddlePaddl |             | Paddle-lite |             |
|             | e(飞桨）    |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| 华为        | MindSpore   |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| 港中文大学  |             |             |             | mmdetection |
|             |             |             |             | （CV，基于Pytor |
|             |             |             |             | ch）        |
|             |             |             |             |             |
|             |             |             |             | mmaction（视频 |
|             |             |             |             | 理解）      |
+-------------+-------------+-------------+-------------+-------------+

### 传统库

传统库指的是一些并非NN，但和NN有关的库，都有Python的实现。

-   **NumPy**：多维数组和矩阵运算的基础库，其核心数据结构是ndarray（n-dimensional
    array）。

-   **Matplotlib**：NumPy的可视化界面库

-   **SciPy**：开源的算法库和数学工具包，包含最优化、线性代数、积分、插值、FFT、信号和图像处理、常微分方程等。

-   **Pandas**：基于NumPy的开源Python库，用于快速分析数据、数据清洗等工作。

-   **Scikit-learn**：Scikit = SciPy
    ToolKit，开源的机器学习库，支持多种分类、回归、聚类算法，比如SVM、随机森林、k-means等。可以和NumPy及SciPy交互。

-   **Scikit-image**：开源的Python图像处理库，可以和NumPy及SciPy交互。

-   **PIL/Pillow**：PIL（Python Imaging
    Library），开源的Python图像处理库，于2011年停止开发侯，后续项目Pillow从PIL的仓库fork出来。

-   **Opencv：**开源的机器视觉库（CV：Computer Vision），可以与Numpy交互

### 通用框架

通用框架是NN中最基础的库，通常实现了以下内容：

-   神经网络的各种基本构成，包括基类及各种实现，比如各种卷积层、池化层、损失函数、激活函数、优化方法、标准化方法等等

-   数据集相关的工具，下载、预处理、数据增广等等

-   有些通用框架还包括了更底层的东西（比如Tensorflow）

通用框架可以用于训练（反向传播）和前向推导，目前最火的两个分别是来自Google的Tensorflow和来自Facebook的Pytorch，第三名则是Amazon的MXNet。

-   **Tensorflow**：这是当前最火的机器学习库之一，来自谷歌大脑团队，于2015年开源。

-   **PyTorch**：基于Torch的开源Python机器学习库，由Facebook开源。2018年Caffe2并入PyTorch，近年来有超越Tensorflow的趋势。

-   **Caffe/Caffe2**：Convolutional Architecture for Fast Feature
    Embedding，开源的机器学习库，来自UB Berkley
    。2017年，Facebook发布Caffe2，加入了RNN等新功能，2018年，Caffe2并入PyTorch。

-   **CNTK**：Microsoft Cognitive
    Toolkit，微软推出的机器学习库，编程语言是C++。

-   **MXNet**：Apache的开源机器学习框架，支持多种语言。

-   **Darknet**：是YOLO系列作者Joseph
    Redmon开发的一个框架，C语言实现，也有一个同名的基础卷积网络模型。

-   **Theano**：开源的Python数学库，用于定义、优化、求值数学表达式。

-   **PaddlePaddle**：百度推出的DL框架

-   **MindSpore**：华为的DL框架

### 专用框架

专用框架、库、工具集，专门针对某些类别任务的高层次框架，**通常基于某通用框架**。专用框架通常包括：

-   面向该领域的特殊的**层**或者**微结构**

-   已经实现的该领域任务的**网络模型**，及其**相关工具**，比如训练器、权重下载等

-   该领域**数据集的相关工具**，比如下载、格式转换等

-   该领域性能的**衡量工具**

具体的专用框架有：

-   **torchvision**：**Facebook**出品，Pytorch的子集，**机器视觉**方面的框架，包括：

    -   **图像分类**

    -   **语义分割**

    -   少量**目标检测**、**实例分割**、**人体关键点检测**

    -   也包括少量视频理解方向的**行为识别**（R3D）。

-   **Detectron2**：**Facebook**推出的**图像处理**的框架，基于Pytorch，包括：

    -   **图像分类**

    -   **目标检测**

    -   **图像分割：语义分割、实例分割、全景分割**

    -   **人体关键点检测**

-   **PySlowFast：Facebook**推出的视频理解框架，基于Pytorch，包括：

    -   **行为识别**：SlowFast、Slow、I3D、C2D、Non-local network

    -   **时序行为检测**：

-   **gluoncv：gluon**（Amazon）的一部分，**机器视觉**方面的框架，包括：

    -   **图像分类**

    -   **目标检测**

    -   **图像分割**

    -   **姿态评估（人体关键点检测）**

    -   **行为识别（SlowFast、I3D）**

-   **gluonts**：**gluon**（Amazon）的一部分，**时间序列**方面的框架

-   **gluonnlp**：**gluon**（Amazon）的一部分，**NLP**方面的框架

-   **mmdetection**：**香港中文大学**推出的**图像处理**的工具集，基于Pytorch，包括：

    -   **图像分类**

    -   **目标检测**

    -   **图像分割：语义分割、实例分割、全景分割**

-   **mmsegmentation**：香港中文大学推出的**图像分割**工具集，基于Pytorch

-   **mmaction**：**香港中文大学**推出的**视频理解**的工具集，基于Pytorch，包括：

    -   **动作识别**（Action Recognition）

    -   **时序动作检测**（Temporal Action Detection）

    -   **时空动作检测**（Spatio-temporal Action Detection）

-   **SimpleDet**：**图森**推出的**图像处理**的框架，基于MXNet，包括：

    -   目标检测

    -   实例分割

-   **PyVideoResearch**：**视频理解**方面的框架

-   **SpaCy**：**自然语言处理（NLP）**开源库，偏重于产品。

-   **NLTK**：**N**atural **L**anguage
    **T**ool**K**it，自然语言处理（NLP）开源库，偏重于研究教学

-   **Hugging Face Transformer**：Hugging
    Face公司推出的基于Transformer的NLP框架

### 边缘计算

用于移动端推理的前向框架。用于部署在手机、平板、IoT等嵌入式设备上，通常适用于对算力要求较小的网络，**只能用于前向推理**，没有训练能力。一般来说由以下两者组成：

-   **converter**：**转换器**，用于将其他格式的model转换为该框架的格式

-   **interpreter**：**解释器**，用于在嵌入式设备上运行该框架的格式的model

下图以MNN为例：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image79.png){width="5.4631944444444445in"
height="1.7652777777777777in"}

具体的边缘计算框架包括：

-   **Tensorflow lite**：**Google**的移动端计算框架

-   **ncnn**：**腾讯**的移动端计算框架

-   **TNN：腾讯**的移动端计算框架

-   **MNN：阿里巴巴**的移动端计算框架

-   **Paddle-lite：百度**的移动端计算框架

-   **QNNPACK：Facebook**的移动端计算框架

-   **CoreML**：**苹果**的移动端计算框架

### 快捷接口

快捷接口用户友好，增加开发调试速度，但灵活性差，类似于高级语言。快捷接口基于某种通用框架。

具体的快捷接口包括：

-   **Keras**：**Google**推出的Tensorflow的前端接口，Python。专注于用户友好，后端的实现目前已经支持TensorFlow、CNTK、Theano。

-   **Gluon**：**Amazon**推出的MXNet的前端接口。

-   **Fast.ai**：**Facebook**推出的Pythorch的前端接口。

**其他：**

-   **ONNX**：**Open Neural Network
    Exchange**，是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。它使得不同的人工智能框架（如Pytorch、MXNet）可以采用相同格式存储模型数据并交互。
    ONNX的规范及代码主要由微软，亚马逊，Facebook和IBM等公司共同开发，以开放源代码的方式托管在Github上。目前官方支持加载ONNX模型并进行推理的深度学习框架有： Caffe2,
    PyTorch, MXNet，ML.NET，TensorRT 和 Microsoft
    CNTK，并且 TensorFlow 也非官方的支持ONNX。

不同格式间的转换
----------------

模型在不同的**通用框架**以不同的格式存储。从存储的位置上来说，从存储的内容来看，分为模型和权重；包括内存中的模型和磁盘上的文件。文件格式有以下：

### 文件格式

-   **Keras**

**.h5/.hdf5**：keras的缺省存储格式，可以用于存储权重或者模型+权重

**.json**：JSON格式，被keras用来存储模型

-   **TensorFlow**

参考《[[NN框架 \> Google \> Tensorflow \>
模型文件格式]{.underline}](\l)》

-   **TensorFlow Lite**

**.tflite**：tensorflow
lite解释器识别的格式，不能直接训练，而是通过其它（主要是tensorflow和h5）转换

-   **MXNet**

**.json**：JSON格式，用于存放MXNet的模型

**.params**：用于存放MXNet框架的权重

-   **Caffe**

**.prototxt**：描述网络模型的文件

**.caffemodel**：描述权重（及模型）的文件

-   **Caffe2**

**.predict-net.pb**：描述权重的文件?

**.init-net.pb**：描述网络模型的文件?

-   **Cntk**

**.model**：模型和权重文件

-   **Pytorch**

.**pth**：模型和权重的文件

.**pkl**：模型和权重的文件，pickle格式

-   **ncnn**

**.bin**：权重文件

**.param**：模型文件

-   **TNN**

**.tnnmodel**：权重文件

**.tnnproto**：模型文件

-   **darknet**

**.cfg**：模型配置、超参数及训练参数

**.weights**：权重文件

-   **CoreML**

**.mlmodel**：模型和权重的文件

### 格式转换

  ---------------- --------------------------------- --------------------------------- -------------------------------------------------- ----------------------------------------------------------------------------------------- ----------------------------------------- --------------------------------------------------
  **to \\ from**   tensorflow                        pytorch                           onnx                                               keras                                                                                     darknet                                   caffe2

  tensorflow       　                                mmconvert                         onnx-tf ( cmdline tool, in onnx\_tf py pacakge)    keras\_to\_tensorflow( from github, https://github.com/amir-abdi/keras\_to\_tensorflow)   　                                        　

  pytorch          mmconvert\                        　                                　                                                 mmconvert                                                                                 　                                        　
                   ( in package mmdnn)                                                                                                                                                                                                                                        

  onnx             mmconvert                         mmconvert                         　                                                 keras2onnx (python package)                                                               　                                        convert-caffe2-to-onnx\
                                                                                                                                                                                                                                                                              (in package onnx-caffe2, now merged into caffe2)

  keras            mmconvert                         pytorch2keras (python package)\   onnx2keras( python package)                        　                                                                                        convert.py ( from github repo\            　
                                                     use onnx2keras in backend                                                                                                                                                      https://github.com/qqwweee/keras-yolo3)   

  caffe            mmconvert                         mmconvert                         　                                                 　                                                                                        　                                        　

  caffe2           mmconvert                         mmconvert                         convert-onnx-to-caffe2\                            　                                                                                        　                                        　
                                                                                       (in package onnx-caffe2, now merged into caffe2)                                                                                                                                       

  tflite           tflite\_convert (cmdline tool,\   　                                　                                                 tflite\_convert (cmdline tool,\                                                           　                                        　
                   in tensorflow py package)                                                                                              in tensorflow py package)                                                                                                           

  ncnn             　                                　                                onnx2ncnn                                          　                                                                                        　                                        caffe2ncnn

  coreML           mmconvert( in package mmdnn,\     mmconvert                         　                                                 　                                                                                        　                                        　
                   depend on coremltools)                                                                                                                                                                                                                                     
  ---------------- --------------------------------- --------------------------------- -------------------------------------------------- ----------------------------------------------------------------------------------------- ----------------------------------------- --------------------------------------------------

  ------------------------------------------------------
  works fine ( at least 1 model converted and tested )
  seems to be fine (model converted, but not tested)
  no model converted successfully
  not test yet
  ------------------------------------------------------

#### Tensorflow to ...

-   tensorflow→pytorch(model+weights)

mmconvert -sf tensorflow -iw xxx.pb \--inNodeName inputNode
\--inputShape 224,224,3 \--dstNodeName outputNode -df pytorch -om
xxx.pth

-   tensorflow→caffe(model+weights)

mmconvert -sf tensorflow -iw xxx.pb \--inNodeName inputNode
\--inputShape 224,224,3 \--dstNodeName outputNode -df caffe -om xxx

生成xxx.prototxt和xxx.caffemodel

#### Keras to ...

-   keras→h5(model+weights)

model.save("xxx.h5")

-   keras→h5(weights)

model.save\_weights("xxx.h5")

-   keras→json(model)

model.to\_json("xxx.json")

-   keras → onnx

安装 keras2onnx：

pip install keras2onnx

通过python转换：

import keras

import onnx

import keras2onnx

from keras.models import load\_model

km = load\_model(\"test.h5\")

om = keras2onnx.convert\_keras(km, km.name)

onnx.save\_model(om, \"./test.onnx\")

-   keras→tflite(model+weights):

python脚本tflite\_convert

tflite\_convert \--keras\_model\_file=xxx.h5 \--output\_file=xxx.tflite

-   h5→keras(weights)

model.load\_weights("xxx.h5")

-   h5→pb(model+weights)

[https://github.com/amir-abdi/keras\_to\_tensorflow]{.underline}

keras\_to\_tensorflow.py \--input\_model=xxx.h5 \--output\_model=xxx.pb

#### Darknet to ...

-   darknet→h5(model+weights)

[[https://github.com/qqwweee/keras-yolo3]{.underline}](https://github.com/qqwweee/keras-yolo3)

convert.py yolo.cfg yolo.weights yolo.h5

#### Onnx to...

-   onnx → keras

安装 onnx2keras：

pip install onnx2keras

通过python转换：

import keras

import onnx

import onnx2keras

om = onnx.load(\"test.onnx\")

km = onnx2keras.onnx\_to\_keras(om, \[\"input\_1\"\])

keras.models.save\_model(km,\'test.h5\',overwrite=True,include\_optimizer=True)

-   onnx → tensorflow

安装 onnx-tf：

pip install onnx-tf

通过python转换：

onnx-tf convert -i a.onnx -o a.pb

-   onnx → CoreML

参考以下github仓库：

[[https://github.com/onnx/onnx-coreml]{.underline}](https://github.com/onnx/onnx-coreml)

#### Pytorch to ...

-   pytorch → keras

安装 pytorch2keras：

pip install pytorch2keras

通过python转换：

import keras

### MMdnn转换

MMdnn是微软开源的DNN模型管理工具（MM=Model
Management），帮助用户在不同深度学习框架之间进行互操作，例如模型转换和可视化。可在Caffe，Keras，MXNet，Tensorflow，CNTK，PyTorch
Onnx和CoreML之间转换模型。

mmdnn的转换在实际实现中使用了不同的工具（包括不少上文提到的其他工具），因此实际效果会有重合。此外目前看来mmconvert的效果并不是非常好。

[[https://github.com/Microsoft/MMdnn]{.underline}](https://github.com/Microsoft/MMdnn)

可以通过pip install mmdnn安装，主要的转换程序是mmconvert，格式为：

mmconvert -sf SourceFramework -in InputNetwork -iw InputWeights -df
DestinationFramework -om OutputModel \--inputShape dim1, dim2, dim3

### ncnn转换

腾讯的手机端前向计算框架ncnn自带三种转换工具：**mxnet2ncnn，caffe2ncnn**和**onnx2ncnn**。可以将对应框架的模型转换为ncnn模型，然后通过NDK开发，在手机上使用这些模型。

传统库
------

### wave

wave是对wave音频文件进行操作的Python库。

-   wave

    -   open()：打开wave文件，返回Wave\_read或者Wave\_write对象

    -   Wave\_read：读方式打开的wave文件对象

        -   getparams()：返回nchannels（通道数，单声道、多声道），sampwidth（数据宽度，8bit或者16bit），framerate（44.1K、48K），nframes（总数据帧数），comptype（压缩方式），compname。

    -   Wave\_write：写方式打开的wave文件对象

### scipy

-   scipy

    -   signal：

        -   lfilter()：用FIR或者IIR过滤器滤波

            -   b：FIR或者IIR滤波器的系数（即firwin函数的返回值）

            -   a：IIR滤波器的系数

            -   x：输入的信号序列

        -   firwin()：根据参数生成FIR滤波器的系数（coefficients）

            -   numtaps：FIR滤波器系数的长度

            -   cutoff：截断频率

            -   fs：信号的采样频率

        -   remez()：用Remez算法生成滤波器

        -   freqz()：

    -   fft：参见numpy.fft

    -   fftpack

        -   helper.fftfreq()：返回离散的频率分布，等于numpy.fft.helper.fftfreq()

            -   n：返回的频率数组的长度

            -   d：采样间隔（单位为秒）

    -   io

        -   wavfile

            -   read()：读取wave文件，参数：路径，返回值：采样频率和数据

            -   write()：写入wave文件

### numpy

Python的张量运算包，所有的神经网络，以及大量的科学计算包都基于此。

-   ceil()：向上圆整

-   floor()：向下圆整

-   isnan()：判断张量中各个元素是否为None，返回同形状的bool类型

-   maximum()：参数为两个同形张量，对每个元素取最大值，得到一个新的同形张量

-   array()：用数组初始化一个ndarray

-   matlib.empty()：建立一个未初始化的数组，参数为张量形状，比如(2,3)

-   matlib.zeros()：建立全零张量，参数为张量的形状，比如(2,3)

-   matlib.ones()：建立全一张量，参数为张量的形状，比如(2,3)

-   matlib.identity()：建立一个正方形的单位矩阵，参数为矩阵边长

-   matlib.eye()：建立一个对角线为1其余元素为0的矩阵，参数为矩阵边长，与identity不同在于可为长宽可以不等

-   core.function\_base.linspace()：返回一个等差数列

    -   start：数列的最小值

    -   stop：数列的最大值

    -   num：数列元素的数量，缺省为50

-   core.numeric.zeros\_like()：建立形似的全零张量

    -   a：建立的张量形似a

    -   dtype： 元素数据类型

-   core.numeric.full()：建立一个每个元素值都为fill\_value的张量

    -   shape：张量的形状

    -   fill\_value：每个元素的值

-   lib.function\_base.delete()：删除元素

-   lib.function\_base.insert()：在ndarray中插入值

-   lib.function\_base.average()：计算张量的加权平均

-   lib.shape\_base.expand\_dims()：在原张量中插入一个新的维度

    -   a：输入的张量

    -   axis：新增维度位置，可以是整数（第几个维度），或tuple（哪几个维度）

-   lib.shape\_base.tile()：通过复制某些维度，来扩展张量的维度

    -   A：输入张量

    -   reps：各维度的重复次数，整数表示在最里一个维度重复，维度超出A的维度则表示扩展其维度

-   core.fromnumeric.reshape()：修改张量的形状（不改变其值）

-   core.fromnumeric.nonzero()：返回张量中所有的非零元素

    -   返回：一个tuple，每个成员都是numpy的向量，为各元素在各个维度的坐标，即如果输入张量为N维，则输出的tuple有N个成员

-   core.fromnumeric.cumsum()：累加函数

-   core.multiarray.where()：返回张量中符合条件的元素

    -   condition：条件

    -   返回：一个tuple，每个成员都是numpy的向量，为各元素在各个维度的坐标，即如果输入张量为N维，则输出的tuple有N个成员

-   core.multiarray.concatenate()：拼接若干张量

    -   (a1, a2,
        \...)：被拼接的张量，除了在拼接的维度外，这些张量必须形状相同

    -   axis：拼接的维度，缺省为0，即在最外的维度拼接

-   fft：快速傅立叶变换相关包

    -   fftpack.fft()：快速傅立叶变换，返回频域的序列，长度与输入相同（即每个元素代表频率/总长度的

        -   a：输入的时域数据序列，其长度等于频率×时间

    -   fftpack.ifft()：

    -   helper.fftfreq()：

-   random：

    -   random()：返回随机张量

        -   size：int或者tuple类型，随机张量的大小，缺省为1（返回单个随机数）

    -   normal()：返回符合正态分布的随机张量

        -   loc：位置参数，即正态分布的期望值

        -   scale：尺度参数，即正态分布的标准差

        -   size：int或者tuple类型，返回的随机张量大小

### pandas

pandas是Python中常用的数据分析库，包含了以**Series**和**DataFrame**为主的数据结构。

User Guide：

[[https://pandas.pydata.org/docs/user\_guide/index.html]{.underline}](https://pandas.pydata.org/docs/user_guide/index.html)

API Reference：

[[https://pandas.pydata.org/docs/reference/index.html]{.underline}](https://pandas.pydata.org/docs/reference/index.html)

**参考**

[[https://www.jianshu.com/p/8024ceef4fe2]{.underline}](https://www.jianshu.com/p/8024ceef4fe2)

**参考**

[[https://blog.csdn.net/weixin\_38168620?t=1]{.underline}](https://blog.csdn.net/weixin_38168620?t=1)

以下为各种pandas数据类型

#### Series

Series是Pandas最基本的数据格式，类似一维数组。Series的数据值（values）带有一一对应的索引（index），values和index都有name。

因为有index和name，因此一个长度为L的Series中其实包含了2L+2个数据：L个**values**，L个**index**，以及index和values的**name**。

-   core.indexes.series.Series：

    -   Series可以创建自己的**索引（index）**，这点上更象是dict，而与list和ndarray不同

    -   既可以通过**index**访问，也可以通过**位置下标**访问，这点与dict不同

    -   Series和ndarray一维数组一样，其数据只能是**相同数据类型**，而list中的数据可以是不同的类型

    -   Series的**index**和**值（values）**都有**名称（name）**，而dict、ndarray和list都没有

    -   支持numpy的张量运算

        **参考**

        [[https://blog.csdn.net/weixin\_38168620/article/details/79572544]{.underline}](https://blog.csdn.net/weixin_38168620/article/details/79572544)

#### DataFrame

是Pandas中最常用的基本数据结构，表示二维表结构。DataFrame除了数据值本身（values）之外，还包括了行索引（index）和列索引（columns），以及这两者的name。但values本身没有name。

因此一个shape为(y,x)的DataFrames，实际包含了xy+x+y+2个数据：x\*y个**values**，y个**index**，x个**columns**，以及index和column的**name**。

-   DataFrame的特点包括（以下df为其实例）：

    -   带有行索引（index）和列索引（columns）

    -   index和columns都可以有name

    -   可变大小，且表中数据的类型可以不一致

    -   df.values返回一个二维的ndarray

    -   df.XXX返回一个指定的列

        -   返回值为Series格式

        -   XXX为某columns，也是返回的Series的values的name

        -   返回的Series的index的name同df.index.name

    -   df.loc(XXX)返回一个指定的行

        -   返回值为Series格式

        -   XXX为某index，也是返回的Series的values的name

        -   返回的Series的index的name同df.columns.name

-   core.frame.DataFrame的成员变量及成员函数包括：

    -   index：行的标签

    -   columns：列的标签

    -   axes：行和列的标签

    -   dtypes：各列数据的类型

    -   head()：返回数据的前若干行（缺省为5）

    -   tail()：返回数据的最后若干行（缺省为5）

    -   to\_numpy()：转换成numpy格式的二维张量，行和列的标签会丢失

    -   describe()：返回各列数据的各项统计值（均值，标准差，最大，最小等）

    -   loc：通过行-列标签定位数据，如data.loc\["number",\["name","sex"\]\]

    -   iloc：通过索引定位数据，如data.iloc\[3,2:4\]

    -   shape：数据的形状，因为是二维数据，形为(4,5)

        **参考**

        [[https://blog.csdn.net/weixin\_38168620/article/details/79572785]{.underline}](https://blog.csdn.net/weixin_38168620/article/details/79572785)

#### Index

Index类描述了Series和DataFrame的索引，可以被视为一个一维数组。

-   core.indexes.base.Index：

    -   series.index返回值为Index类型

    -   dataframe.index和dataframe.columns返回值为Index类型

    -   Index对象创建后不可修改

#### MultiIndex

MultiIndex描述了一个多级标签。

**参考**

[[https://blog.csdn.net/weixin\_38168620/article/details/79580272]{.underline}](https://blog.csdn.net/weixin_38168620/article/details/79580272)

#### Timestamp/Period/Timedelta

Timestamp是继承自标准库datetime的类，表示了一个精确到秒的**时间戳**。

Period表示一个标准的**时间段**。例如某年、某月、某日、某小时等。时间的长短由freq决定。

Timedelta表示一个**时间间隔**。

**参考**

[[https://blog.csdn.net/weixin\_38168620/article/details/79596526]{.underline}](https://blog.csdn.net/weixin_38168620/article/details/79596526)

#### DatatimeIndex/PeriodIndex/TimedeltaIndex

Timestamp、Period和Timedelta对象都是单个值，这些值都可以放在索引或数据中。作为索引的时间序列有：DatetimeIndex、PeriodIndex和TimedeltaIndex，它们都可以作为Series和DataFrame的索引。

**参考**

[[https://blog.csdn.net/weixin\_38168620/article/details/79596564]{.underline}](https://blog.csdn.net/weixin_38168620/article/details/79596564)

### matplotlib

**matplotlib** 是 Python 的绘图库。它可与 NumPy
一起使用，提供了一种有效的 MatLab
开源替代方案。matplotlib的语法类似matlab面向过程.

matplotlib.pyplot是最常用的模块，通常通过如下语句引用：

import matplotlib.pyplot as plt

官网：

[[https://matplotlib.org/]{.underline}](https://matplotlib.org/)

常用的pyplot模块函数包括：

-   plt.subplots()：创建一个图表及其子图表

    -   返回figure.Figure：创建的图表

    -   返回axes.Axes或者np.ndarray：子图（只有一个）或者子图的阵列（若干个）

<!-- -->

-   plt.subplot()：将子图表激活，参数为2,3,5,或者235，表示2行3列，被激活的子图为第5个（第2行第2列），之后所有操作均是对该子图表操作

-   plt.title()：设置图表的名称

-   plt.xlabel()：设置x轴的名称

-   plt.ylabel()：设置y轴的名称

-   plt.axis()：控制坐标轴的显示

-   plt.grid()：网格显示的控制

-   plt.legend()：图例（说明）的显示

-   plt.bar()：绘制柱状图

-   plt.plot()：绘制图表

-   plt.imshow()：在图表中显示图片

-   plt.axvline()：画一条竖线

-   plt.show()：显示，在以上各种操作完成后调用

其数据结构有：

-   figure.Figure：表示图表的类

-   axes.\_axes.Axes：

NN框架
======

本章接续上一章，按照来源（公司/组织）+具体框架的组织方式，介绍了各个神经网络的框架/库。

Google
------

Google提供的各种深度学习框架

### Tensorflow（基础框架）

**TensorFlow**是Google开源的神经网络框架，最流行的两个神经网络通用框架之一（另一个是FB的Pytorch），相对于Pytorch，TF的原语更偏底层计算图。

由于低阶API偏底层，Tensorflow的框架易读性不好，而且各版本之间兼容性做得也不行，以下的介绍并非在所有版本都适用。

中文教程：

[[http://c.biancheng.net/tensorflow/]{.underline}](http://c.biancheng.net/tensorflow/)

Tensorflow模型部署：

[[https://blog.csdn.net/chongtong/article/details/90379347]{.underline}](https://blog.csdn.net/chongtong/article/details/90379347)

**低阶API：基本概念、训练、导入导出**

相较于一般框架中的"层"、"Normalization"等操作，Tensorflow的低阶原语非常底层，类似于汇编与高级语言的区别。

-   Tensor：张量常量，通常表示数据流图中的**边**/**Edge**（Operation的输入输出）

    -   graph：该常量所属的数据流图

-   constant()：生成张量常量

-   Variable：张量变量，keras中用变量来表示**权重**

    -   graph：该变量所属的数据流图

-   Operation：对张量的计算/操作，即数据流图中的**节点**（也叫**node**或**op**）

-   Graph：数据流图，也叫计算图，包括了Operation（计算）和Tensor（数据），但不包括Variable（权重），缺省的Graph通过tf.get\_default\_graph()获得

    -   get\_operations()：返回数据流图中Operation的列表

    -   get\_tensor\_by\_name()：根据名字返回Tensor

    -   as\_graph\_def()：序列化，将Graph转换成GraphDef并返回

    -   as\_default()：返回一个上下文管理器，当前Graph是其缺省Graph

-   get\_default\_graph()：获取当前线程的缺省Graph

-   GraphDef：对应Graph的序列化数据结构（pb），用于将Graph从Python中传递到C底层，以便进行计算。GraphDef包含了描述节点（计算）的NodeDef（对应于Graph中的Operation），以及张量常量Tensor，但不包括张量变量Variable（权重）。

    -   ParseFromString()：从字符串（通常读取自文件）中读取序列化信息进GraphDef（通常原来为空）

-   import\_graph\_def()：从文件中导入GraphDef，返回Graph，对应的导出函数为tf.train.write\_graph()

-   NodeDef：对应Operation的序列化数据格式（pb）

-   MetaGraphDef：也是对应Graph的序列化数据结构，包含了一个数据流图和其相关的元数据，包含信息比GraphDef多，具体包括：

1.  MetaInfoDef：Graph的元数据

2.  GraphDef：包括Operation和Tensor

3.  SaverDef：包括Variable

4.  CollectionDef：

-   Session：会话，用来表示客户端程序（python）和C++运行时之间的连接，Session还使得用户可以操作本地和远程的多个设备（分布式tf），Session还可以缓存Graph的信息

    -   \_\_init\_\_()：接受三个参数（均有缺省值）：

        -   target：设备，缺省情况下是本地设备，也可以通过grpc协议指定服务器

        -   graph：绑定的graph，缺省情况下绑定当前的默认graph

        -   config：配置选项

    -   graph：这个Session中运行的Graph

    -   run()：运行，获得Graph元素（或其组合）的值

        -   fetches：需要运行的Graph元素（Operation、Tensor或者对应的名称字符串）或者是其组合（tuple、队列、字典）

        -   feed\_dict：张量，用于替换计算图中placeholder，缺省为None

        -   返回：fetches的值，形状同fetches

-   InteractiveSession：交互式Session

-   global\_variables\_initializer()：全局变量初始化

-   trainable\_variables()：返回所有可训练的变量

-   train：

    -   Saver：（在TF2.0之前）用于保存Variable（权重）

        -   restore()：从指定位置的ckpt文件中载入权重

        -   save()：保存权重到指定位置，生成\*.ckpt文件

    -   SaverDef：对应Saver的序列化数据结构（pb）

    -   Checkpoint：（在TF2.0之后）保存检查点，包括Variable，和Saver的区别在于是以对象而非名称来保存

    -   write\_graph()：以GraphDef导出Graph到文件

    -   import\_meta\_graph()：从文件中导入MetaGraphDef为返回的Graph

    -   export\_meta\_graph()：导出Graph的MetaGraphDef信息到文件

    -   latest\_checkpoint()：

-   summary：

    -   FileWriter：

        -   add\_graph()：

        -   flush()：

            **参考**

            [[https://blog.csdn.net/weixin\_39721347/article/details/86171990]{.underline}](https://blog.csdn.net/weixin_39721347/article/details/86171990)

[[https://blog.csdn.net/qq\_35799003/article/details/84948527]{.underline}](https://blog.csdn.net/qq_35799003/article/details/84948527)

[[https://www.aiuai.cn/aifarm701.html]{.underline}](https://www.aiuai.cn/aifarm701.html)

**张量相关**

-   contant()：创建常数张量，contant(\[2,3,4\]）表示3维向量

-   zeros()：创建全0张量，比如zeros(\[2,3\],tf.int32)，一个全0的2×3矩阵

-   zeros\_like(t)：创建一个形状似参数t的全0张量

-   ones()：创建全1张量，类似zeros()

-   ones\_like(t)：创建一个形状似参数t的全1张量

-   linspace(start,stop,num)：创建等差数列，从start开始，stop结束，个数为num

-   range(start,limit,delta)：创建等差数列

-   random\_normal(shape,mean,stddev,seed)：创建正态分布的张量，型为shape，均值为mean（缺省为0），标准差为stddev（缺省为1），随机数种子为seed

-   eye()：创建单位矩阵

-   add()：张量的加法，也可以用"+"代替（运算符重载）

-   placeholder()：占位符

-   device()：选择设备，比如device('/cpu:0'）

#### 模型文件格式

**.pb**：protocol buffer格式，tensorflow用来存储（训练好的）模型和权重

**.meta**：保存计算结构图（即网络模型）的文件

**.ckpt**：checkpoint文件，在v0.11之前用于保存权重，之后被.index和.data文件替代

**.index**： it is a string-string immutable
table(tensorflow::table::Table). Each key is a name of a tensor and its
value is a serialized BundleEntryProto. Each BundleEntryProto describes
the metadata of a tensor: which of the \"data\" files contains the
content of a tensor, the offset into that file, checksum, some auxiliary
data, etc.

**.data-00000-of-00001**：在v0.11之后，用于保存参数名和权重的文件

对于tensorflow的模型来说，可以使用:

1.meta+ckpt（v0.11之前）

2.meta+index+data（v0.11之后）

3.一个pb文件

这三种方式来描述一个带有权重的模型。这其中1和2通常用于描述一个训练中的模型（即checkpoint），因为多个文件更灵活。而单个的pb文件则比较直观，用于描述训练好的模型。

#### 保存训练模型

saver = tf.train.Saver()

with tf.Session() as sess:

sess.run(tf.global\_variables\_initializer())

\_, loss\_value = sess.run(\[train\_step, losses\], feed\_dict={image:
x\_train, y: y\_train}) \#训练模型

saver.save(sess, "./save\_tf/tf\_model")

#### 训练模型\--\>推理模型

以下代码描述了如何将读取训练模型文件，并导出pb格式的推理模型

import tensorflow as tf

from tensorflow.python.framework import graph\_io

from tensorflow.python.framework.graph\_util import
convert\_variables\_to\_constants

\"\"\"\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--导入tensorflow模型\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--\"\"\"

saver = tf.train.import\_meta\_graph(\'./saved\_tf/tf\_model.meta\')

graph = tf.get\_default\_graph()

sess = tf.Session()

sess.run(tf.global\_variables\_initializer())

saver.restore(sess, tf.train.latest\_checkpoint(\'./saved\_tf\'))

\"\"\"\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--保存为.pb格式\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--\"\"\"

graph = sess.graph

with graph.as\_default():

output\_names = \[\"output/Softmax\"\]

input\_graph\_def = graph.as\_graph\_def()

frozen\_graph = convert\_variables\_to\_constants(sess,
input\_graph\_def, output\_names)

graph\_io.write\_graph(frozen\_graph, \'./saved\_pb\',
\'tensorflow.pb\', as\_text=False)

#### 训练模型文件恢复及推理

从meta文件中读取训练中的模型，并进行推理

\# 读取模型

saver = tf.train.import\_meta\_graph(\'./saved\_tf/tf\_model.meta\')

graph = tf.get\_default\_graph()

images\_placeholder = graph.get\_tensor\_by\_name(\"input:0\")

output = graph.get\_tensor\_by\_name(\"output/Softmax:0\")

with tf.Session() as sess:

sess.run(tf.global\_variables\_initializer())

saver.restore(sess, tf.train.latest\_checkpoint(\'./saved\_tf\'))

feed\_dict = {images\_placeholder: x\_test}

\# 推理模型

logits=sess.run(output, feed\_dict=feed\_dict)

print(logits.astype(np.int32))

#### pb文件读取及推理

从pb文件中读取训练好的模型，并进行推理

\# 读取模型

output\_graph\_def = tf.GraphDef()

\# 打开.pb模型

with open(\'./saved\_pb/tensorflow.pb\', \"rb\") as f:

output\_graph\_def.ParseFromString(f.read())

tensors = tf.import\_graph\_def(output\_graph\_def, name=\"\")

graph = tf.get\_default\_graph()

input\_x = graph.get\_tensor\_by\_name(\"input:0\")

out\_softmax = graph.get\_tensor\_by\_name(\"output/Softmax:0\")

with tf.Session() as sess:

sess.run(tf.global\_variables\_initializer())

\# 推理模型

logits = sess.run(out\_softmax, feed\_dict={input\_x: x\_test})

print(logits.astype(np.int32))

#### pb文件读取及分析

从pb文件中读取训练好的模型，并进行分析

\# 加载模型

output\_graph\_def = tf.GraphDef()

with open(\'../model/saved\_pb/tensorflow.pb\', \"rb\") as f:

output\_graph\_def.ParseFromString(f.read())

tensors = tf.import\_graph\_def(output\_graph\_def, name=\"\")

sess = tf.Session()

sess.run(tf.global\_variables\_initializer())

graph = tf.get\_default\_graph()

\# 生成tensorboard文件

file\_writer = tf.summary.FileWriter(\'./pb\')

file\_writer.add\_graph(graph)

file\_writer.flush()

\# 打印模型中所有的操作

op = graph.get\_operations()

for i, m in enumerate(op):

print(\'op{}:\'.format(i), m.values())

#### python模型部署

class python\_model():

def \_\_init\_\_(self,model\_path):

\# 读取模型

output\_graph\_def = tf.GraphDef()

\# 打开.pb模型

with open(model\_path, \"rb\") as f:

output\_graph\_def.ParseFromString(f.read())

tensors = tf.import\_graph\_def(output\_graph\_def, name=\"\")

self.\_\_sess = tf.Session()

self.\_\_sess.run(tf.global\_variables\_initializer())

graph = tf.get\_default\_graph()

self.\_\_input = graph.get\_tensor\_by\_name(\"input:0\")

self.\_\_output = graph.get\_tensor\_by\_name(\"output/Softmax:0\")

def inference(self,input):

output = self.\_\_sess.run(self.\_\_output, feed\_dict={self.\_\_input:
input})

return output

model = python\_model(model\_path=\'../model/saved\_pb/tensorflow.pb\')

\# 读取数据

f = np.load(\'../model/data/mnist.npz\')

x\_test, y\_test = f\[\'x\_test\'\], f\[\'y\_test\'\]

x\_test = np.reshape(x\_test, \[-1, 784\])

output=model.inference(x\_test)

print(output.astype(np.int32))

### Keras（高级接口）

**Keras**是Google开源的神经网络前端接口。基于Tensorflow，Keras的出现是为了让用户更快速的开发并试用模型。其特点包括：

-   相比Tensorflow，专注于用户友好

-   后端的多种实现，目前已经支持**TensorFlow、CNTK、Theano**

-   包括了一个基本的图像处理库，模型在keras.applications中。

keras包现已合并到tensorflow中，为tensorflow.keras

-   **keras.layers**：包含了各种层类型

    -   **core**：各种核心层

        -   Dense：全连接层

        -   Activation：激活函数层，可单独加入model，也可作为参数加入其他层

        -   Dropout：在进行学习的时候使得某些神经元失效，以减少过拟合

        -   Flatten：展平输入为向量（不影响batch），reshape的特例

        -   Input：输入层，初始化Keras张量

        -   Reshape：对张量进行尺寸/维度变换，比如将（None,2,3,4）变为（None,8,3），None为batch

        -   Permute：维度置换（比如将2维卷积的第1维和第2维对调），例如将\[\[1,2,3\],\[4,5,6\]\]变为\[\[1,4\],\[2,5\],\[3,6\]\]

        -   RepeatVector：将输入的向量（2维张量，包括batch）重复N次，从（None,
            32）变为（None, N, 32）

        -   Lambda：将指定函数包装为层

        -   ActivityRegularization：

        -   Masking：

        -   SpatialDropout1/2/3D：类似Dropout，但会丢弃整个（1维/2维/3维）特征图，而非单个神经元。比如SpatialDropout2D，输入为（None,
            X, Y, C），dropout时会将某个通道的特征图（X,Y）全部丢弃。

    -   **convolutional**：各种卷积相关层

        -   Conv1D/Conv2D/Conv3D：一维二维三维的常规卷积层

        -   SeparableConv1/2D：深度分离卷积，包括一个逐层卷积层（Depthwise）和一个逐点卷积层（Pointwise）

        -   DepthwiseConv2D：单独的逐层卷积层

        -   Conv2/3DTranpose：反卷积

        -   Cropping1/2/3D：1维2维3维的裁剪

        -   UpSampling1/2/3D：1/2/3维的上采样，对每个输入元素重复若干遍

        -   ZeroPadding1/2/3D：值为0的Padding

    -   **pooling**：各种池化层

        -   MaxPooling1/2/3D ：Max池化层

        -   AveragePooling1/2/3D：Average池化层

        -   GlobalMaxPooling1/2/3D：全局Max池化层

        -   GlobalAveragePooling1/2/3D：全局Average池化层

    -   **local**：本地连接层，即不共享权重的卷积层

        -   LocallyConnected1/2D：1维/2维的本地卷积层

    -   **recurrent**：循环层

        -   RNN：

        -   SimpleRNN：

        -   SimpleRNNCell：

        -   GRU：

        -   GRUCell：

        -   LSTM：

        -   LSTMCell：

        -   ConvLSTM2D：

        -   ConvLSTM2DCell：

        -   StackedRNNCell：

    -   **embeddings**：嵌入层，只有一个Embedding

        -   Embedding：

    -   **merge**：融合层，融合多个输入

        -   Add：将输入的张量逐元素相加（所有输入张量必须有相同尺寸）

        -   Subtract：将输入的两个张量相减

        -   Multiply：将输入的张量逐元素相乘（所有输入张量必须有相同尺寸）

        -   Average：将输入的张量逐元素取均值（所有输入张量必须有相同尺寸）

        -   Maximum：将输入的张量逐元素取最大值（输入张量必须有相同尺寸）

        -   Minimum：将输入的张量逐元素取最小值（所有输入张量必须有相同尺寸）

        -   Concatenate：连接输入的张量（输入张量除了连接轴之外，尺寸相同）

        -   Dot：计算输入张量的点积

    -   **advanced\_activations**：高级激活层，包括

        -   LeakyReLU：

        -   PReLU：

        -   ELU：

        -   ThresholdedReLU：

        -   Softmax：

        -   ReLU：

    -   **normalization**：正规化层，只有一个BN

        -   BatchNormalization：

    -   **noise**：噪声层，用于缓解过拟合，包括

        -   GaussianDropout：

        -   GaussianNoise：

        -   AlphaDropout：

    -   **wrappers**：层封装器

        -   Bidirectional：双向封装，比如双向RNN，双向LSTM等

        -   TimeDistributed：

    -   **convolutional\_recurrent**：

        -   ConvLSTM2D：

        -   ConvLSTM2DCell：

        -   ConvRNN2D：

    -   **cudnn\_recurrent**：

        -   CuDNNGRU：

        -   CuDNNLSTM：

-   **keras.backend**：各种后端实现的支持（Theano，CNTK，TensorFlow），common为三家共同支持的接口

-   **keras.preprocessing**：预处理，包括：

    -   **sequence**：时序数据的预处理工具包，包括TimeseriesGenerator

    -   **text**：文本的预处理工具包

    -   **image**：图像的预处理工具包

        -   DirectoryIterator：遍历指定目录的迭代器

        -   ImageDataGenerator：图像数据产生器（用于数据增广）

            -   apply\_transform()：

            -   fit()：

            -   flow()：

            -   flow\_from\_dataframe()：

            -   flow\_from\_directory()：从指定目录产生增广数据，指定目录中每个子目录被视为一个类别

            -   get\_random\_transform()：

            -   random\_transform()：

            -   standardize()：

        -   Iterator：

        -   NumpyArrayIterator：

        -   array\_to\_img()：将3D numpy array转为PIL图像

        -   img\_to\_array()：将PIL图像转为3D numpy array

        -   save\_img()：以ndarray形式保存图像到文件

-   **keras.losses**：内置的损失函数，包括MSE，MAE，KLD，MAPE，MSLE等

-   **keras.metrics**：内置的评价函数，评价函数不用于训练，用于评估当前模型的性能。

-   **keras.optimizers**：内置的优化器，包括SGD，RMSprop，Adagrad，Adadelta，Adam，Adamax，Nadam

-   **keras.activations**：内置的激活函数，包括softmax，relu，tanh，sigmoid等。

    -   softmax()：

    -   elu()：

    -   selu()：

    -   softplus()：

    -   softsign()：

    -   relu()：

    -   tanh()：

    -   sigmoid()：

    -   exponential()：

    -   linear()：

-   **keras.callbacks**：内置的回调函数，回调函数在训练期间的特定时间点被调用。

    -   BaseLogger()：

    -   TerminateOnNaN()：

    -   ProgbarLogger()：

    -   History()：

    -   ModelCheckpoint()：在每个epoch之后保存模型

    -   EarlyStopping()：当被监测的数据不再提升，则停止训练

    -   RemoteMonitor()：

    -   LearningRateScheduler()：

    -   TensorBoard()：写Tensor board日志到指定目录

    -   ReduceLROnPlateau()：在学习进入平台期之后降低学习率

    -   CSVLogger()：

    -   LambdaCallback()：

-   **keras.datasets**：内置的数据集

    -   cifar:CIFAR格式的文件的解析工具

        -   load\_batch():解析CIFAR格式的数据文件，返回格式为(data,
            labels)

    -   cifar10：CIFAR10数据集

        -   load\_data():返回格式为(x\_train,y\_train),
            (x\_test,y\_test)

    -   cifar100：CIFAR100数据集

        -   load\_data():返回格式为(x\_train,y\_train),
            (x\_test,y\_test)

    -   imdb：IMDB电影评论情感分类数据集

        -   get\_words\_index():

        -   load\_data():返回格式为(x\_train,y\_train),
            (x\_test,y\_test)

    -   reuters：路透社新闻主题分类

        -   get\_words\_index():

        -   load\_data():返回格式为(x\_train,y\_train),
            (x\_test,y\_test)

    -   mnist：MNIST数据集

        -   load\_data():返回格式为(x\_train,y\_train),
            (x\_test,y\_test)

    -   fashion\_mnist：时尚物品数据集

        -   load\_data():返回格式为(x\_train,y\_train),
            (x\_test,y\_test)

    -   boston\_housing：波士顿房价回归数据集

        -   load\_data():返回格式为(x\_train,y\_train),
            (x\_test,y\_test)

-   **keras.applications**：带有预训练权值的网络模型，可用来预测、特征提取和微调

    -   densenet**：**

        -   DenseNet121()：返回一个DenseNet121模型，可带预训练权重

        -   DenseNet169()：返回一个DenseNet169模型，可带预训练权重

        -   DenseNet201()：返回一个DenseNet201模型，可带预训练权重

    -   inception\_resnet\_v2**：**

        -   InceptionResNetV2()：返回一个InceptionResNetV2模型，可带预训练权重

    -   inception\_v3**：**

        -   InceptionV3()：返回一个InceptionV3模型，可带预训练权重

    -   mobilenet**：**

        -   MobileNet()：返回一个MobileNet模型，可带预训练权重

    -   mobilenet\_v2**：**

        -   MobileNetV2()：返回一个MobileNetV2的模型，可带预训练权重

    -   nasnet:

        -   NASNetLarge()：返回一个NASNetLarge的模型，可带预训练权重

        -   NASNetMobile()：返回一个NASNetMobile的模型，可带预训练权重

    -   resnet50:

        -   ResNet50()：返回一个ResNet50模型，可带预训练权重

    -   vgg16**：**

        -   VGG16()：返回一个VGG16模型，可带预训练权重

    -   vgg19**：**

        -   VGG19()：返回一个VGG19模型，可带预训练权重

    -   xception**：**

        -   Xception()：返回一个Xception模型，可带预训练权重

-   **keras.initializers**：内置的初始化器

-   **keras.regularizers**：内置的正则化器

-   **keras.constraints**：内置的约束项，包括MaxNorm，MinMaxNorm，NonNeg，UnitNorm

-   **keras.models**：模型相关的

    -   Model：

        -   compile()：编译模型

        -   fit()：训练模型

        -   evaluate()：评估模型的准确性

        -   predict()：使用模型预测

        -   train\_on\_batch()：

        -   test\_on\_batch()：

        -   predict\_on\_batch()：

        -   fit\_generator()：基于数据增广器训练模型

        -   evaluate\_generator()：基于数据增广器进行评估

        -   predict\_generator()：

        -   get\_layer()：

    -   Sequential：

Facebook
--------

Facebook提供的各种深度学习框架，主要是以

### Pytorch（基础框架）

官方API文档：

[[https://pytorch.org/docs/stable/torch.html]{.underline}](https://pytorch.org/docs/stable/torch.html)

中文版教程：

[https://www.pytorch123.com/]{.underline}

#### torch

torch包本身包括了一些基本的张量操作，功能和numpy类似，可以支持GPU。

-   torch

-   torch.is\_tensor(): 如果参数是一个pytorch张量，返回True

-   torch.empty()：生成一个空矩阵

    -   rand()：生成一个随机矩阵

    -   zeros()：生成一个全零矩阵

    -   tensor()：生成一个矩阵，元素由（list格式的）参数给定

    -   is\_storage(): 如果参数是一个pytorch storage，返回True

    -   numel(): 返回张量中的元素个数

    -   eye()：创建一个单位矩阵

    -   from\_numpy(): 将numpy的ndarry转换为pytorch张量

    -   linspace():
        返回一个1维张量，包含在区间start 和 end 上均匀间隔的steps个点。
        输出1维张量的长度为steps。

    -   ones()：返回指定形状的全1张量

    -   serialization.save(): 保存一个对象到文件中

    -   serialization.load(): 从文件中读取一个save()保存的对象

    -   tensor.Tensor: 表示pytorch张量的类

    -   jit:
        TorchScript格式相关，在非Python环境下运行（这个格式不能处理条件控制，比如if。也不能处理第三方的函数，只能处理pytorch中的部分操作）

        -   ScriptModule：TorchScript格式的模块，类似torch.nn.Module

        -   ScriptFunction：TorchScript格式的函数

        -   trace()：将一个函数转化为ScriptFunction格式并返回。

        -   trace\_module()：将torch.nn.Module转化为ScriptModule格式并返回。

    -   quantization：

        -   quantize.convert()：将一个浮点模型量化

#### torch.nn

nn是torch最大的子pacakge，其中最主要是modules子package，里面包含了各种网络模型相关的类，包括模型、各种卷积层、池化层、RNN层、激活函数、dropout、损失函数。

##### nn.Module

nn.Module是所有神经网络模块的基类，自己定义的网络模块也必须是Module的子类。网络模块中可以包含其他网络模块，从而由小（层、微结构）形成一个完整的神经网络。

所有Module的子类都要实现**forward()**函数，作为前向推理的工作函数。forward()函数会在Module的\_\_call\_\_()函数中被间接调用（通过\_call\_impl和\_slow\_forward），之所以没有直接实现为\_\_call\_\_()，是因为需要在执行前后进行一些其他的操作。

因此，如果Module A中包含Module B，一般来说他们的调用顺序是这样的：

A.\_\_call\_\_() \--\> A.forward() \--\> B.\_\_call\_\_() \--\>
B.forward()

举例如下：

import torch.nn as nn

import torch.nn.functional as F

class Model(nn.Module):

def \_\_init\_\_(self):

> super(Model, self).\_\_init\_\_()
>
> self.conv1 = nn.Conv2d(1, 20, 5)
>
> self.conv2 = nn.Conv2d(20, 20, 5)

def forward(self, x):

> x = F.relu(self.conv1(x))
>
> return F.relu(self.conv2(x))

-   nn.modules.module.Module: 所有神经网络模块的基类

    -   **forword**：需要子类实现的函数，所有子类必须实现该函数

    -   add\_module()： 添加子模块到当前模型

    -   children(): 返回当前模型子模块的迭代器

    -   cpu(): 将所有参数和buffers复制到CPU

    -   cuda(): 将所有参数和buffers复制到GPU

    -   eval(): 模型转换为evaluation（预测）模式（影响Dropout层和BN层）

    -   double(): 将参数和buffers的类型转为double

    -   float(): 将参数和buffers的类型转为float

    -   half(): 将参数和buffers的类型转为half

    -   modules(): 返回当前模型所有模块（各子孙模块）的迭代器

    -   named\_children(): 同children()，但同时yield子模块名字

    -   named\_modules(): 同modules()，但同时yield子孙模块名字

    -   parameters(): 返回包含所有参数的迭代器

    -   register\_backward\_hook()

    -   register\_buffer()

    -   register\_forward\_hook()

    -   register\_parameter()

    -   state\_dict(): 返回一个字典，包含模型的所有状态

    -   load\_state\_dict():

    -   train(): 模型设置为training模式（影响Dropout层和BN层）

    -   zero\_grad(): 将模型所有参数梯度设置为0

-   nn.modules.container.Sequential: 顺序模型，Module的子类

-   nn.modules.linear.Linear: 全连接层，Module的子类

    -   in\_features：输入特征数量

    -   out\_features：输出特征数量

    -   weight：权重，Tensor类

    -   \_\_init\_\_()：构造函数

        -   in\_features：输入特征数量

        -   out\_featues：输出特征数量

        -   bias：是否有偏置，bool类型

-   nn.modules.conv.Conv(1\|2\|3)d: 描述卷积层的模块

-   nn.modules.conv.ConvTranspose(1\|2\|3)d:

-   nn.modules.pooling.AdaptiveAvgPool(1\|2\|3)d:
    自适应（输出尺寸固定）平均池化

-   nn.modules.pooling.AdaptiveMaxPool(1\|2\|3)d:
    自适应（输出尺寸固定）最大池化

-   nn.modules.pooling.AvgPool(1\|2\|3)d: 平均池化

-   nn.modules.pooling.FractionalMaxPool(2\|3)d:

-   nn.modules.pooling.LPPool(1\|2)d:

-   nn.modules.pooling.MaxPool(1\|2\|3)d: 最大池化

-   nn.modules.pooling.MaxUnpool(1\|2\|3)d:
    逆最大池化（丢失的部分设置为0）

-   nn.modules.activation.ReLU,RReLU,ReLU6,ELU,CELU,SELU,GLU,LeakyReLU,PReLU

-   nn.modules.activation.Hardtanh,Tanh

-   nn.modules.batchnorm.BatchNorm(1\|2\|3)d:

-   nn.modules.batchnorm.SyncBatchNorm:

-   nn.modules.rnn：各种RNN单元（微结构），包括RNNBase

-   nn.modules.rnn.RNN:

-   nn.modules.rnn.LSTM:

-   nn.modules.rnn.GRU:

-   nn.modules.rnn.RNNCellBase:

-   nn.modules.rnn.RNNCell:

-   nn.modules.rnn.LSTMCell:

-   nn.modules.rnn.GRUCell:

-   nn.modules.dropout.Dropout(\|2d\|3d),AlphaDropout,FeatureAlphaDropout:

-   nn.modules.sparse.Embedding.EmbeddingBag:

-   nn.modules.loss.MSELoss:

-   nn.parameter.Parameters:

#### torch.autograd

#### torch.optim

Pytorch中的各种优化方法

-   torch.optim：

    -   adadelta.Adadelta：

    -   adagrad.Adagrad：

    -   adam.Adam：

    -   adamw.AdamW：

    -   sparse\_adam.SparseAdam：

    -   adamax.Adamax：

    -   asgd.ASGD：

    -   sgd.SGD：

    -   rprop.Rprop：

    -   rmsprop.RMSprop：

    -   optimizer.Optimizer：

    -   lbfgs.LBFGS：

#### torch.utils

相关工具

-   torch.utils.data：

    -   dataset.Dataset：数据集的抽象基类，所有具体的数据集都是其子类，子类需实现**\_\_getitem\_\_**()函数

    <!-- -->

    -   dataset.IterableDataset：

    -   dataset.TensorDataset：

    -   dataset.ConcatDataset：

    -   dataset.ChainDataset：

    -   dataset.Subset：

    -   dataloader.DataLoader：

    -   sampler.Sampler：

    -   sampler.SequentialSampler：

    -   sampler.RandomSampler：

    -   sampler.SubsetRandomSampler：

    -   sampler.WeightedRandomSampler：

    -   sampler.BatchSampler：

    -   distributed.DistributedSampler：

-   torch.utils.data：

### torchvision（图像处理）

Pytorch的**机器视觉**相关的框架，包括**图像分类**、**语义分割**、及少量**目标检测**、**实例分割**、**人体关键点检测**、**行为识别**（视频）。

官方文档：

[[https://pytorch.org/docs/stable/torchvision/]{.underline}](https://pytorch.org/docs/stable/torchvision/)

-   torchvision：包含了目前流行的数据集、网络模型和图片处理工具

    -   datasets: 数据集

        -   cifar

            -   CIFAR10: 表示CIFAR10数据集的类

            -   CIFAR100: 表示CIFAR100数据集的类

        -   mnist.MNIST:

        -   coco.CocoCaptions:

        -   coco.CocoDetections:

        -   \...

    -   models: 模型卷积网络

        -   alexnet:

            -   AlexNet：表示Alexnet的类

            -   alexnet():
                返回Alexnet对象，pretrained参数决定是否加载权重

        -   densenet：densenet模块

            -   DenseNet：Densenet的类

            -   densenet121()：返回densenet121结构的DenseNet对象

            -   densenet161()

            -   densenet169()

            -   densenet201()

        -   inception：inception模块

            -   Inception3：Inception V3的类

            -   inception\_v3()：返回Inception3的对象

            -   \...

        -   resnet:

            -   BasicBlock: residual block的类

            -   Bottleneck: bottleneck结构的类

            -   ResNet: resnet的类

            -   resnet50()：返回resnet50结构的ResNet对象

            -   resnet101()：返回resnet101结构的ResNet对象

            -   resnet152()：返回resnet152结构的ResNet对象

        -   mobilenet：mobilenet模块

            -   MobileNetV2：mobilenet v2的类

            -   mobilenet\_v2()：返回一个MobileNetV2的对象

            -   ConvBNReLU：卷积+BN+ReLU组成的模块

            -   InvertedResidual：Inverted Residual Block

    -   transforms: 对PIL.Image进行变换的模块

        -   transforms.Compose:

        -   transforms.CenterCrop: 对图片进行中心切割（中心位置同原图）

        -   transforms.RandomCrop: 对图片进行随机切割（中心位置随机）

        -   transforms.RandomHorizontalFlip：对图片随机水平翻转，有一半几率水平翻转

        -   transforms.RandomSizeCrop:
            对图片进行随机大小的切割，然后resize到指定大小

        -   transforms.Pad: 填充

        -   transforms.ToTensor：转换图片为pytorch张量

        -   transforms.ToPILImage：转换pytorch张量或者numpy.ndarray为PIL图片

        -   transforms.Lambda: 使用指定转换器

    -   utils:

        -   make\_grid():

        -   save\_image(): 将指定pytorch张量存为图片

### Detectron（已废弃）

原**Caffe2**的图像处理框架，**Caffe2**合并入**Pytorch**后被废弃，由Detectron2继承。

### Maskrcnn-benchmark（已废弃）

一个Facebook的图像处理框架，已废弃，Detectron2继承

### Detectron2（计算机视觉）

Detectron2也是Facebook推出的计算机视觉库，包括**目标检测、语义分割、全景分割**等

Detectron基于Caffe2，而maskrcnn-benchmark基于Pytorch，Caffe2和Pytorch合并之后，Facebook在这两者基础上推出了**Detectron2**。

**代码**

[https://github.com/facebookresearch/detectron2]{.underline}

预训练模型介绍：

[[https://github.com/facebookresearch/detectron2/blob/master/MODEL\_ZOO.md]{.underline}](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)

**参考**

[[https://www.aiuai.cn/aifarm1288.html]{.underline}](https://www.aiuai.cn/aifarm1288.html)

-   config

    -   config：网络模型的配置文件

        -   CfgNode：描述网络模型的类，包括结构、权重、数据集等等

            -   merge\_from\_file()：从config文件中读取网络模型的设置

            -   dump()：

        -   get\_cfg()：返回当前的网络模型cfg

        -   set\_global\_cfg()：将当前的网络模型cfg，设置为参数给定的网络模型cfg

-   model\_zoo：包含各个网络模型的config文件（config目录），以及三个函数（model\_zoo模块中，参数均为config文件的相对路径）

    -   model\_zoo.get()：根据config文件的相对路径，得到对应的model对象

    -   model\_zoo.get\_config\_file()：将config文件的相对路径转换为绝对路径（相对于detectron2/model\_zoo/config）

    -   model\_zoo.get\_checkpoint\_url()：根据config文件名（相对路径），获得网络模型对应的预训练权重的URL

    -   config：保存各种网络模型的config文件（格式为yaml）

-   modeling：

    -   meta\_arch.build.build\_model()：根据cfg生成返回一个torch.nn.Module

    -   meta\_arch.panoptic\_fpn.PanopticFPN：全景分割FPN

    -   meta\_arch.rcnn.GeneralizedRCNN：通用化的RCNN，包括特征提取+Regional
        Proposal+每个候选区域预测

    -   meta\_arch.retinanet.RetinaNet：RetinaNet

    -   backbone.backbone.Backbone：描述基础网络的基类

    -   backbone.resnet.ResNet：基础网络ResNet

    -   backbone.resnet.ResNetBlockBase,BottleneckBlock,DeformBottleneckBlock：ResNet的各种block

    -   backbone.fpn.FPN：基础网络FPN

    -   backbone.fpn.build\_resnet\_fpn\_backbone()：生成一个以ResNet为基础网络的基础网络FPN

-   engine

    -   defaults.DefaultPredictor：根据CfgNode生成一个predictor

    -   defaults.DefaultTrainer：根据CfgNode生成一个trainer

        -   resume\_or\_load()：从checkpoint恢复，或者根据CfgNode生成模型

        -   train()：训练

-   checkpoint：保存模型的checkpoint

    -   detection\_checkpoint.Checkpointer：用于保存模型的checkpoint,参数为torch.nn.modules.module.Module类型

    -   detection\_checkpoint.DetectionCheckpointer：同Checkpointer，区别在于可以接受detectron和detectron2格式的model作为参数。

    -   detection\_checkpoint.PeriodicCheckpointer：

-   data：数据集相关

    -   catalog.DatasetCatalog：数据集的目录

        -   register()：注册新的数据集

        -   get()：

        -   list()：列出所有的注册的数据集

    -   catalog.MetadataCatalog：描述一个数据集的元数据，比如分成哪些类，每个类用什么颜色表示等

        -   get()：返回对应某个数据集的metadata（singleton实例）

        -   list()：列出所有注册了metadata的数据集

-   utils：各种工具

    -   logger：log相关

        -   setup\_logger()：初始化log

    -   visualizer

        -   Visualizer：图像的具像化工具，用原始图像初始化

            -   draw\_instance\_predictions()：画出图像的instance-level的预测结果

            -   draw\_sem\_seg()：画出图像的语义分割的预测结果

            -   draw\_panoptic\_seg\_predictions()：画出图像的全景分割（panoptic
                segmentation）的预测结果

            -   draw\_dataset\_dict()：画出图像（目标检测、图像分割等）的Ground
                Truth

### PySlowFast（视频理解）

**PySlowFast**是Facebook的**视频理解**的库，包括Slow、SlowFast、I3D、C2D、Non-local
network等几个网络模型。

PySlowFast类似Darknet，无须编写python等代码。训练、测试、推理是通过命令行进行，配置是通过配置文件。

github（也是**SlowFast模型**的仓库）：

[[https://github.com/facebookresearch/SlowFast]{.underline}](https://github.com/facebookresearch/SlowFast)

github仓库结构：

-   configs：配置文件（yaml格式），包括：

    -   对AVA数据集的配置：SLOW\_8x8\_R50\_SHORT.yaml等

    -   对Charades配置：

    -   对Kinetics的配置：

    -   对SSv2的配置

-   demo：演示相关，包括yaml配置文件、label文件

-   projects：multigrid相关介绍

-   slowfast：SlowFast源码

-   tools：

    -   run\_net.py：运行某个网络

    -   train\_net.py：训练某个网络

    -   test\_net.py：测试某个网络

    -   benchmark.py：衡量某个网络

    -   demo\_net.py

    -   visualization.py

-   setup.py：build脚本

-   各种md：安装、数据集下载、训练等的介绍文档

PySlowFast的运行、训练、测试需要：

-   **程序文件**：即tools目录下的run\_net.py等

-   **配置文件**：yaml格式，在configs目录下有（配置也可以在命令行被覆盖及指定）

-   **权重文件**：pkl格式的checkpoints，在配置文件中指定

命令行格式形如：

./tools/run\_net.py \--cfg configs/Kinetics/C2D\_8x8\_R50.yaml NUM\_GPUS
1

### Prophet（时间序列分析）

Facebook推出的一个时间序列分析算法。

官网：

[[https://facebook.github.io/prophet/]{.underline}](https://facebook.github.io/prophet/)

**参考**

[[https://zhuanlan.zhihu.com/p/52330017]{.underline}](https://zhuanlan.zhihu.com/p/52330017)

**代码**

[[https://github.com/facebook/prophet]{.underline}](https://github.com/facebook/prophet)

Amazon
------

Amazon的DL框架是DL生态的三强之一（另外两家是Google和Facebook），其底层库是**MXNet**，上层接口是**gluon**，此外包括**gluoncv**（机器视觉）、**gluonts**（时间序列）和**gluonnlp**（NLP）等专用框架。

### MXNet（基础框架）

**MXNet**是亚马逊的深度学习框架，算是当前三大深度学习框架之一，但相较于主流的**Pytorch**（Facebook）和**Tensorflow**/**Keras**（Google），流行程度要差一些。

官方API文档：[[https://mxnet.apache.org/api/python/docs/api/index.html]{.underline}](https://mxnet.apache.org/api/python/docs/api/index.html)

-   mxnet.nd （mxnet.ndarray）

    -   NDArray：mxnet的张量类

        -   attach\_grad()：申请计算梯度所需要的内存

    -   array()：

    -   ones()：生成所有元素值为1的张量，参数为shape

    -   zeros()：生成所有元素值为0的张量，参数为shape

    -   random：随机数相关

        -   normal()：生成正态分布的随机数

-   mxnet.autograd ：自动求导相关

    -   record()：返回一个上下文，要求MXNet记录与梯度相关的计算

    -   backward()：计算梯度

-   mxnet.image：图像相关

-   mxnet.gluon：MXNet的高层接口，类似Keras之于Tensorflow

    -   data：数据操作工具

        -   DataLoader：从数据集中加载数据并返回mini-batch的迭代器

    -   loss：损失函数

        -   SoftmaxCrossEntropyLoss：Softmax及交叉熵损失

    -   nn：各神经网络层

    -   Trainer：

### Gluon（高级接口）

gluon是Amazon推出的高级接口，基于MXNet，gluon包括几个包：

-   gluon：类似于Keras，已合并到mxnet下

<!-- -->

-   gluon.models

### gluoncv（图像处理）

gluoncv是Amazon的**图像处理**相关的包，包括**图像分类、目标检测、图像分割**任务。

官网：

[[https://gluon-cv.mxnet.io/]{.underline}](https://gluon-cv.mxnet.io/)

-   loss：损失函数

-   nn：

    -   bbox：bounding box相关操作

    -   block：

    -   coder：

-   data：数据集相关

-   utils：工具集

-   model\_zoo：

    -   model\_zoo：Model Zoo

        -   get\_model()：获取指定的网络模型实例

        -   get\_model\_list()：获取支持的网络模型列表

    -   alexnet：

        -   AlexNet：AlexNet的类

        -   alexnet：返回AlexNet的实例

    -   \...：各种网络模型的实现

### gluonts（时间序列）

参考《[[研究方向：TS \> 框架 \> gluonts]{.underline}](\l)》

### gluonnlp（自然语言处理）

gluonnlp是亚马逊推出的自然语言的专用框架。

CUHK
----

CUHK（香港中文大学）在**计算机视觉**方面颇有建树，推出了**mmcv**、**mmdetection**、**mmsegmentation**、**mmaction**等几个开源的框架及库。这几个库都基于Pytorch，其中：

-   mmdetection是**图像处理**方面的

-   mmsegmentation是**图像分割**方面的

-   mmaction是**视频理解**方面的

-   mmcv则无关神经网络，是传统CV方面的。

### mmdetection（图像处理）

**mmdetection**是港中文大学推出的**图像处理**框架，包括图像分类、目标检测、图像分割等任务。mmdetection基于Pytorch。

**代码**

[[https://github.com/open-mmlab/mmdetection]{.underline}](https://github.com/open-mmlab/mmaction)

文档：

[[https://mmdetection.readthedocs.io/en/latest/]{.underline}](https://mmdetection.readthedocs.io/en/latest/)

mmdetection的github仓库目录结构：

-   setup.py：编译、安装的脚本

-   config：描述模型的配置文件

-   demo：存放demo

-   docker：docker相关

-   docs：文档

-   mmdet：mmdetection的源码目录，也是安装之后site-packages下mmdet目录的来源

-   requirements：依赖的python包列表

-   tests：

-   tools：用于训练、推理的脚本

### mmsegmentation（图像分割）

mmsegmentation是CUHK推出的**图像分割**专用框架。

支持的骨干网络包括：ResNet、ResNeXt、HRNet等。

支持的模型包括：FCN、PSPNet、DeepLabV3、PSANet、UPerNet、[NonLocal
Net](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/nonlocal_net)、[EncNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/encnet)、[CCNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ccnet)、[DANet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/danet)、[GCNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/gcnet)、[ANN](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ann)、[OCRNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ocrnet)等

**代码**

[[https://github.com/open-mmlab/mmsegmentation]{.underline}](https://github.com/open-mmlab/mmsegmentation)

**参考**

[[https://zhuanlan.zhihu.com/p/164489668]{.underline}](https://zhuanlan.zhihu.com/p/164489668)

### mmaction（视频理解）

**mmaction**是CUHK港中文大学推出的**视频理解**专用框架。

-   全面支持视频动作分析的各种任务，包括**动作识别**（action
    recognition）、**时序动作检测**（temporal action
    detection）以及**时空动作检测**（spatial-temporal action
    detection）。

    -   **时序动作检测**目前仅支持基于**thumos14**数据集训练的**SSN**。

    -   **时空动作检测**目前仅支持基于**ava**数据集训练的**fast-rcnn**。

-   支持多种流行的数据集，包括
    **Kinetics、THUMOS、UCF101、ActivityNet、Something-Something**、以及
    **AVA** 等。

-   已实现多种动作分析算法框架，包括 TSN、I3D、SSN、以及新的
    spatial-temporal action detection 方法。MMAction 还通过 Model Zoo
    提供了多个预训练模型，以及它们在不同数据集上的性能指标。

-   采用高度模块化设计。用户可以根据需要对不同模块，比如 backbone
    网络、采样方案等等进行灵活重组，以满足不同的应用需要。

**参考**

[[https://baijiahao.baidu.com/s?id=1636836312959143467&wfr=spider&for=pc]{.underline}](https://baijiahao.baidu.com/s?id=1636836312959143467&wfr=spider&for=pc)

**代码**

[[https://github.com/open-mmlab/mmaction]{.underline}](https://github.com/open-mmlab/mmaction)

mmaction的github仓库目录结构如下：

-   compile.sh：编译脚本

-   setup.py：安装脚本（python setup.py develop）

-   data：空目录，用于存放会用到的数据集（由data\_tools中的脚本下载）

-   data\_tools：各个数据集的下载脚本

    -   ava：AVA数据集的工具脚本

        -   download\_annotations.sh：下载标签

        -   download\_videos.sh：下载视频

        -   extract\_frames.sh：抽取视频帧的脚本

    -   hmdb51：HMDB51的工具脚本

    -   kinetics400：Kinetics400的工具脚本

    -   thumos14：Thumos14的工具脚本

    -   ucf101：UCF101的工具脚本

-   mmaction：mmaction的源码目录，安装后python的site-pacakge/mmaction.egg-link也连接至此

-   modelzoo：空目录，用于存放下载的模型权重文件

-   configs和test\_configs：模型的配置文件

    -   ava：时空行为检测的网络模型，目前仅有AVA数据集上的**Fast-RCNN**模型

        -   ava\_fast\_rcnn\_nl\_r50\_c4\_1x\_kinetics\_pretrain\_crop.py

    -   hmdb51：hmdb51数据集上的模型

    -   thumos14：时序行为检测的网络，目前仅有thumos14数据集上的**SSN**

        -   ssn\_thumos14\_rgb\_bn\_inception.py：

    -   CSN：CSN网络模型的配置

    -   I3D\_Flow：I3D光流模型的配置

    -   I3D\_RGB：I3D RGB模型的配置

    -   R2plus1D：R(2+1)D模型的配置

    -   SlowFast：SlowFast模型的配置

    -   SlowOnly：Slow模型的配置

    -   TSN：TSN模型的配置

-   third\_party：用到的第三方源码

-   tools：训练、测试各种网络的脚本

    -   test\_recognizer.py：测试**行为识别**网络

    -   test\_localizer.py：测试**时序行为识别**网络

    -   test\_detector.py：测试**时空行为识别**网络

-   INSTALL.md：安装指南

-   MODEL\_ZOO.md：下载预训练权重的指南

-   DATASET.md：下载数据集的指南

-   GETTING\_STARTED.md：操作指南

mmaction的训练、测试需要：

-   **程序文件**：即tools目录下的py文件等

-   **配置文件**：configs目录下的py文件

-   **权重文件**：.pth格式文件，一般下载到modelzoo中

命令行格式形如：

./tools/test\_xxx.py configs/xxx.py

例如

./tools/test\_detector.py
test\_configs/TSN/tsn\_kinetics400\_2d\_rgb\_r50\_seg3\_f1s1.py

图森未来
--------

### SimpleDet（图像处理）

**SimpleDet**是**图森未来（TuSimple）**推出的图像处理框架。包括**目标检测**和**实例分割**任务

代码：

[[https://github.com/TuSimple/simpledet]{.underline}](https://github.com/TuSimple/simpledet)

Hugging Face
------------

Hugging
Face总部位于纽约，是一家专注于自然语言处理、人工智能和分布式系统的创业公司。他们所提供的聊天机器人技术一直颇受欢迎，但更出名的是他们在NLP开源社区上的贡献。Huggingface一直致力于自然语言处理NLP技术的平民化(democratize)，希望每个人都能用上最先进(SOTA,
state-of-the-art)的NLP技术，而非困窘于训练资源的匮乏。

官方网站：

[[https://huggingface.co/]{.underline}](https://huggingface.co/)

**代码**

[[https://github.com/huggingface]{.underline}](https://github.com/huggingface)

### Transformers

参考《[[研究方向：NLP \> 框架 \> Transformers]{.underline}](\l)》

阿里巴巴
--------

### MNN

参考《[[边缘计算 \> MNN]{.underline}](\l)》

### 腾讯

### ncnn

参考《[[边缘计算 \> ncnn]{.underline}](\l)》

Redmon
------

Joseph
Redmon是著名的目标检测模型**YOLO**的作者，YOLO并没有使用各种流行框架，还是他自己实现了一个简单的Darknet框架（C语言），通过配置文件定义网络模型。

Redmon自己实现了三个版本的YOLO（即YOLO、YOLOv2、YOLOv3），之后为了抗议YOLO被用于军事和隐私问题（他自己讲的，我看其实是他觉得烦了），不再做YOLO的研发。

而YOLOv4算是"半官方"，是他的同事开发的（也一直有在参与之前YOLO的开发），主要是加入了很多trick使得模型更准确，并没有什么特别创新的思想提出。YOLOv5则完全不官方，是一家德国公司做的工程化的产品，也是加了大量的trick，效果和速度据说比YOLOv4更好。

### Darknet

**Darknet**是**YOLO**系列原作者Redmon开发的一套框架，C语言的实现和接口。Darknet虽然通用性不够，但是工程化相当不错。

官网：

[[https://pjreddie.com/darknet/yolo/]{.underline}](https://pjreddie.com/darknet/yolo/)

代码：

[[https://github.com/pjreddie/darknet]{.underline}](https://github.com/pjreddie/darknet)

**Darknet**是C语言的，下载（clone）后通过make生成一个名为darknet的可执行文件，该文件是darknet框架的主文件，可以用于训练、推理等等一系列操作。

**混淆注意！**

Darknet是**有歧义的**，可以指代这个**深度学习框架**，也可以指代使用这个框架开发的一个**网络模型**，这个网络模型本身可以用于图像分类，同时也是**YOLO**（**目标检测**网络）的骨干网络。

#### 使用说明

以下为Darknet主程序（用于训练、执行darknet格式的网络配置和权重）使用说明

-   下载及编译

git clone https://github.com/pjreddie/darknet

cd darknet

可以选择修改Makefile（GPU=1, CUDNN=1, OPENCV=1)

make

-   获取权重

cd cfg

wget https://pjreddie.com/media/files/yolov3.weights

wget https://pjreddie.com/media/files/yolov3-tiny.weights

-   图像分类

./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg
cfg/darknet19.weights data/dog.jpg

or input image path on prompt:

./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg
cfg/darknet19.weights

-   目标检测

./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg

相当于：

./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights
data/dog.jpg

手动输入图片路径：

./darknet detect cfg/yolov3.cfg yolov3.weights

指定阈值 (缺省为0.25):

./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0.1

使用yolo-tiny:

./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg

摄像头实时检测：

./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights

检测视频文件：

./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights
\<video file\>

-   训练

在VOC上训练（需要设置好其他配置）：

./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg
darknet53.conv.74

在COCO上训练（需要设置好其他配置）：

./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74

-   冻结部分网络（前N层，下例为15）

./darknet partial cfg/yolov3-tiny.cfg yolov3-tiny.weights
yolov3-tiny.conv.15 15

-   OpenCV测试

./darknet imtest data/eagle.jpg

#### 文件说明

-   darknet文件

darknet的主体，编译生成的ELF执行文件

-   .cfg文件

Darknet的.cfg文件用于描述**网络结构**及**超参数**，通常位于cfg目录下，其参数定义：

\[net\] 字段

[[https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section]{.underline}](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5bnet%5d-section)

层字段

[[https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-different-layers]{.underline}](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-different-layers)

-   .weights文件

.weights文件是权重数据，须与cfg文件描述的网络结构对应

-   .data文件

.data文件用于配置目标检测器（Object Detection）或者图像分类器（Image
classification），通常位于cfg目录下。

目标检测器的配置cfg/coco.data，定义如下：

classes= 20 \#种类

train = /home/pjreddie/data/voc/train.txt \#训练集图片列表

valid = /home/pjreddie/data/voc/2007\_test.txt \#验证集图片列表

names = data/voc.names \#各个分类的名字

backup = backup \#存放训练中备份的权重文件的目录

-   数据格式

YOLO有自己的数据格式，训练集或者验证集的图片列表被放在某个txt文件里，这些文件被.data文件中的train、valid字段指定。训练集图片的label文件有如下格式：

1.  label文件与图片文件同名，扩展名改为txt

2.  label文件与图片文件同目录，或者图片文件在images目录,label文件在同级labels目录，形如aaa/bbb/images/pic1.jpg、aaa/bbb/labels/pic1.txt

3.  label文件中每一行代表一个object，有5个值，分别为类别、中心x、中心y、宽度、高度，这5个值以空格隔开，后面四个值取值范围从0到1，为点或者长度占全宽/全高的比例。

-   .names文件

.names文件通常位于data目录中，按行存放了每个分类的名字

其它
----

### PyVideoResearch

代码：

[[https://github.com/gsig/PyVideoResearch]{.underline}](https://github.com/gsig/PyVideoResearch)

### Theano

Theano是蒙特利尔大学开发的深度学习框架，流行性低，且已经停止开发。

官方API文档：

[[http://www.deeplearning.net/software/theano/library/index.html]{.underline}](http://www.deeplearning.net/software/theano/library/index.html)

### DNN (OpenCV)

OpenCV的DNN模块（Deep Neural Network）

doc：

[[https://docs.opencv.org/master/d2/d58/tutorial\_table\_of\_content\_dnn.html]{.underline}](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)

传统机器学习（sklearn）
=======================

本章结合sklearn，介绍了（**非深度学习的**）**传统机器学习**。sklearn（scikit-learn）是传统机器学习最主流和热门的python库。

**机器学习（Machine
Learning，ML）**是人工智能（AI）的一个分支，是深度学习的**超集**。

通过算法和统计数据的学习，来完成特定的任务。ML无需明确的指令，而是依赖于模式和推论。ML基于样本数据（训练数据集）建立一个数学模型，以便在不明确编码的情况下来完成预测或者决定。

机器学习已广泛应用于数据挖掘、计算机视觉、自然语言处理、生物特征识别、搜索引擎、医学诊断、检测信用卡欺诈、证券市场分析、DNA序列测序、语音和手写识别、战略游戏和机器人等领域。

机器学习有下面几种定义：

-   机器学习是一门人工智能的科学，该领域的主要研究对象是人工智能，特别是如何在经验学习中改善具体算法的性能。

-   机器学习是对能通过经验自动改进的计算机算法的研究。

-   机器学习是用数据或以往的经验，以此优化计算机程序的性能标准。

一种经常引用的英文定义是：A computer program is said to learn from
experience **E** with respect to some class of tasks **T** and
performance measure **P**, if its performance at tasks in **T**, as
measured by **P**, improves with experience **E**.

**机器学习**和**数据挖掘**的关系：

机器学习和数据挖掘经常使用同样的方法，且他们高度重合。

**机器学习**和**深度学习**的关系：

深度学习是机器学习的一种，是使用DNN（深度神经网络）的机器学习。

本节包含了归属于**机器学习**（**Machine
Learning**），但不属于**深度学习**（**Deep Learning**）的概念及算法。

这里有张关系图：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image80.jpeg){width="5.092361111111111in"
height="3.1326388888888888in"}

sklearn
-------

sklearn（scikit-learn）是机器学习领域最知名的算法库，下面的章节会按照sklearn的tutorial的结构对算法进行解释。

官方中文文档：

[[https://sklearn.apachecn.org/]{.underline}](https://sklearn.apachecn.org/)

官方英文文档：

[[https://scikit-learn.org/stable/user\_guide.html]{.underline}](https://scikit-learn.org/stable/user_guide.html)

sklearn官方网站有张图，教你如何选择合适的算法：

[[https://scikit-learn.org/stable/\_static/ml\_map.png]{.underline}](https://scikit-learn.org/stable/_static/ml_map.png)

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image81.png){width="6.39375in"
height="3.9868055555555557in"}

概念
----

### 监督学习

**监督学习**（**Supervised
learning**），是机器学习的一种方法，可以由训练资料中学到或建立一个模式（函数
/ learning
model），并依此模式推测新的实例。训练资料是由输入（通常是向量）和预期输出所组成。根据函数的输出，监督学习可主要分为两大类：

-   **回归分析**：输出是一个连续的值，例如猜测一个房子的价格

-   **分类**：输出是离散的值，例如预测一个分类标签

监督学习在于学习两个数据集的联系：观察数据 X
和我们正在尝试预测的额外变量 y (通常称"目标"或"标签")， 而且通常是长度为
n\_samples 的一维数组。

如果预测任务是为了将观察值分类到有限的标签集合中，换句话说，就是给观察对象命名，那任务就被称为**分类**任务。另外，如果任务是为了预测一个连续的目标变量，那就被称为**回归**任务。

目前最广泛被使用的分类器有**人工神经网络**（即**深度学习**）**、支持向量机、最近邻居法、高斯混合模型、朴素贝叶斯方法、决策树**和**径向基函数分类**。

**主动式学习**：一个情况是，有大量尚未标示的资料，但去标示资料则是很耗成本的。一种方法则是，学习算法会主动去向使用者或老师去询问标签。
这种形态的监督式学习称为主动式学习。既然学习者可以选择例子，学习中要使用到的例子个数通常会比一般的监督式学习来得少。
以这种策略则有一个风险是，算法可能会专注在于一些不重要或不合法的例子。

**参考**

[[维基百科]{.underline}](https://zh.wikipedia.org/wiki/%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0)

### 分类

**分类（Classification）**是机器学习非常重要的一个组成部分，它的目标是根据已知样本的某些特征，判断一个新的样本属于哪种已知的样本类。分类是**监督学习**的一个实例，根据已知训练集提供的样本，通过计算选择特征参数，创建判别函数以对样本进行的分类。与之相对的是**无监督学习**，例如**聚类分析**。

分类观测的变量是离散的，分类是监督学习的两大方向之一（另一个是回归分析）。

### 回归分析

**回归分析（Regression
Analysis）**是一种统计学上分析数据的方法，目的在于了解两个或多个变量间是否相关、相关方向与强度，并建立数学模型以便观察特定变量来预测研究者感兴趣的变量。更具体的来说，回归分析可以帮助人们了解在只有一个自变量变化时因变量的变化量。一般来说，通过回归分析我们可以由给出的自变量估计因变量的条件期望。

回归分析是建立因变量Y（或称依变量，反因变量）与自变量X或称独变量，解释变量）之间关系的模型。简单线性回归使用一个自变量X，复回归使用超过一个自变量（X~1~,
X~2~...X~i~）。

回归的最早形式是最小二乘法。

### 无监督学习

**无监督学习（Unsupervised
learning）**是机器学习的一种类型，没有给定事先标记过的训练示例，自动对输入的数据进行分类或分群。无监督学习的主要运用包含：**聚类分析(Cluster
Analysis)、关系规则(Association Rule)、维度缩减(Dimensionality
Reduce)**。它是监督式学习和强化学习之外的一种选择。

一个常见的无监督学习是数据**聚类**。在人工神经网络中，**生成对抗网络（GAN）**、**自组织映射（SOM）**和**适应性共振理论（ART）**则是最常用的非监督式学习。

ART模型允许集群的个数可随着问题的大小而变动，并让用户控制成员和同一个集群之间的相似度分数，其方式为透过一个由用户自定而被称为[警觉参数]{.underline}的常量。ART也用于[模式识别]{.underline}，如[自动目标识别]{.underline}和[数字信号处理]{.underline}。第一个版本为\"ART1\"，是由卡本特和葛罗斯柏格所发展的。

### 聚类分析

**聚类分析（Cluster
analysis）**亦称为群集分析，是对于统计数据分析的一门技术，在许多领域受到广泛应用，包括**机器学习，数据挖掘，模式识别，图像分析以及生物信息**。聚类是把相似的对象通过静态分类的方法分成不同的组别或者更多的子集（subset），这样让在同一个子集中的成员对象都有相似的一些属性，常见的包括在坐标系中更加短的空间距离等。

### 强化学习

**强化学习（Reinforcement
learning，RL）**是机器学习中的一个领域，强调如何基于环境而行动，以取得最大化的预期利益。其灵感来源于心理学中的行为主义理论，即有机体如何在环境给予的奖励或惩罚的刺激下，逐步形成对刺激的预期，产生能获得最大利益的习惯性行为。这个方法具有普适性，因此在其他许多领域都有研究，例如博弈论、控制论、运筹学、信息论、仿真优化、多主体系统学习、群体智能、统计学以及遗传算法。在运筹学和控制理论研究的语境下，强化学习被称作"近似动态规划"（approximate
dynamic
programming，ADP）。在最优控制理论中也有研究这个问题，虽然大部分的研究是关于最优解的存在和特性，并非是学习或者近似方面。在经济学和博弈论中，强化学习被用来解释在有限理性的条件下如何出现平衡。

在机器学习问题中，环境通常被规范为**马可夫决策过程（MDP）**，所以许多强化学习算法在这种情况下使用动态规划技巧。传统的技术和强化学习算法的主要区别是，后者不需要关于MDP的知识，而且针对无法找到确切方法的大规模MDP。

强化学习和标准的**监督学习**之间的区别在于，它并不需要出现正确的输入/输出对，也不需要精确校正次优化的行为。强化学习更加专注于在线规划，需要在探索（在未知的领域）和遵从（现有知识）之间找到平衡。强化学习中的"探索-遵从"的交换，在多臂老虎机问题和有限MDP中研究得最多。

### L1/L2正则化

机器学习中，为了避免过拟合，可以在**损失函数**中添加L1及L2正则化项（参考
[[概念及定义 \> 线性代数/几何学 \> 范数]{.underline}](\l)）。

L2正则化即在原来的损失函数上添加权重参数的平方和：

![](/home/jimzeus/outputs/AANN/images/media/image82.png){width="1.1909722222222223in"
height="0.41180555555555554in"}

L2正则化过的线性回归即为**岭回归**。

L1正则化则是在原来的损失函数上添加权重参数的绝对值之和：

![](/home/jimzeus/outputs/AANN/images/media/image83.png){width="1.3798611111111112in"
height="0.41180555555555554in"}

L1正则化过的线性回归为**Lasso回归**。

**参考**

机器学习中L1和L2正则化的直观解释

[[https://blog.csdn.net/red\_stone1/article/details/80755144]{.underline}](https://blog.csdn.net/red_stone1/article/details/80755144)

### 感知器

**感知器（Perceptron）**是Frank
Rosenblatt在1957年就职于康奈尔航空实验室（Cornell Aeronautical
Laboratory）时所发明的一种**人工神经网络**。它可以被视为一种最简单形式的前馈神经网络，是一种二元线性分类器。

Frank
Rosenblatt给出了相应的感知机学习算法，常用的有感知机学习、最小二乘法和梯度下降法。譬如，感知机利用梯度下降法对损失函数进行极小化，求出可将训练数据进行线性划分的分离超平面，从而求得感知机模型。

在人工神经网络领域中，感知机也被指为单层的**人工神经网络**，以区别于较复杂的多层感知机（MLP，Multi-Layer
Perceptron）。作为一种线性分类器，（单层）感知机可说是最简单的前向人工神经网络形式。尽管结构简单，感知机能够学习并解决相当复杂的问题。感知机主要的本质缺陷是它不能处理线性不可分问题。

### 隐马尔可夫模型

**隐马尔可夫模型**（**Hidden Markov
Model，HMM**）或称作隐性马尔可夫模型，是统计模型，它用来描述一个含有隐含未知参数的**马尔可夫过程**。其难点是从可观察的参数中确定该过程的隐含参数。然后利用这些参数来作进一步的分析。

任何一个HMM都可以通过下列五元组来描述：

-   observations：观测序列

-   states：隐状态

-   start\_prob：初始概率（隐状态）

-   trans\_prob：转移概率（隐状态）

-   emit\_prob：发射概率 （隐状态表现为显状态的概率）

隐马尔可夫模型作了两个基本假设（**o~t~**为t时刻的观测状态，**i~t~**为t时刻的隐状态）：

-   **马尔可夫性假设**，即假设**隐藏的马尔可夫链**在任意时刻t的状态只依赖于其前一时刻的状态，与其它时刻的状态及观测无关，也与时刻t无关：

    **p(i~t~\|i~t-1~,o~i-1~,\...,i~1~,o~1~) = p(i~t~\|i~t-1~),
    t=1,2,\...,T**

-   **观测独立性假设**，即假设任意时刻的观测只依赖于该时刻的马尔可夫链的状态，与其他观测及状态无关：

    **p(o~t~\|i~T~,o~T~,i~T-1~,o~T-1~,\...,i~t+1~,o~t+1~,i~t~,o~t~,\...,i~1~,o~1~)
    = p(o~t~\|i~t~)**

举一个经典的例子：一个东京的朋友每天根据天气{下雨，天晴}决定当天的活动{公园散步,购物,清理房间}中的一种，我每天只能在twitter上看到她发的推"啊，我前天公园散步、昨天购物、今天清理房间了！"，那么我可以根据她发的推特推断东京这三天的天气。在这个例子里，显状态是活动，隐状态是天气。

states = (\'Rainy\', \'Sunny\')

observations = (\'walk\', \'shop\', \'clean\')

start\_prob = {\'Rainy\': 0.6, \'Sunny\': 0.4}

trans\_prob = { \'Rainy\' : {\'Rainy\': 0.7, \'Sunny\': 0.3}, \'Sunny\'
: {\'Rainy\': 0.4, \'Sunny\': 0.6}, }

emission\_prob = {\'Rainy\' : {\'walk\': 0.1, \'shop\': 0.4, \'clean\':
0.5}, \'Sunny\' : {\'walk\': 0.6, \'shop\': 0.3, \'clean\': 0.1}, }

**参考**

[[https://www.zhihu.com/question/20962240]{.underline}](https://www.zhihu.com/question/20962240)

[[https://blog.csdn.net/hudashi/article/details/87867916]{.underline}](https://blog.csdn.net/hudashi/article/details/87867916)

### 神经网络

**人工神经网络（Artificial Neural
Network，ANN）**，也叫**神经网络（Neural
Network，NN）**或**类神经网络**，在机器学习和认知科学领域，是一种模仿生物神经网络（动物的中枢神经系统，特别是大脑）的结构和功能的数学模型或计算模型，用于对函数进行估计或近似。神经网络由大量的人工神经元联结进行计算。大多数情况下人工神经网络能在外界信息的基础上改变内部结构，是一种自适应系统，通俗的讲就是具备学习功能。现代神经网络是一种非线性统计性数据建模工具。

神经网络的构筑理念是受到生物（人或其他动物）神经网络功能的运作启发而产生的。

工神经网络通常是通过一个基于数学统计学类型的学习方法（Learning
Method）得以优化，所以也是数学统计学方法的一种实际应用，通过统计学的标准数学方法我们能够得到大量的可以用函数来表达的局部结构空间，另一方面在人工智能学的人工感知领域，我们通过数学统计学的应用可以来做人工感知方面的决定问题（也就是说通过统计学的方法，人工神经网络能够类似人一样具有简单的决定能力和简单的判断能力），这种方法比起正式的逻辑学推理演算更具有优势。

和其他机器学习方法一样，神经网络已经被用于解决各种各样的问题，例如**机器视觉**和**语音识别**。这些问题都是很难被传统基于规则的编程所解决的。

**支持向量机**和其他更简单的方法（例如**线性分类器**）在机器学习领域的流行度层逐渐超过了神经网络，但是在2000年代后期出现的**深度学习**重新激发了人们对神经网络的兴趣。

**参考**

[[维基百科]{.underline}](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)

算法
----

### 线性模型

**线性模型**（**Linear Model**）既可以用于**回归**（比如线性回归Linear
Regression），也可以用于**分类**（比如逻辑回归Logistic Regression）。

#### 线性回归

假设在问题中有因变量y以及自变量x~1~,x~2~,\...,x~n~，可以设想y的值由两部分组成：一部分由自变量x的影响所致，这一部分表示为x到y的函数形式f(x~1~,x~2~,\...,x~n~)，另一部分则由其他因素导致，可视为一种随机误差，记为b，于是得到模型：

**y = f(x~1~,x~2~,\...,x~n~) + b**

函数f即被称为y对f(x~1~,x~2~,\...,x~n~)的**回归函数**。

如果回归函数为线性函数，即：

**y = w~0~ + w~1~x~1~ + w~2~x~2~ + \... + w~n~x~n~+ b**

则被称为**线性回归**（**Linear Regression**）。

**损失函数**也叫**目标函数**，是用于度量预测的好坏（即模型预测值y^\^^与真实值y之间的误差）。

**参考**

线性回归模型的原理、公式推导、Python实现和应用

[[https://zhuanlan.zhihu.com/p/80887841]{.underline}](https://zhuanlan.zhihu.com/p/80887841)

用人话讲明白线性回归

[[https://zhuanlan.zhihu.com/p/72513104]{.underline}](https://zhuanlan.zhihu.com/p/72513104)

#### 最小二乘法

**最小二乘法**（**Ordinary Least
Squares**）是损失函数为**MSE**的**线性回归**（其英文名称更加直观）。所谓二乘，指的就是平方（在台湾翻译为最小平方法）。

最小二乘法的损失函数MSE：

**L =**
$\frac{\mathbf{1}}{\mathbf{2m}}\sum_{\mathbf{i = 1}}^{\mathbf{m}}\mathbf{(y - y\hat{})}$**^2^**

**混淆注意：**

**线性回归**有时也被用来指代**最小二乘法**，而**最小二乘法**有时也用于指代其损失函数**MSE**。

**参考**

最小二乘法的本质是什么？

[[https://www.zhihu.com/question/37031188/answer/411760828]{.underline}](https://www.zhihu.com/question/37031188/answer/411760828)

#### 岭回归（Ridge regression）

**岭回归**（**Ridge Regression**），也叫**吉洪诺夫正则化（Tikhonov
regulation）**,其和普通线性回归的区别在于**损失函数**是一种改良的最小二乘法，在MSE的基础上增加了一个惩罚项（L2正则化项）。

通过放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价获得回归系数更为符合实际、更可靠的回归方法，对病态数据的拟合要强于最小二乘法。

岭回归和最小二乘法的区别在于**损失函数**多了一个L2正则化项，即所有系数的平方：

![](/home/jimzeus/outputs/AANN/images/media/image82.png){width="1.1909722222222223in"
height="0.41180555555555554in"}

其中E~in~为最小二乘法的损失函数MSE

**参考**

深入理解L1、L2正则化

[[https://zhuanlan.zhihu.com/p/29360425]{.underline}](https://zhuanlan.zhihu.com/p/29360425)

#### Lasso回归

**Lasso回归（Least Absolute Shrinkage and Selection
Operator）**类似岭回归，也是在最小二乘法的损失函数上增加了一个惩罚项（L1正则化），及所有系数的绝对值。

采用了L1正则会使得部分学习到的特征权值为0，从而达到稀疏化和特征选择的目的。

其损失函数为：

![](/home/jimzeus/outputs/AANN/images/media/image83.png){width="1.3798611111111112in"
height="0.41180555555555554in"}

其中E~in~为最小二乘法的损失函数MSE。

#### 弹性网络回归（Elastic Net）

**弹性网络回归（Elastic Net
Regression）**，是岭回归和Lasso回归的结合，其损失函数同时使用了L1和L2正则项：

**L = E~in~ + λ~1~**$\sum_{\mathbf{j}}^{}\mathbf{|w}$**~j~\| +
λ~2~**$\sum_{\mathbf{j}}^{}\mathbf{w}$**~j~^2^**

**参考**

[[https://zhuanlan.zhihu.com/p/61929632]{.underline}](https://zhuanlan.zhihu.com/p/61929632)

#### 最小角回归（LARS）

**最小角回归（Least Angle Regression）**，

#### OMP

Orthogonal Matching Pursuit，

#### 贝叶斯线性回归

#### 逻辑回归

**参考**

用人话讲明白逻辑回归

[[https://zhuanlan.zhihu.com/p/139122386]{.underline}](https://zhuanlan.zhihu.com/p/139122386)

#### 广义线性回归

#### SGD

#### 感知器

#### 被动攻击算法

#### 稳健回归（Robustness Regression)

#### 多项式回归（Polynomial regression）

### 核岭回归

**参考**

[[https://zhuanlan.zhihu.com/p/72517223]{.underline}](https://zhuanlan.zhihu.com/p/72517223)

### SVM（支持向量机）

在机器学习中，**支持向量机（support vector machine，
SVM，支持向量网络）**是在分类与回归分析中分析数据的监督式学习模型与相关的学习算法。

给定一组训练样例，每个样例被标记为属于两个类别中的一个或另一个，SVM训练算法创建一个将新的样例分配给两个类别之一的模型，使其成为非概率二元线性分类器。

将实例表示为空间中的点，SVM模型的映射就使得单独类别的实例被尽可能宽的明显的间隔分开。然后，将新的实例映射到同一空间，并基于它们落在间隔的哪一侧来预测所属类别。

当数据未被标记时，不能进行**监督式学习**，需要用**非监督式学习**，它会尝试找出数据到簇的自然聚类，并将新数据映射到这些已形成的簇。将支持向量机改进的聚类算法被称为**支持向量聚类**，当数据未被标记或者仅一些数据被标记时，支持向量聚类经常在工业应用中用作分类步骤的预处理。

假设某些给定的数据点各自属于两个类之一，而目标是确定新数据点将在哪个类中。对于支持向量机来说，数据点被视为**p维向量**，而我们想知道是否可以用**(p-1)维超平面**来分开这些点。这就是所谓的**线性分类器**。可能有许多超平面可以把数据分类。最佳超平面的一个合理选择是以最大间隔把两个类分开的超平面。因此，我们要选择能够让到每边最近的数据点的距离最大化的超平面。如果存在这样的超平面，则称为最大间隔超平面，而其定义的线性分类器被称为最大间隔分类器。

更正式地来说，支持向量机在高维或无限维空间中构造超平面或超平面集合，其可以用于分类、回归或其他任务。直观来说，分类边界距离最近的训练数据点越远越好，因为这样可以缩小分类器的泛化误差。

**参考**

[[维基百科]{.underline}](https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA)

\[机器学习\]支持向量机SVM（非常详细）

[[https://zhuanlan.zhihu.com/p/77750026]{.underline}](https://zhuanlan.zhihu.com/p/77750026)

用人话讲明白支持向量机

上：[[https://zhuanlan.zhihu.com/p/73477179]{.underline}](https://zhuanlan.zhihu.com/p/73477179)

下：[[https://zhuanlan.zhihu.com/p/74484361]{.underline}](https://zhuanlan.zhihu.com/p/74484361)

#### 线性可分

首先以二维数据为例，说明什么是线性分类：

假设有两类要区分的样本点，一类用黄色圆点代表，另一类用红色方形代表，中间这条直线就是一条能将两类样本完全分开的分类函数。

用前面的数学化语言描述一下这个图，就是：

-   **样本数据**：11个样本，2个输入(x~1~,x~2~)，一个输出y

-   **第i个样本的输入**：X~i~ = (x^i^~1~,x^i^~2~)^T^, i=1,2,3,\...,11

-   **输出y**：用1（红色方形）和-1（黄色圆点）作为标签

-   **训练样本集D**：![](/home/jimzeus/outputs/AANN/images/media/image84.png){width="1.6118055555555555in"
    height="0.6666666666666666in"}

-   **训练目的**：以训练样本为研究对象，找到一条直线w~1~x~1~+w~2~x~2~+b=0，将两类样本有效分开

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image85.png){width="2.4347222222222222in"
height="2.397222222222222in"}

这里我们引出**线性可分**的定义：**如果一个线性函数能够将样本分开，就称这些样本是线性可分的**。线性函数在一维空间里，就是一个小小的点；在二维可视化图像中，是一条直直的线；在三维空间中，是一个平平的面；在更高维的空间中，是无法直观用图形展示的超平面。

回想一下线性回归：

-   在一元线性回归中我们要找拟合一条直线，使得样本点（x，y）都落在直线附近

-   在二元线性回归中，要拟合一个平面，使得样本点（x1，x2，y）都落在该平面附近

-   在更高维的情况下，就是拟合超平面。

对应的线性分类：

-   当样本点为（x，y）时（注意，和回归不同，由于y是分类标签，y的数字表示是只有属性含义，是没有真正的数值意义的，因此当只有一个自变量时，不是二维问题而是一维问题），要找到一个点wx+b=0，即x=-b/w这个点，使得该点左边的是一类，右边的是另一类。

-   当样本点为（x~1~,x~2~,
    y）时，要找到一条直线**w~1~x~1~+w~2~x~2~+b=0**，将平面划分成两块，使得
    w~1~x~1~+w~2~x~2~+b\>0
    的区域是y=1类的点，w~1~x~1~+w~2~x~2~+b\<0的区域是y=-1类别的点。

-   更高维度以此类推，由于更高维度的的超平面要写成
    w~1~x~1~+w~2~x~2~+\...w~p~x~p~=0
    ，会有点麻烦，一般会用矩阵表达式代替，即上面的W^T^X+b=0。

#### 软边界（Soft Margin）

当样例线性不可分，但近似线性可分的时候，可以通过软边界来实现一个近似的线性分类其。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image86.jpeg){width="5.136111111111111in"
height="2.703472222222222in"}

#### 高维映射（Feature Expansion）

在分类问题中，有时候数据集是线性不可分的，此时除了通过**Soft
Margin**方式之外，还有一种方法可以对数据做分类，即**高维映射（Feature
Expansion）**。

比如下图一维数据集中的白色点和黑色点线性不可分，即无法找到一个n-1维，也就是0维的点将它们分开：

![tmp](/home/jimzeus/outputs/AANN/images/media/image87.png){width="2.9166666666666665in"
height="0.3125in"}

但如果将通过函数**φ(x) = \[x,
x^2^\]^T^**将一维的数据映射到二维平面中，则此时数据集线性可分：

![图表1](/home/jimzeus/outputs/AANN/images/media/image88.png){width="2.8625in"
height="2.729861111111111in"}

事实上，任何样本集，如果用高维映射转换到无限维上都可以线性分开。

**参考**

高维映射与核方法

[[https://zhuanlan.zhihu.com/p/45223109]{.underline}](https://zhuanlan.zhihu.com/p/45223109)

#### 核函数

**核函数**（**Kernel function**，或者叫**核技巧**，**Kernel
trick**）在多种机器学习方法中都有用到。

在高维映射中，映射函数φ往往不是那么容易求出来，因此有核函数K，对于高维空间中任意两点，有：

**K(X1, X2) = φ(X~1~)^T^φ(X~2~)**

即核函数是二元函数，输入是映射之前的两个向量，其输出等价于两个向量映射之后的内积。

**参考**

核函数粗浅的理解

[[https://zhuanlan.zhihu.com/p/47541349]{.underline}](https://zhuanlan.zhihu.com/p/47541349)

用尽洪荒之力把核函数与核技巧讲得明明白白（精华篇）

[[https://zhuanlan.zhihu.com/p/136106284]{.underline}](https://zhuanlan.zhihu.com/p/136106284)

### 邻近算法（kNN）

**邻近算法，K最近邻，kNN，k Nearest
Neighbor，最近邻算法**，是一种分类算法。是数据挖掘分类技术中最简单的方法之一。所谓K最近邻，就是k个最近的邻居的意思，说的是每个样本都可以用它最接近的k个邻居来代表。

其步骤是：

1.  计算测试数据与各个训练数据之间的距离

2.  按照距离的递增关系进行排序；

3.  选取距离最小的K个点；

4.  确定前K个点所在类别的出现频率；

5.  返回前K个点中出现频率最高的类别作为测试数据的预测分类

**K值**（**临近数**，即在预测目标点时取几个临近的点来预测）的选取非常重要，因为：

1.  如果当K的取值过小时，一旦有噪声得成分存在们将会对预测产生比较大影响，例如取K值为1时，一旦最近的一个点是噪声，那么就会出现偏差，K值的减小就意味着整体模型变得复杂，容易发生过拟合；

2.  如果K的值取的过大时，就相当于用较大邻域中的训练实例进行预测，学习的近似误差会增大。这时与输入目标点较远实例也会对预测起作用，使预测发生错误。K值的增大就意味着整体的模型变得简单；

3.  如果K==N的时候，那么就是取全部的实例，即为取实例中某分类下最多的点，就对预测没有什么实际的意义了；

-   K的取值尽量要取奇数，以保证在计算结果最后会产生一个较多的类别，如果取偶数可能会产生相等的情况，不利于预测

-   无论是分类还是回归，衡量邻居的权重都非常有用，使较近邻居的权重比较远邻居的权重大。例如，一种常见的加权方案是给每个邻居权重赋值为1/
    d，其中d是到邻居的距离。

-   邻居都取自一组已经正确分类（在回归的情况下，指属性值正确）的对象。虽然没要求明确的训练步骤，但这也可以当作是此算法的一个训练样本集。

-   k-近邻算法的缺点是对数据的局部结构非常敏感。

    **混淆注意！**

    kNN算法与K-平均算法（k-means）没有关系，请勿与之混淆。

    **参考**

    用人话讲明白近邻算法KNN

    [[https://zhuanlan.zhihu.com/p/79531731]{.underline}](https://zhuanlan.zhihu.com/p/79531731)

    **sklearn示例**

    \>\>\> from sklearn.neighbors import KNeighborsClassifier

    \>\>\> knn = KNeighborsClassifier()

    \>\>\> knn.fit(X\_train, y\_train)

    KNeighborsClassifier(algorithm=\'auto\', leaf\_size=30,
    metric=\'minkowski\',

    metric\_params=None, n\_jobs=1, n\_neighbors=5, p=2,

    weights=\'uniform\')

    \>\>\> knn.predict(X\_test)

### k-means

**k-means
clustering**，**k-means聚类**，**k均值聚类**，**k平均算法**，一种迭代求解的聚类分析算法。

其步骤是：

-   将数据分为K组，随机选取K个对象作为初始的**聚类中心**。

-   然后计算每个对象与各个种子聚类中心之间的距离，把每个对象分配给距离它最近的聚类中心。聚类中心以及分配给它们的对象就代表一个聚类。

-   每分配一个样本，聚类中心会根据聚类中现有的对象被重新计算。这个过程将不断重复直到满足某个终止条件。

终止条件可以是：

-   没有（或最小数目）对象被重新分配给不同的聚类

-   没有（或最小数目）聚类中心再发生变化

-   误差平方和局部最小。

**参考**

用人话讲明白快速聚类kmeans

[[https://zhuanlan.zhihu.com/p/75477709]{.underline}](https://zhuanlan.zhihu.com/p/75477709)

### 高斯过程

Gaussian Process，

### 朴素贝叶斯方法

在机器学习中，**朴素贝叶斯分类器（Naïve Bayes
Classifier）**是一系列以假设特征之间强（朴素）独立下运用**贝叶斯定理**为基础的简单概率分类器。

**贝叶斯分类**是一类分类算法的总称，这类算法均以**贝叶斯定理**为基础，故统称为贝叶斯分类。而**朴素贝叶斯分类**是贝叶斯分类中最简单，也是常见的一种分类方法。

首先回忆下著名的**贝叶斯公式**：

**P(B\|A)=P(A\|B)·P(B)/P(A)**

如果我们把B替换为分类的类别，而A替换为特征，在P(B\|A)即表示在某种特征下的分类概率，即有：

**P(类别\|特征)=**
$\frac{\mathbf{P(}\mathbf{特征}\mathbf{|}\mathbf{类别}\mathbf{)}\mathbf{\cdot}\mathbf{P(}\mathbf{类别}\mathbf{)}}{\mathbf{P(}\mathbf{特征}\mathbf{)}}$

通常情况下特征值为多维，则有：

**P(类别\|特征1,特征2,\...特征n)=**
$\frac{\mathbf{P(}\mathbf{特征}\mathbf{1,}\mathbf{特征}\mathbf{2}\mathbf{,...,}\mathbf{特征}\mathbf{n|}\mathbf{类别}\mathbf{)}\mathbf{\cdot}\mathbf{P(}\mathbf{类别}\mathbf{)}}{\mathbf{P(}\mathbf{特征}\mathbf{1,}\mathbf{特征}\mathbf{2,...,}\mathbf{特征}\mathbf{n)}}$

而之所以称之为朴素，是因为朴素贝叶斯有一个基本的假设，就是条件间相互独立，由此可从上式得到：

**P(类别\|特征1,特征2,\...特征n) =**
$\frac{\mathbf{P(}\mathbf{特征}\mathbf{1|}\mathbf{类别}\mathbf{)}\mathbf{\cdot}\mathbf{P(}\mathbf{特征}\mathbf{2}\mathbf{|}\mathbf{类别}\mathbf{)}\mathbf{\cdot}\mathbf{\text{...}}\mathbf{\cdot}\mathbf{P(}\mathbf{特征}\mathbf{n|}\mathbf{类别}\mathbf{)}\mathbf{\cdot}\mathbf{P(}\mathbf{类别}\mathbf{)}}{\mathbf{P(}\mathbf{特征}\mathbf{1,}\mathbf{特征}\mathbf{2,...,}\mathbf{特征}\mathbf{n)}}$

由于当给定输入（即各个特征值，比如一副图片或者一段文字）时，P(特征1,特征2,\...,特征N)是个常量，因此可以使用下面的分类规则：

**P(类别\|特征1,特征2,\...,特征n）**$\mathbf{\propto}$
**P(类别)**$\prod_{\mathbf{i = 1}}^{\mathbf{n}}{\mathbf{P(}\mathbf{特征}\mathbf{i}\mathbf{\ }\mathbf{|}\mathbf{\ }\mathbf{类别}\mathbf{)}}$

我们可以使用最大后验概率来估计P(类别)和P(特征i\|类别），其中P(类别)是训练集中该类别的相对频率

而sklearn中各种朴素贝叶斯分类器的差别大部分来自于处理**P(特征i\|类别)**分布时所作的假设的不同。

尽管其假设过于简单，在很多实际情况下，朴素贝叶斯工作得很好，特别是文档分类和垃圾邮件过滤。这些工作都要求
一个小的训练集来估计必需参数。

相比于其他更复杂的方法，朴素贝叶斯学习器和分类器非常快。
分类条件分布的解耦意味着可以独立单独地把每个特征视为一维分布来估计。这样反过来有助于缓解维度灾难带来的问题。

另一方面，尽管朴素贝叶斯被认为是一种相当不错的分类器，但却不是好的估计器(estimator)，所以不能太过于重视从
predict\_proba 输出的概率。

**参考**

带你理解朴素贝叶斯分类算法

[[https://zhuanlan.zhihu.com/p/26262151]{.underline}](https://zhuanlan.zhihu.com/p/26262151)

一文详解朴素贝叶斯（Naive Bayes）原理

[[https://zhuanlan.zhihu.com/p/37575364]{.underline}](https://zhuanlan.zhihu.com/p/37575364)

sklearn用户指南：1.9朴素贝叶斯

[[https://sklearn.apachecn.org/docs/master/10.html]{.underline}](https://sklearn.apachecn.org/docs/master/10.html)

#### 高斯朴素贝叶斯（Gaussian Naive Bayes）

GaussianNB
实现了运用于分类的高斯朴素贝叶斯算法。特征的可能性P(x~i~\|y)（即上文中的P(特征i\|类别)）假设为高斯分布:

![](/home/jimzeus/outputs/AANN/images/media/image89.png){width="2.3583333333333334in"
height="0.4361111111111111in"}

参数σ~y~和μ~y~使用最大似然法估计。

\>\>\> from sklearn import datasets

\>\>\> iris = datasets.load\_iris()

\>\>\> from sklearn.naive\_bayes import GaussianNB

\>\>\> gnb = GaussianNB()

\>\>\> y\_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

\>\>\> print(\"Number of mislabeled points out of a total %d points :
%d\"

\... % (iris.data.shape\[0\],(iris.target != y\_pred).sum()))

Number of mislabeled points out of a total 150 points : 6

**参考**

[[https://scikit-learn.org/stable/modules/naive\_bayes.html\#gaussian-naive-bayes]{.underline}](https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes)

#### 多项式朴素贝叶斯（Multinomial Naive Bayes）

MultinomialNB
实现了服从多项分布数据的朴素贝叶斯算法，也是用于文本分类(这个领域中数据往往以词向量表示，尽管在实践中
tf-idf 向量在预测时表现良好)的两大经典朴素贝叶斯算法之一。

**参考**

[[https://scikit-learn.org/stable/modules/naive\_bayes.html\#multinomial-naive-bayes]{.underline}](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes)

#### 补充朴素贝叶斯（Complement Naive Bayes）

ComplementNB实现了补充朴素贝叶斯(CNB)算法。CNB是标准多项式朴素贝叶斯(MNB)算法的一种改进，特别适用于不平衡数据集。具体来说，CNB使用来自每个类的补数的统计数据来计算模型的权重。CNB的发明者的研究表明，CNB的参数估计比MNB的参数估计更稳定。此外，CNB在文本分类任务上通常比MNB表现得更好(通常有相当大的优势)。

**参考**

[[英文用户指南]{.underline}](https://scikit-learn.org/stable/modules/naive_bayes.html#complement-naive-bayes)

[[官方API文档]{.underline}](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB)

#### 伯努利朴素贝叶斯（Bernoulli Naive Bayes）

BernoulliNB
实现了用于多重伯努利分布数据的朴素贝叶斯训练和分类算法，即有多个特征，但每个特征
都假设是一个二元 (Bernoulli, boolean) 变量。
因此，这类算法要求样本以二元值特征向量表示；如果样本含有其他类型的数据，
一个 BernoulliNB 实例会将其二值化(取决于 binarize 参数)。

**参考**

[[英文用户指南]{.underline}](https://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes)

[[官方API]{.underline}](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB)

### 决策树

机器学习中，**决策树（Decision
tree）**是一个预测模型；他代表的是对象属性与对象值之间的一种映射关系。树中每个节点表示某个对象，而每个分叉路径则代表某个可能的属性值，而每个叶节点则对应从根节点到该叶节点所经历的路径所表示的对象的值。决策树仅有单一输出，若欲有复数输出，可以建立独立的决策树以处理不同输出。 数据挖掘中决策树是一种经常要用到的技术，可以用于分析数据，同样也可以用来作预测。

从数据产生决策树的机器学习技术叫做决策树学习,通俗说就是决策树。一个决策树包含三种类型的节点：

-   决策节点：通常用矩形框来表示

-   机会节点：通常用圆圈来表示

-   终结点：通常用三角形来表

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image90.png){width="5.114583333333333in"
height="1.46875in"}

**参考**

[[维基百科]{.underline}](https://zh.wikipedia.org/wiki/%E5%86%B3%E7%AD%96%E6%A0%91)

【机器学习】决策树（上）------ID3、C4.5、CART（非常详细）

[[https://zhuanlan.zhihu.com/p/85731206]{.underline}](https://zhuanlan.zhihu.com/p/85731206)

传统图像视频处理（opencv）
==========================

本章以OpenCV的tutorial的结构为顺序，介绍一些传统图像视频处理的概念。

官网：

[[https://opencv.org/]{.underline}](https://opencv.org/)

英文python教程：

[[https://docs.opencv.org/master/d6/d00/tutorial\_py\_root.html]{.underline}](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

中文python教程：

[[http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/tutorials.html]{.underline}](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/tutorials.html)

**OpenCV Python：**

-   imread()：从文件中读取图像（返回numpy.ndarray格式）

-   imwrite()：将图像写入文件

-   imshow()：显示图像

-   VideoCapture：视频流，读取自文件或者摄像头（参数为数字时）

-   read()：读取一帧图像（返回numpy.ndarray格式）

-   release()：释放该视频流

-   get()：获取视频流的相关信息（高、宽、帧率等）

-   VideoWriter：写入视频流

-   write()：写入一帧图像

-   release()：释放img

-   copyTo()：掩码

基本绘图
--------

**OpenCV Python：**

-   line()：画线

-   circle()：画圆

    -   img：输入图像

    -   center：圆心坐标（x,y）

    -   radius：半径

    -   color：颜色（R,G,B）

    -   thickness：圆环的粗细，-1表示充满整个圆

-   rectangle()：绘制矩形

    -   img：输入图像

    -   pt1：第一个顶点

    -   pt2：第二个顶点（第一个的对角）

    -   color：颜色

    -   thickness：轮廓线的粗细

色彩空间（Colorspace）
----------------------

**色彩空间**是对色彩的组织方式，**RGB格式**的色彩空间很简单，分别用三个分量表示红色R、绿色G、蓝色B的大小。

相较于**RGB**格式的色彩空间，**HSV色彩空间**更加贴合人类的感受。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image91.jpeg){width="4.830555555555556in"
height="3.6347222222222224in"}

H：**色相（Hue）**，即平时所说的颜色，黄色、红色等

S：**饱和度（Saturation）**，指色彩的纯度，越高色彩越纯，低则逐渐变灰，取0-100%

V：**明度（Value）**，即亮度

**RGB**到**HSV**的转换公式如下，假设max为R,G,B中最大者，min为最小者：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image92.png){width="2.7631944444444443in"
height="0.9569444444444445in"}

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image93.png){width="2.2493055555555554in"
height="0.3375in"}

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image94.png){width="0.5625in"
height="8.333333333333333e-2in"}

**OpenCV Python：**

-   split()：参数为h\*w\*c的图像，返回三路颜色通道，长宽为h\*w

-   merge()：合并N路颜色通道

-   cvtColor()：颜色空间的转换，比如BGR2GRAY

    -   src：输入图像

    -   code：转换方式，型为COLOR\_xxx2yyy，格式可以为BGR、BGRA、BGR565、RGB、RGBA、GRAY、HSV、XYZ等等，例如：

        -   cv2.COLOR\_BGR2BGRA：BGR通道格式加上Alpha通道

        -   cv2.COLOR\_BGR2GRAY：置为灰色

        -   cv2.COLOR\_BGR2HSV：转为HSV色彩空间

    -   返回：输出图像

-   inRange()：检查每个像素是否在某个范围之内

    -   src：输入图像

    -   lowerb：像素取值下限

    -   upperb：像素取值上限

    -   返回：同形灰度矩阵（8位单通道），对应像素在范围内则为255（白），否则为0（黑）

colorsys包中有hsv\_to\_rgb()等函数可直接进行RGB和HSV三元组之间的函数转换

几何变换（Geometry Transformation）
-----------------------------------

参考《[[概念定义 \> 线性代数/几何学 \> 齐次坐标]{.underline}](\l)》

变换模型是指根据待匹配图像和背景图象之间几何畸变的情况，所选择的能最佳拟合两幅图像之间变化的几何变换模型，可采用的变换模型有**刚性变换**、**仿射变换**、**透视变换**等等。

这些变换由一些基础的线性变换构成，包括：

-   平移（Translation)

-   旋转（Rotation）

-   镜像（Reflect）

-   剪切（Shear）

-   缩放（Scale）

-   Elation

    各种基础变换的**变换矩阵**和效果图示如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image95.png){width="5.282638888888889in"
height="5.738888888888889in"}

### 刚体变换（Rigid Transformation）

如果一幅图像中任意两点的距离经变换后仍保持不变，则这种变换称为**刚体变换**（**刚性变换**），刚体变换通常包括：

-   平移（Translation)

-   旋转（Rotation）

-   镜像（Reflect）

在二维空间中，点(x,y)经过**刚体变换**到点(x',y')的变换矩阵为：

$\begin{bmatrix}
x' \\
y' \\
1 \\
\end{bmatrix}$=$\begin{bmatrix}
 \pm \cos\varphi & \pm \sin\varphi & t_{x} \\
 \pm \sin\varphi & \pm \cos\varphi & t_{y} \\
0 & 0 & 1 \\
\end{bmatrix}\begin{bmatrix}
x \\
y \\
1 \\
\end{bmatrix}$

上式中φ表示旋转的角度，\[t~x~,t~y~\]^T^表示平移变量，参考之前各种基础变换的矩阵，可以看出这个矩阵是由平移、旋转、镜像三个变换的矩阵构成

### 仿射变换（Affine Transformation）

仿射变换是对一个向量空间进行一次线性变换，并接上一个平移，变换为另一个向量空间。

**仿射变换**保持了：

-   点的**共线性**：在同一直线上的三个或者更多的点在变换后仍然在同一直线上

-   线的**平行性**：parallelness，在原图中平行的线变换后仍然平行

-   平行线段的长度的比例

二维空间中，点(x,y)经过**仿射变换**到点(x',y')的变换矩阵为：

$\begin{bmatrix}
x' \\
y' \\
1 \\
\end{bmatrix}$=$\begin{bmatrix}
a_{1} & a_{2} & t_{x} \\
a_{3} & a_{4} & t_{y} \\
0 & 0 & 1 \\
\end{bmatrix}\begin{bmatrix}
x \\
y \\
1 \\
\end{bmatrix}$

其中$\begin{bmatrix}
a_{1} & a_{2} \\
a_{3} & a_{4} \\
\end{bmatrix}$为线性变换的矩阵，\[t~x~,t~y~\]^T^为平移变量。

由于线性变换的矩阵可以为任意值，因此**仿射变换**涵盖了：

-   平移（Translation)

-   旋转（Rotation）

-   镜像（Reflect）

-   缩放（Scale）

-   剪切（Shear）

### 投影变换（Projective Transformation）

首先来了解一下**3D投影**。

#### 投影

**投影（Projection）**指的是将一个3维物体显示在一个2维的平面（被称为**视平面**，**Viewing
Plane**，或者叫**投影平面**，**Projection
Plane**）上，比如设计中用到的三视图，即是一种投影。投影大致分为**平行投影**和**透视投影**两大类，详细分类如下：

-   **平行投影**（**Parallel
    Projection**）：投影中心和投影平面的距离是无限的，投影线相互平行。因此在3维物体中平行的线在投影中依然平行

    -   **正投影**（**Orthographic Projection**）：投影线垂直于投影平面

        -   **多视图投影**（**Multiview
            > Projection**）：物体的3个坐标面分别与投影面平行，形成三视图：正视图、侧视图、俯视图

        -   **轴测投影**（**Axonometric
            > Projection**）：物体的坐标面与投影面都不平行

    -   **斜投影**（**Oblique Projection**）：投影线不垂直于投影平面

-   **透视投影**（**Perspective
    Projection**）：投影中心和投影平面的距离是**有限的**（此时投影线汇于一点，即人眼或镜头）。因此在3维物体中平行的线在投影中可能不平行。

    -   **一点透视**（1**-point
        > Perspective**）：有1个灭点，物体有两个维度的棱平行于视平面，剩下1个维度的棱消失于唯一的灭点

    -   **两点透视**（2**-point
        > Perspective**）：有2个灭点，物体有一个维度的棱平行于视平面，剩下2个维度的棱分别消失于2个灭点

    -   **三点透视**（3**-point
        > Perspective**）：有3个灭点，物体三个维度的棱均不平行于视平面，而消失于3个灭点

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image96.png){width="3.7118055555555554in"
height="4.241666666666666in"}

**投影变换**本质上是**射影几何**的**齐次坐标系**的3维向量，经过一个3x3的矩阵变换为另一个3维向量，其变换矩阵为：

$\begin{bmatrix}
u \\
v \\
w \\
\end{bmatrix}$=$\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{bmatrix}\begin{bmatrix}
u' \\
v' \\
w' \\
\end{bmatrix}$

=\> k~p2~$\begin{bmatrix}
x \\
y \\
1 \\
\end{bmatrix}$=$\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{bmatrix}k_{p1}\begin{bmatrix}
x' \\
y' \\
1 \\
\end{bmatrix}$

=\> $\begin{bmatrix}
x \\
y \\
1 \\
\end{bmatrix}$=$\frac{k_{p1}}{k_{p2}}\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{bmatrix}\begin{bmatrix}
x' \\
y' \\
1 \\
\end{bmatrix}$

因为齐次坐标系中的系数可以为任意值而不会改变意义，将矩阵整体乘以1/a~33~，可得：

=\> $\begin{bmatrix}
x \\
y \\
1 \\
\end{bmatrix}$=$k\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & 1 \\
\end{bmatrix}\begin{bmatrix}
x' \\
y' \\
1 \\
\end{bmatrix}$

这使得这个变换矩阵的**自由度**降为8个。

下图中**齐次坐标系**的两个点（即3维空间中的红蓝色直线）分别是变换前和变换后的点：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image97.png){width="2.5381944444444446in"
height="2.5381944444444446in"}

下图中**齐次坐标系**的的两条直线（即3维空间中的红蓝色平面）分别为变换前后的线：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image98.png){width="3.2993055555555557in"
height="3.2993055555555557in"}

图中的绿色箭头表示了变换的方向。

**投影变换**的3x3矩阵在某些特殊值的情况下，可以表示一些特殊的变换，比如之前提到的刚体变换和仿射变换：

#### **等距同构变换**

等距同构（Isometric）变换指的是变换前后的距离保持不变（比如**旋转**、**平移**），包括3个自由度，即旋转角度θ和在两个方向上的平移t~x~和t~y~，变换矩阵如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image99.png){width="2.1569444444444446in"
height="0.8430555555555556in"}

其中ε等于$\pm 1$。可以将等距同构变换矩阵拆解为4个部分：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image100.png){width="2.1326388888888888in"
height="0.7583333333333333in"}

#### **相似变换**

**相似（Similar）**变换包括**缩放**、**平移**、**旋转**，包括4个自由度：旋转角度θ、缩放系数s和在两个方向上的平移t~x~和t~y~，变换矩阵如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image101.png){width="3.222916666666667in"
height="1.2965277777777777in"}

其中s为缩放的比例，其他同等距同构矩阵，也可以被分为四个部分：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image102.png){width="2.4097222222222223in"
height="0.8854166666666666in"}

#### **仿射变换**

**仿射变换**的包括了**旋转、平移、缩放、剪切**，变换矩阵如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image103.png){width="2.5277777777777777in"
height="1.2375in"}

仿射变换包括6个自由度，对应于矩阵中的6个元素，如果将变换矩阵拆解为4个部分：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image104.png){width="2.375in"
height="0.9430555555555555in"}

这6个自由度的其中2个包含在上图中的t里（位移t~x~和t~y~），另外4个也可以用更加可视化的方式表现出来，通过将上图中的非奇异矩阵A拆解为如下几个矩阵的积：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image105.png){width="3.198611111111111in"
height="1.3583333333333334in"}

剩下4个自由度为：旋转角度θ和φ、两个轴上的缩放系数λ~1~和λ~2~。这6个自由度的具像化的表示为：

1.  将图像旋转φ度

2.  在x和y轴上分别缩放λ~1~和λ~2~倍

3.  将图像旋转回去（旋转-φ度）

4.  再旋转θ度

5.  做两个轴上分别平移t~x~和t~y~

#### **投影变换（2D平面上）**

完全的**投影变换**包括8个自由度，其变换矩阵如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image106.png){width="2.1770833333333335in"
height="0.9923611111111111in"}

这里v=1或者0，拆解成4个部分则为：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image107.png){width="1.8854166666666667in"
height="0.7291666666666666in"}

可以将整个投影变换矩阵表示为四个矩阵之积：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image108.png){width="2.8555555555555556in"
height="1.6458333333333333in"}

矩阵链中的每个矩阵也可以拆为四个部分：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image109.png){width="2.917361111111111in"
height="0.9006944444444445in"}

1.  最左的矩阵为一个相似变换的矩阵，有4个自由度

2.  第二个矩阵其实是一个剪切变换矩阵，有1个自由度

3.  第三个矩阵是一个1个自由度的缩放矩阵，分别在x和y轴上放大了λ和1/λ倍

4.  最右边的矩阵是一个全新的变换elation，有2个自由度

#### **3D空间的理解**

以上是从在2D平面做变换的角度去理解投影变换。但既然**投影变换**包括投影（Perspective），自然跟投影也有关系。

通常**投影变换**都是指**透视变换（Perspective
Transformation）**，**透视变换**是**透视投影变换**的缩写，和在同一个平面的变换（**刚体变换**、**仿射变换**及各种基础变换）不同，**透视变换**是将物体/点（在**视平面π**的成像）投影到一个新的**视平面π'**。然后根据这个平面得到物体/点在其中的2D坐标。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image110.png){width="4.43125in"
height="2.5833333333333335in"}

**投影变换**是一个"群"（Group，数学概念），换句话说，任何一个投影变换的逆变换也是投影变换，任意两个投影变换的组合也是一个投影变换。

举个很常用的例子，从世界平面投影到一个camera1的成像平面是一个**投影变换**，到camera2也是。下图是世界平面到2个camera的投影变换：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image111.png){width="4.670833333333333in"
height="2.6555555555555554in"}

考虑到刚才所说，从camera1的成像平面，投影回世界平面（逆变换），再投影到camera2的成像平面（组合变换），这个变换组合可以被视作一个单独的投影变换，由于3D不好表示，用2D表示如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image112.png){width="4.9in"
height="2.1006944444444446in"}

这种组合变换代表的意义即是对一个平面的不同角度的摄像画面。

**参考**

[[https://mc.ai/part-ii-projective-transformations-in-2d/]{.underline}](https://mc.ai/part-ii-projective-transformations-in-2d/)

**混淆注意！**

**透视变换**和一个3D物体分别投影在两个不同的**视平面**是有区别的，当物体投影在第一个**视平面**的时候，其位置信息已经丢失了一部分（3D变为2D），因此不可能无损的将一个物体在视平面A的投影转换为其在视平面B的投影。

**透视变换**和**透视投影**这两个名词经常混淆，文中定义的**透视变换**有时候也会用"**透视投影**"这个词来表示，那么根据在这里的定义，这两者有什么区别和联系呢？

-   区别：**透视投影**是3D物体到2D**视平面**的投影，而透视变换中不涉及3D**物体**

-   联系：**透视变换**可以被视作是一个2D物体（视平面A）到2D视平面的**透视投影**

**透视变换**和**仿射变换**的区别和联系如下：

-   区别：**透视变换**中有2个视平面，是物体从一个视平面到另一个视平面的投射

-   联系：**仿射变换**可以被视为是透视变换的一个特例，当透视变换矩阵中的部分元素取特定值时，即变成仿射变换。

### OpenCV Python

OpenCV Python中和几何变换相关的各种函数

-   resize()：缩放图片

    -   src：输入图片

    -   dsize：目标大小，格式为(width, height)

    -   dst：可选参数，输出图片

    -   fx：可选参数，水平方向放大倍数，dsize需设为None

    -   fy：可选参数，垂直方向放大倍数，dsize需设为None

    -   interpolation：可选参数，插值方式

        -   INTER\_LINEAR：bilinear插值

        -   INTER\_CUBIC：bicubic插值

        -   \...

    -   返回：输出图片

-   getRotationMatrix2D()：生成旋转变换矩阵

    -   center：旋转中心的坐标，格式为(x,y)，单位为像素

    -   angle：旋转的角度，单位为度

    -   scale：缩放比例

-   getAffineTransform()：生成仿射变换所需矩阵

    -   src：原图的三个点，(3,2)的ndarray，dtype为float32

    -   dst：目标图像的三个对应的点，(3,2)的ndarray，dtype为float32

    -   返回：生成的仿射变换矩阵，(2,3)的ndarray

-   warpAffine()：根据变换矩阵，对图像进行仿射变换

    -   src：输入图片

    -   M：仿射变换矩阵，(2,3)的ndarray，可由getAffineTransform生成，也可自行定义（参考几何变换）

    -   dsize：输出图片的尺寸

    -   返回：输出图片

-   getPerspectiveTransform()：生成透视变换所需矩阵

    -   src：原图的四个点，(4,2)的ndarray，dtype为float32

    -   dst：目标图像的四个对应的点，(4,2)的ndarray，dtype为float32

    -   返回：生成的透视变换矩阵，(3,3)的ndarray

-   warpPerspective()：根据变换矩阵，对图像进行透视变换

    -   src：输入图片

    -   M：透视变换矩阵，(3,3)的ndarray

    -   dsize：输出图片的尺寸

    -   返回：输出图片

阈值处理（Threshold）
---------------------

### 大津算法（Otsu's Method）

在计算机视觉和图像处理中，**大津二值化法**用来自动对基于聚类的图像进行**二值化**，或者说，将一个灰度图像退化为二值图像。该算法以大津展之命名。

算法假定该图像根据**双模直方图**（前景像素和背景像素）把包含两类像素，于是它要计算能将两类分开的最佳阈值，使得它们的类内方差最小；由于两两平方距离恒定，所以即它们的类间方差最大。

直观的来说，就是找到图像的**直方图**中如下红线位置：

![u=4036814783,202893756&fm=26&gp=0](/home/jimzeus/outputs/AANN/images/media/image113.jpeg){width="2.4930555555555554in"
height="1.8909722222222223in"}

**OpenCV Python：**

-   threshold()：阈值处理

    -   src：输入图像，可以为单通道或者多通道（对每个通道分别做阈值处理）

    -   thresh：阈值

    -   maxval：THRESH\_BINARY和THRESH\_BINARY\_INV用到

    -   type：类型，如下

        -   cv2.THRESH\_BINARY：大于阈值则为maxval，否则为0

        -   cv2.THRESH\_BINARY\_INV：大于阈值则为0，否则为maxval

        -   cv2.THRESH\_TRUNC：大于阈值则为thresh，否则不变

        -   cv2.THRESH\_TOZERO：大于阈值不变，否则为0

        -   cv2.THRESH\_TOZERO\_INV：大于阈值为0，否则不变

        -   cv2.THRESH\_OTSU：大津算法，参考《[[图像视频处理 \>
            阈值处理 \> 大津算法]{.underline}](\l)》

-   adaptiveThreshold()：自适应二值化处理

    -   src：输入图像

    -   maxValue：最大值，即1的情况下填的值

    -   adaptiveMethod：自适应阈值处理方法

        -   cv2.ADAPTIVE\_THRESH\_MEAN\_C：阈值为该像素周围区域（长宽都为blockSize）的平均值减去常量C

        -   cv2.ADAPTIVE\_THRESH\_GAUSSIAN\_C：阈值为该像素周围区域（长宽都为blockSize）的加权和（即高斯模糊）减去常量C

    -   thresholdType：

        -   cv2.THRESH\_BINARY：大于阈值时为maxValue，否则为0

        -   cv2.THRESH\_BINARY\_INV：小于阈值时为maxValue，否则为0

    -   blocksize：区域的长和宽

    -   C：常量C

过滤器-模糊
-----------

类似1维信号中的HPF（高通滤波器），或者LPF（低通滤波器），图像中也有类似的处理，可用于过滤噪声，平滑图像。被称为**图像模糊（Image
Blurring）**，或者**图像平滑（Image Smoothing**）。

### 2D卷积（2D Convolution）

根据卷积核对图像做卷积，是下面的**均值模糊**等操作的一般情况。

**OpenCV Python：**

-   filter2D()：对图像做卷积运算（模糊）

    -   src：输入图像

    -   ddepth：图像位数，-1表示跟输入一致

    -   kernel：卷积核

### 均值模糊（Averaging Blur）

取每个点周围区域的平均值作为该点的值。例如一个3x3的均值模糊卷积核为：

![](/home/jimzeus/outputs/AANN/images/media/image114.png){width="1.5104166666666667in"
height="0.7645833333333333in"}

**OpenCV Python：**

-   blur()：2D卷积的特殊情况，卷积核固定为ksize指定的均值矩阵

    -   src：输入图像

    -   ksize：卷积核尺寸，例如（5,5）

    -   ddepth：图像位数，-1表示跟输入一致

### 中值模糊（Median Blur）

也叫**中值滤波**（**Median
Filter**），用于去除图像或者信号中噪声的技术，具有保存边缘的特点。

对于图像来说，其算法就是在像素取值为其周围区域（被称为窗口/window）的中值（注意不是均值）。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image115.png){width="3.473611111111111in"
height="1.8118055555555554in"}

**OpenCV Python：**

-   medianBlur()：中值模糊

    -   src：输入图像

    -   ksize：卷积核尺寸，例如（5,5）

**参考**

[[https://en.wikipedia.org/wiki/Median\_filter]{.underline}](https://en.wikipedia.org/wiki/Median_filter)

### 高斯模糊（Gaussian Blur）

也叫**高斯平滑（Gaussian Smoothing）**或者**高斯滤波（Gaussian
Filter）**，图像处理手段，用于减少图像噪声，及降低细节。

其算法就是图像与**正态分布**做卷积，由于正态分布又叫**高斯分布**，因此被叫做高斯模糊。

**OpenCV Python：**

-   GaussianBlur()：高斯模糊

    -   src：输入图像，每个通道独立卷积

    -   ksize：卷积核尺寸，例如（5,5）

    -   sigmaX：标准差，为0表示由ksize计算得来

**参考**

[[https://zh.wikipedia.org/wiki/%E9%AB%98%E6%96%AF%E6%A8%A1%E7%B3%8A]{.underline}](https://zh.wikipedia.org/wiki/%E9%AB%98%E6%96%AF%E6%A8%A1%E7%B3%8A)

### 双边滤波（Bilateral Filter）

在图像处理上，**双边滤波器**为使影像平滑化的非线性滤波器。

图像去噪的方法很多，如**中值滤波，高斯滤波，维纳滤波**等等。但这些降噪方法容易模糊图片的边缘细节，对于高频细节的保护效果并不明显。相比较而言，双边滤波器可以很好的边缘保护，即可以在去噪的同时，保护图像的边缘特性。双边滤波（Bilateral
filter）是一种非线性的滤波方法，是结合图像的**空间邻近度**和**像素值相似度**的一种折衷处理，同时考虑空域信息和灰度相似性，达到保边去噪的目的。

双边滤波器之所以能够做到在平滑去噪的同时还能够很好的保存边缘（Edge
Preserve），是由于其滤波器的核由两个函数生成：**空间域核**和**值域核**

-   空间域核w~d~衡量的是p, q两点之间的距离，距离越远权重越低。

![BF](/home/jimzeus/outputs/AANN/images/media/image116.png){width="2.276388888888889in"
height="0.5090277777777777in"}

-   值域核w~r~衡量的是p, q两点之间的像素值相似程度，越相似权重越大。

![BF
(复件)](/home/jimzeus/outputs/AANN/images/media/image117.png){width="2.3152777777777778in"
height="0.5118055555555555in"}

在平坦区域，临近像素的像素值的差值较小，对应值域权重接近于1，此时空域权重起主要作用，相当于直接对此区域进行高斯模糊。因此，平坦区域相当于进行高斯模糊。

在边缘区域，临近像素的像素值的差值较大，对应值域权重接近于0，导致此处核函数下降（因），当前像素受到的影响就越小，从而保持了原始图像的边缘的细节信息。

最终的权值则为w~d~和w~r~之积：

![BF
(另一个复件)](/home/jimzeus/outputs/AANN/images/media/image118.png){width="4.729166666666667in"
height="0.47708333333333336in"}

**OpenCV Python：**

-   bilateralFilter()：双边滤波

    -   src：输入图像

    -   d：核的直径

    -   sigmaColor：值域核的标准差

    -   sigmaSpace：空间域核的标准差

**参考**

[[https://blog.csdn.net/guyuealian/java/article/details/82660826]{.underline}](https://blog.csdn.net/guyuealian/java/article/details/82660826)

[[https://en.wikipedia.org/wiki/Bilateral\_filter]{.underline}](https://en.wikipedia.org/wiki/Bilateral_filter)

形态变换（Morphological Transformation）
----------------------------------------

以下三图分别为**原图、侵蚀（Erosion）、膨胀（Dilation）**

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image119.png){width="0.7819444444444444in"
height="1.0472222222222223in"}
![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image120.png){width="0.7833333333333333in"
height="1.0493055555555555in"}
![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image121.png){width="0.7854166666666667in"
height="1.0527777777777778in"}

下图为**Opening**，是一个**侵蚀**后跟一个**膨胀**，可用于消除小的噪点

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image122.png){width="1.5381944444444444in"
height="1.0298611111111111in"}

下图为**Closing**，是一个**膨胀**后跟一个**侵蚀**，可用于消除小的孔洞

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image123.png){width="1.5763888888888888in"
height="1.0555555555555556in"}

下图为**Morphological
Gradient**，是**膨胀**和**侵蚀**的差，可形成物体的轮廓

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image124.png){width="1.59375in"
height="1.0673611111111112in"}

**OpenCV Python：**

-   erode()：缩减，边缘侵蚀

    -   src：输入图像

    -   kernel：核，比如np.ones((5,5), np.uint8)

-   dilate()：扩张，膨胀

    -   src：输入图像

    -   kernel：核，比如np.ones((5,5), np.uint8)

-   morphologyEx()：基于erode和dilate的高级形态变换

    -   src：输入图像

    -   op：变换类型

        -   cv2.MORPH\_ERODE：同erode()

        -   cv2.MORPH\_DILATE：同dilate()

        -   cv2.MORPH\_OPEN：先erode再dilate，可以用于消除背景中的噪点

        -   cv2.MORPH\_CLOSE：先dilate再erode，可以用于填充物体中的小洞

        -   cv2.MORPH\_GRADIENT：dilate()和erode()之差，输出结果物体的轮廓

        -   cv2.MORPH\_TOPHAT：输入图像和其Opening图像之差

        -   cv2.MORPH\_BLACKHAT：输入图像和其closing图像之差

    -   kernel：核，比如np.ones((5,5), np.uint8)。

图像导数（Image Gradient）
--------------------------

**图像导数**指的是将像素位置视为x，像素取值视为y，得到的函数的导数。其中**一阶导数**通常用于**图像边缘检测**，**二阶导数**的符号可以确定边缘的过渡是从亮到暗还是从暗到亮，在对图像做导数之前最好先进行平滑处理，因为导数操作对噪声敏感。

### 索伯算子（Sobel Operator）

图像处理中的算子之一，在机器视觉领域常被用于做边缘检测。也叫**索伯边缘检测**、**索伯变换**。

**索伯算子**是2组n×n（通常为3×3）的矩阵，分别为横向边缘检测和纵向边缘检测的卷积核，用该两组卷积核和图像做平面卷积，即可得到横向及纵向的边缘检测结果。

如果A为原始图像，Gx为经横向边缘检测后的图像，Gy为纵向边缘检测的图像，矩阵大小为3×3，则有：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image125.png){width="3.50625in"
height="0.5090277777777777in"}

索伯变换的效果如下图：

![gradients](/home/jimzeus/outputs/AANN/images/media/image126.jpeg){width="2.785416666666667in"
height="3.0145833333333334in"}

**参考**

[[https://en.wikipedia.org/wiki/Sobel\_operator]{.underline}](https://en.wikipedia.org/wiki/Sobel_operator)

[[https://docs.opencv.org/master/d5/d0f/tutorial\_py\_gradients.html]{.underline}](https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html)

**OpenCV Python：**

-   Sobel()：索伯变换，用于检测横向或纵向的边缘

    -   src：输入图像

    -   ddepth：输出图像深度

    -   dx：x轴方向的标准差

    -   dy：y轴方向的标准差

    -   ksize：核的尺寸，缺省为3

### Scharr算子（Scharr Operator）

**Scharr算子**是索伯算子的优化和变形，在图像处理中通常用以下3×3卷积核表示：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image127.png){width="2.9611111111111112in"
height="0.65in"}

以下三幅图分别为原图、索伯变换、Scharr变换：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image128.jpeg){width="2.4784722222222224in"
height="1.8590277777777777in"}　![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image130.jpeg){width="2.535416666666667in"
height="1.8972222222222221in"}

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image131.png){width="2.7465277777777777in"
height="2.058333333333333in"}

**OpenCV Python：**

-   Scharr()：Scharr变换，索伯变换的变形

    -   src：输入图像

    -   ddepth：输出图像深度

    -   dx：x轴方向的标准差

    -   dy：y轴方向的标准差

### 拉普拉斯算子（Laplace Operator）

**拉普拉斯算子（Laplace
Operator，或Laplacian）**是n维欧几里德空间中的一个二阶微分算子。

在图像处理中来说，**拉普拉斯算子**经常被用来作为边缘检测的手段。图像中边缘处的像素值变化更大，其一阶导数必然更大，而一阶导数极值位置，二阶导数必然为0，下面三图分别为原函数图像、一阶导数、二阶导数：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image132.jpeg){width="1.8347222222222221in"
height="1.3333333333333333in"}![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image133.jpeg){width="1.8305555555555555in"
height="1.3493055555555555in"}![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image134.jpeg){width="1.8055555555555556in"
height="1.336111111111111in"}

数字图像处理中，拉普拉斯算子通常被表现为如下的3×3卷积核：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image135.png){width="1.3520833333333333in"
height="0.6083333333333333in"}

OpenCV中的拉普拉斯函数Laplacian()缺省采取以上矩阵，此外也采用以下扩展矩阵作为卷积核：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image136.png){width="1.4027777777777777in"
height="0.6013888888888889in"}

**参考**

[[https://blog.csdn.net/songzitea/article/details/12842825]{.underline}](https://blog.csdn.net/songzitea/article/details/12842825)

[[https://en.wikipedia.org/wiki/Discrete\_Laplace\_operator]{.underline}](https://en.wikipedia.org/wiki/Discrete_Laplace_operator)

**OpenCV Python：**

-   Laplacian()：拉普拉斯变换

    -   src：输入图像

<!-- -->

-   ddepth：输出图像深度

Canny边缘检测算法
-----------------

**Canny边缘检测算子（Canny Edge
Detector）**是澳洲计算机科学家约翰·坎尼（John F.
Canny）于1986年开发出来的一个多级边缘检测算法。更为重要的是Canny创立了"边缘检测计算理论"（computational
theory of edge detection）解释这项技术如何工作。

简单来讲，Canny将边缘检测需要达到的目标总结为如下三条准则：

1.  **好的检测**：检测算法应该精确地找到图像中的尽可能多的边缘，减少漏检和误检。

2.  **最优定位**：检测的边缘点应该精确地定位于边缘的中心。

3.  **最小响应**：图像中的任意边缘应该只被标记一次，同时图像噪声不应产生伪边缘。

为此，Canny采用了如下步骤：

1.  **使用高斯滤波器，以平滑图像，滤除噪声。**

2.  **计算图像中每个像素点的梯度强度和方向。**

采用Sobel算子、Prewitt算子等等来实现，OpenCV中用的是Sobel算子。

3.  **应用非极大值抑制（Non-Maximum
    Suppression），消除边缘检测带来的杂散响应。**

这个步骤的目的是为了上述的第二个准则**最优定位**，注意这个步骤是沿着梯度方向进行的，简单来说就是使得检测出来的边缘变窄并位于中心。

具体操作是：

1.  将当前像素的梯度强度与沿正负梯度方向上的两个像素进行比较。

2.  如果当前像素的梯度强度与另外两个像素相比最大，则该像素点保留为边缘点，否则该像素点将被抑制。

<!-- -->

4.  **应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。**

相较于简单的单阈值决定是否边缘，双阈值检测的逻辑如下：

3.  如果边缘像素梯度高于**高阈值**，则视为**强边缘**

4.  如果梯度低于**低阈值**，则视为**噪点**被抑制

5.  如果介于**高低阈值**之间，则被视为**弱边缘**

<!-- -->

5.  **通过抑制孤立的弱边缘最终完成边缘检测。**

检测每一个弱边缘像素，如果其相邻的8个像素中有一个是强边缘，则该边缘像素被视为强边缘保留。而所有没有连接强边缘的弱边缘则会被抑制。

**参考**

[[https://www.cnblogs.com/nowgood/p/cannyedge.html]{.underline}](https://www.cnblogs.com/nowgood/p/cannyedge.html)

[[https://en.wikipedia.org/wiki/Canny\_edge\_detector]{.underline}](https://en.wikipedia.org/wiki/Canny_edge_detector)

**OpenCV Python：**

-   Canny()：Canny边缘检测

    -   image：输入图像

    -   threshold1：低阈值

    -   threshold2：高阈值

### Douglas-Peucker算法（TODO）

图像金字塔（Image Pyramids）
----------------------------

**OpenCV Python：**

-   pyrDown()：向下采样，采用Gaussian Pyramid

-   pyrUp()：向上采样，采用Laplacian Pyramid

轮廓（Contour）
---------------

### 矩（moment）

图像的**矩**是指图像的某些特定像素灰度的加权平均值（矩），或者是图像具有类似功能或意义的属性。

图像矩通常用来描述图像中**分割后的对象**。可以通过图像的矩来获得图像的部分性质，包括**面积**(或**总体亮度**)，以及有关**几何中心**和**方向**的信息
。

矩分为**原始矩**（也叫**几何矩**）和**中心矩**，中心矩具有平移不变性。

#### 原始矩

对于二维连续函数 f (x, y), (p+q) 阶的原始矩被定义为

![moment](/home/jimzeus/outputs/AANN/images/media/image137.png){width="2.145138888888889in"
height="0.6145833333333334in"}

对于灰度图像的像素的强度 I(x,y), 图像的原始矩 Mij 被计算为：

![moment
(复件)](/home/jimzeus/outputs/AANN/images/media/image138.png){width="1.7208333333333334in"
height="0.44930555555555557in"}

原始矩包括以下一些有关图像性质的信息：

1.  **M~00~**为图像的**灰度质量**（如果是二值图则为**图像的面积**）

2.  图像的**几何中心**可以表示为![moment4](/home/jimzeus/outputs/AANN/images/media/image139.png){width="1.1444444444444444in"
    height="0.3625in"}

#### 中心矩

中心矩在平移时具有**平移不变性**，**中心矩**定义为：

![moment
(另一个复件)](/home/jimzeus/outputs/AANN/images/media/image140.png){width="2.4756944444444446in"
height="0.5319444444444444in"}

其中![moment4](/home/jimzeus/outputs/AANN/images/media/image141.png){width="0.44930555555555557in"
height="0.2326388888888889in"}为图像的**几何中心**，如果 ƒ(x, y)是一个数字图像，则前一公式等价于

![moment
(第3个复件)](/home/jimzeus/outputs/AANN/images/media/image142.png){width="2.28125in"
height="0.46319444444444446in"}

**参考**

[[https://www.cnblogs.com/ronny/p/3985810.html]{.underline}](https://www.cnblogs.com/ronny/p/3985810.html)

[[https://zh.wikipedia.org/wiki/%E7%9F%A9\_(%E5%9B%BE%E5%83%8F)]{.underline}](https://zh.wikipedia.org/wiki/%E7%9F%A9_(%E5%9B%BE%E5%83%8F))

**OpenCV Python：**

-   findContours()：寻找二值图像（8位单通道图像的非0像素均被视为1）中的轮廓

    -   image：输入图像，8位单通道

    -   mode：组织模式

        -   cv2.RETR\_EXTERNAL：仅寻找图像中物体的外部轮廓

        -   cv2.RETR\_LIST：寻找所有轮廓，但不确立他们的层次结构

        -   cv2.RETR\_CCOMP：寻找所有轮廓，并分为两级，上一级为外部轮廓，外部轮廓中的为内部轮廓（即物体中洞的轮廓），如果某个内部轮廓中又有轮廓，则又被视为外部轮廓

        -   cv2.RETR\_TREE：寻找所有轮廓，并根据他们的包含关系建立一个完整的层次结构

    -   method：轮廓近似模式

        -   cv2.CHAIN\_APPROX\_NONE：列出所有的点

        -   cv2.CHAIN\_APPROX\_SIMPLE：只列出线段（横线、竖线、斜线）端点

    -   返回：两个返回值

        -   contours：轮廓列表

        -   hierarchy：轮廓的层次结构，三阶的ndarray，第二个阶是各个轮廓，最后一阶的维度为4，表示每个轮廓的\[Next,
            > Previous, First\_Child, Parent\]

-   drawContours()：画出轮廓

    -   image：图像

    -   contours：轮廓列表

    -   contourIdx：需要画出的轮廓的索引，-1表示所有轮廓

    -   color：轮廓的颜色

    -   thickness：轮廓线的粗细，缺省为1，-1表示填满轮廓中的部分

-   moments()：获得图像的矩

-   contourArea()：计算轮廓面积

    -   contour：轮廓

-   arcLength()：计算弧线（轮廓）周长

    -   curve：一段弧线或者轮廓

    -   closed：布尔值，是否为闭环（比如轮廓）

-   approxPolyDP()：得到弧线（轮廓）的更少顶点的近似线段（多边形），利用Douglas-Peucker算法

    -   curve：输入的线段（轮廓）

    -   epsilon：新线段和原弧线之间允许的最大距离，用于表示精度

    -   closed：布尔值，是否为闭环（轮廓）

-   convexHull()：得到一系列点的外接凸多边形（有时和apporxPolyDP输出相同）

    -   points：一系列点

-   isContourConvex()：检查轮廓是否为凸多边形

    -   contour：轮廓

-   boundingRect()：得到弧线（轮廓）的straight bounding
    box（横平竖直那种）

    -   array：一段弧线或轮廓

    -   返回：bounding rect的x, y（左上角）, w, h

-   minAreaRect()：得到一系列点的最小面积bounding box（可以斜着）

    -   points：一系列点（可以是弧线、轮廓）

-   minEnclosingCircle()：得到一系列点的最小外接圆

    -   points：一系列点（可以是弧线、轮廓）

    -   返回：圆心和半径

-   fitEllipse()

-   fitLine()

-   convexityDefects()：

-   pointPolygonTest()：

-   matchShapes()：

直方图（Histogram）
-------------------

**方向梯度直方图**（**Histogram of Oriented
Gradient**），是应用在计算机视觉和图像处理领域，用于[目标检测](https://zh.wikipedia.org/w/index.php?title=%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B&action=edit&redlink=1)的特征描述器。这项技术是用来计算局部图像梯度的方向信息的统计值。

模板匹配（Matching Template）
-----------------------------

**模板匹配（Matching Template）**用于在一幅图像中寻找一个小的模板图像

**OpenCV Python：**

-   matchTemplate()：搜索目标图像，比对模板

    -   image：目标图像，大小为(W, H)

    -   templ：模板图案，大小为(w, h)

    -   method：搜索方法

        -   cv2.TM\_SQDIFF：目标和模板的平方差，匹配越好，匹配值越小

        -   cv2.TM\_SQDIFF\_NORMED：标准化的平方差（除以方-和-积-根）

        -   cv2.TM\_CCORR：互相关匹配，目标和模板的乘积之和，匹配越好，匹配值越大

        -   cv2.TM\_CCORR\_NORMED：标准化的互相关匹配

        -   cv2.TM\_CCOEFF："目标像素减平均值"和"模板像素减平均值"的乘积之和，匹配越好，匹配值越大

        -   cv2.TM\_CCOEFF\_NORMED：标准化的CCOEFF

    -   返回：匹配值矩阵，大小为（W-w+1,
        > H-h+1），每个元素为模板顶点（而非中心）在目标图像中的位置

-   minMaxLoc()：从矩阵中找到极大值、极小值以及他们的位置

    -   src：匹配值矩阵

    -   返回：最小值，最大值，最小值位置，最大值位置

霍夫变换（Hough Transform）
---------------------------

**霍夫变换**是一种特征提取，被广泛应用在图像处理和机器视觉。
霍夫变换是用来辨别找出物件中的特征。他的算法流程大致如下，给定一个物件、要辨别的形状的种类，算法会在[参数空间](https://zh.wikipedia.org/w/index.php?title=%E5%8F%83%E6%95%B8%E7%A9%BA%E9%96%93&action=edit&redlink=1)中执行投票来决定物体的形状，而这是由累加空间（accumulator
space）里的局部最大值来决定。

经典的霍夫变换是侦测图片中的直线，之后，霍夫变换不仅能识别直线，也能够识别任何形状，常见的有圆形、椭圆形。

**参考**

[[https://zh.wikipedia.org/wiki/%E9%9C%8D%E5%A4%AB%E5%8F%98%E6%8D%A2]{.underline}](https://zh.wikipedia.org/wiki/%E9%9C%8D%E5%A4%AB%E5%8F%98%E6%8D%A2)

[[http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/hough\_lines/hough\_lines.html\#hough-lines]{.underline}](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html#hough-lines)

**OpenCV Python：**

-   HoughLines()：标准霍夫线变换，检测直线

    -   返回：一组极径和极角的组合![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image143.png){width="0.28680555555555554in"
        > height="0.13958333333333334in"}

-   HoughLinesP()：统计概率霍夫变换，检测线段，执行效率更高

    -   返回：一组线段的端点(x0,y0,x1,y1)

-   HoughCircles()：霍夫圆变换，检测圆形

    -   返回：描述圆心和直径的三元组(x, y, r)

    -   image：输入图像（8位、单通道、灰度图像）

    -   method：检测方法，目前只实现了cv2.HOUGH\_GRADIENT

    -   dp：

    -   minDist：检测到的圆之间的最小距离，如果这个值太小，可能会重复检测一个圆（即一个圆被检测到多次），如果太大，可能会漏检。

    -   circles：同返回值

    -   param1：用到的**Canny边缘检测**的高阈值（低阈值为其一半）

    -   param2：cv2.HOUGH\_GRADIENT方法的累加器阈值，越小检测的圆越多

    -   minRadius：检测出的圆允许的最小半径

    -   maxRadius：检测出的圆允许的最大半径

Harris角检测器（TODO）
----------------------

SIFT算法（TODO）
----------------

SURF算法（TODO）
----------------

图像分割（Image Segmentation）
------------------------------

**OpenCV Python：**

-   watershed()：分水岭算法，用于分割图像

-   grabCut()：GrabCut算法

视频处理（Video Process）
-------------------------

### 均值漂移（mean-shift）

**均值飘移（mean-shift）**算法，在聚类、图像平滑、图像分割和跟踪方面得到了比较广泛的应用。

1.  在未被标记的数据点中随机选择一个点作为起始中心点center；

2.  找出以center为中心半径为radius的区域中出现的所有数据点。

3.  以center为中心点，计算从center开始到集合M中每个元素的向量，将这些向量相加，得到向量shift。

4.  center = center +
    shift。即center沿着shift的方向移动，移动距离是\|\|shift\|\|。

5.  重复步骤2、3、4，直到shift的很小（就是迭代到收敛），得到最终收敛的center。

### 光流

**光流**(**Optical flow，optic
flow**)是关于视域中的物体运动检测中的概念。用来描述相对于观察者的运动所造成的观测目标、表面或边缘的运动。光流法可用于运动检测、物件切割、碰撞时间与物体膨胀的计算、运动补偿编码，或者通过物体表面与边缘进行立体的测量等等。

相机标定（Camera Calibration）
------------------------------

相机标定用于纠正相机镜头导致的畸变。

### 光心、焦距、焦点

下图可使得**光心（optical center）**、**焦距 （focal
length）**和**焦点（focal point）**的概念一目了然：

![](/home/jimzeus/outputs/AANN/images/media/image144.png){width="2.6819444444444445in"
height="1.7368055555555555in"}

### 坐标系

图像处理和立体视觉常说到的坐标系有四个，即世界坐标系、相机坐标系、图像坐标系、像素坐标系。

#### **世界坐标系**

O~w~\~X~w~Y~w~Z~w~，独立于相机的坐标系，可以用于描述相机位置，单位m。世界坐标系中的点，即为现实世界中的点。构建世界坐标系只是为了更好的描述相机的位置。

在双目视觉（两个摄像头）中一般将世界坐标系原点定在左相机或者右相机，或者两者x轴方向的中点。

对于单目视觉，世界坐标系可以跟相机坐标系重合。

#### **相机坐标系**

O~c~\~X~c~Y~c~Z~c~，以光心为原点，单位m。从世界坐标系转换到相机坐标系为刚体变换，仅涉及到**旋转**和**平移**，物体或空间不会发生形变。

以下为旋转矩阵：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image145.png){width="5.597916666666666in"
height="3.9770833333333333in"}

假设平移向量为T，则点从世界坐标系O~w~到相机坐标系O~c~的变换为：

![](/home/jimzeus/outputs/AANN/images/media/image146.png){width="5.764583333333333in"
height="1.0625in"}

#### **图像坐标系**

图像坐标系是二维坐标系，单位是mm，从相机坐标系到图像坐标系，是透视投影关系。

以下是相机坐标系中点的位置P(X~c~,Y~c~,Z~c~)，到图像坐标系的位置p(x,y)的变换：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image147.png){width="5.0in"
height="3.0229166666666667in"}

#### **像素坐标系**

**像素坐标系**和**图像坐标系**在同一平面，且没有选转，只是各自的原点和度量单位不一样。图像坐标系的原点为相机光轴与成像平面的交点，通常情况下是成像平面的中点（principal
point）。图像坐标系的单位是mm，属于物理单位，而像素坐标系的单位是pixel，我们平常描述一个像素点都是几行几列。

图像坐标系到像素坐标系的转换如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image148.png){width="4.073611111111111in"
height="2.3944444444444444in"}

#### **内参和外参**

通过以上四个坐标系的转换可以得到一个点从**世界坐标系**如何转换到**像素坐标系**：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image149.png){width="6.164583333333334in"
height="1.270138888888889in"}

其中从**世界坐标系**到**相机坐标系**的转换矩阵即相机的**外参（extrinsic
property）**，而从**相机坐标系**到**像素坐标系**的3个转换矩阵被合为一个矩阵，即相机的**内参（intrinsic
property）**，也称相机矩阵（camera matrix）。

**参考**

[[https://blog.csdn.net/chentravelling/article/details/53558096]{.underline}](https://blog.csdn.net/chentravelling/article/details/53558096)

**参考**

[[https://docs.opencv.org/master/dc/dbb/tutorial\_py\_calibration.html]{.underline}](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)

### 畸变（distortion）

畸变分为两种，**径向畸变（radial
distortion**，或者叫**径向失真）**和**切向畸变（tangential
distortion**，也叫**切向失真）**。

#### **径向畸变**

**径向畸变**使得直线看上去弯曲，离图像中心越远，径向失真越明显。下图中棋盘的线都应该是直线，然而和标示出来的红色直线相比，棋盘上的线都仿佛凸出来了。![](/home/jimzeus/outputs/AANN/images/media/image150.png){width="2.0805555555555557in"
height="2.4166666666666665in"}

**径向畸变**的原因主要是镜头的径向曲率造成的（光在远离透镜中心的地方比靠近中心的地方更加弯曲）。导致真实成像点向外或者向内偏离理想成像点。其中如果**径向畸变**的像点相较于理想像点向外偏移，远离中心的，称为**枕形畸变（Pincushion
distortion）：**

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image151.png){width="2.7840277777777778in"
height="1.7881944444444444in"}

如果**径向畸变**的像点相对于理想像点沿径向向中心靠拢，称为**桶状畸变（Barrel
Distortion）：**

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image152.png){width="2.8118055555555554in"
height="1.8305555555555555in"}

**径向畸变**可以用如下公式表示：

**x~distorted~ = x(1+k~1~r^2^+k~2~r^4^+k~3~r^6^)**

**y~distorted~ = y(1+k~1~r^2^+k~2~r^4^+k~3~r^6^)**

#### **切向畸变**

切向畸变发生的原因是因为镜头并非完全与成像平面（imaging
plane）平行，因此图像中的某些部分，会显得比实际更加靠近：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image153.png){width="4.128472222222222in"
height="2.2104166666666667in"}

镜头和成像平面通常都平行，切向失真相对于径向失真影响小很多。切向失真的公式如下：

**x~distorted~ = x + \[2p~1~xy + p~2~(r^2^ + 2x^2^)\]**

**y~distorted~ = y + \[p~1~(r^2^ + 2y^2^) + 2p~2~xy\]**

#### **畸变系数**

考虑到径向畸变和切向畸变，我们需要五个参数来表示畸变，通常被称为**畸变系数（distort
coefficients）**:

**Distortion coefficients = (k~1~ k~2~ p~1~ p~2~ k~3~)**

k~3~在p~1~和p~2~的后面是因为有时候会不使用k~3~，这样会损失些微的精确性。事实上如果要更精确，**径向畸变**还可以有k~4~、k~5~等系数。

### 相机标定

**相机标定**（**Camera calibration**）有2个目的：

-   获得相机的**内参**矩阵

-   获得相机的**畸变系数**，以便校正图片

#### **张正友标定法**

**张正友标定法**利用棋盘格标定板，在得到一张标定板的图像之后，可以利用相应的图像检测算法得到每一个角点的像素坐标 (u,v)。

张正友标定法将**世界坐标系**固定于棋盘格上，则棋盘格上任一点的物理坐标 ![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image154.png){width="0.10416666666666667in"
height="2.0833333333333332e-2in"}，由于标定板的世界坐标系是人为事先定义好的，标定板上每一个格子的大小是已知的，我们可以计算得到每一个角点在世界坐标系下的物理坐标(U,V,W=0)。

于是我们有了以下这些信息：

-   每一个角点的**像素坐标** (u,v) 

-   每一个角点的**世界坐标**(U,V,W=0)

来进行相机的标定，获得相机的内外参矩阵、畸变参数。

**参考**

[[https://zhuanlan.zhihu.com/p/94244568]{.underline}](https://zhuanlan.zhihu.com/p/94244568)

#### **Opencv calibrate**

opencv的源码中，跟相机标定相关的有：

-   samples/python/calibrate.py：校正程序

-   samples/data/chessboard.png：棋盘图片，用于打印出来拍摄

-   samples/data/left??.jpg：样例图片，calibrate.py会缺省使用这些图片

    calibrate.py的格式如下：

    usage:

    calibrate.py \[\--debug \<output path\>\] \[\--square\_size\]
    \[\<image mask\>\]

    default values:

    \--debug: ./output/

    \--square\_size: 1.0

    \<image mask\> defaults to ../data/left\*.jpg

    chessboard.jpg为7×7的点（黑白格子的交点，calibrate.py中为9×6）

    ![chessboard](/home/jimzeus/outputs/AANN/images/media/image155.png){width="2.8631944444444444in"
    height="2.9652777777777777in"}

    具体的操作步骤：

1.  将chessboard.jpg打印在纸上

2.  用需要标定的相机拍摄10张以上棋盘纸的图片（参考left??.jpg）

3.  运行calibrate.py得到该相机**畸变系数**和**相机矩阵**

### Opencv Python

相机标定相关的OpenCV Python函数

-   findChessboardCorners()：在输入图片中寻找给定长宽的棋盘格，如果成功返回非0，失败则返回0。

    -   image：输入的带有棋盘格的图片，8bit的黑白或彩色图片

    -   patternSize：棋盘格角点的长宽数量，格式为(columns, rows)

    -   corners：可选参数，输出的探测到的角（同返回值中的corners，C++用）

    -   flags：可选参数，标志位

        -   CALIB\_CB\_ADAPTIVE\_THRESH：该函数的默认方式是根据图像的平均亮度值进行图像 二值化，设立此标志位的含义是采用变化的阈值进行自适应二值化

        -   CALIB\_CB\_NORMALIZE\_IMAGE：在二值化之前，调用EqualizeHist()函数进行图像归一化处理

        -   CALIB\_CB\_FILTER\_QUADS：二值化完成后，函数开始定位图像中的四边形（这里不应该称之为正方形，因为存在畸变），这个标志设立后，函数开始使用面积、周长等参数来筛选方块，从而使得角点检测更准确更严格

        -   CALIB\_CB\_FAST\_CHECK：快速检测选项，对于检测角点极可能不成功检测的情况，这个标志位可以使函数效率提升

    -   返回：

        -   retval：成功则为非0，失败则为0

        -   corners：（不精确的）角点坐标，需调用cornerSubPix()进一步优化

-   cornerSubPix()：进一步改善角点坐标的精度，检测亚像素级角点

    -   image：棋盘格图片，8位单通道图片

    -   corners：角点坐标，既是输入（来自findChessboardCorner()）也是输出

    -   winSize：求亚像素角点的窗口大小，格式为(d,d)，**通常为(5,5)**（d为半径，窗口大小为(2d+1) \*
        > (2d+1)）

    -   zeroZone：设置的"零区域"，在搜索窗口内，设置的"零区域"内的值不会被累加，权重值为0。**通常为(-1,-1)**，表示没有这样的区域

    -   criteria：条件阈值，包括迭代次数阈值和误差精度阈值，一旦其中一项条件满足设置的阈值，则停止迭代，获得亚像素角点。**通常为(cv.TERM\_CRITERIA\_EPS +
        > cv.TERM\_CRITERIA\_COUNT, 30, 0.1)**。

-   drawChessboardCorners()：将角点在棋盘格图片中标示出来

    -   image：棋盘格图片，8位彩色图片

    -   patternSize：棋盘格角点的长宽

    -   corners：找到的角点（来自findChessboardConers()和CornerSubPix()）

    -   patternWasFound：来自findChessboardCorners()的retval

-   calibrateCamera()：通过点的**世界坐标**和**像素坐标**，标定摄像头

    -   objectPoints：点的世界坐标，

    -   imagePoints：点的像素坐标，比如findChessboardCorners()得到

    -   imageSize：

    -   cameraMatrix：同返回的相机矩阵，**通常为None**

    -   distCoeffs：同返回的畸变系数，**通常设为None**

    -   返回：

        -   retval：

        -   cameraMatrix：返回的相机矩阵（内参）

        -   distCoeffs：返回的畸变系数

        -   rvecs：外参中的旋转矩阵

        -   tvecs：外参中的平移矩阵

-   getOptimalNewCameraMatrix()：根据free scaling参数，计算新的相机矩阵

    -   cameraMatrix：相机矩阵

    -   distCoeffs：畸变系数

    -   imageSize：原始的图片大小（单位是像素）

    -   alpha：free scaling系数，介于0到1之间

        -   1：表示原始图片中所有像素都在校正后的新图片中得到体现（此时新图片是一个扭曲的形状，矩形的其余部分被全黑像素填满）

        -   0：表示校正之后的新图片中所有像素都是有效的（即所有超出有效矩形的部分都被裁剪）

    -   newImageSize：可选参数，校正后的图片大小（单位是像素），缺省为imageSize

    -   centerPrinciplePoint：可选参数，标志位，决定原点是否在图像中心。缺省根据alpha自动调整到最适合的值

    -   返回：

        -   newCameraMatrix：新的相机矩阵

        -   validPixROI：有效像素的范围

-   undistort()：校正图像

    -   src：输入图片

    -   cameraMatrix：相机矩阵

    -   distCoeffs：畸变系数

    -   dst：可选参数，同返回的dst，

    -   newCameraMatrix：可选参数，新的相机矩阵，缺省同cameraMatrix

    -   返回：

        -   dst：输出图片

网络构成
========

介绍了构成神经网络的各种层结构、微结构、激活函数、损失函数、优化方法、标准化方法等。

DNN及CNN微结构
--------------

描述了DNN和CNN的微结构，包括单层的结构（比如全连接层和卷积层），或者若干相邻层组成的组合结构（block,
比如Depthwise
Separable卷积），或者某种构建层结构的思想（比如3\*3卷积）。

在后期的**基础CNN网络**构建中，**同类block的堆叠**是一个常用的构建方式。

神

### 全连接层（Dense）

**全连接层（Full-Connection
Layer），**又名**FC层**，或者**Dense层**。相邻两层的所有神经元都有连接。

![Image result for full connection
layer](/home/jimzeus/outputs/AANN/images/media/image156.png){width="4.036111111111111in"
height="1.8354166666666667in"}

全连接层的参数数量W（权重数量），B（偏置数量），P（参数总数）分别是：

**W = I \* O**

**B = O**

**P = W+B =（I+1）\* O**

其中I为输入神经元的数量，O为输出神经元的数量。如果输入来自二维卷积层，则

I = C \* H \*
W，C、H、W分别为卷积层输出特征图的通道数、长、宽。相对于其他层而言，FC层的参数数量是最多的。

全连接层的FLOPs为：

**FLOPs = I \* O**

即对每个输出神经元，都要做I次权重乘法，I -1 次权重累加，1次偏置累加。

### 池化层（Pooling）

**池化层（Pooling
Layer）**，进行缩小长和高方向上的空间的运算的层，池化层不改变通道数量。通常是Max池化，也有Average池化等。

池化层只有定义池化过滤器大小、padding大小、stride大小的超参数，没有可学习的参数。

池化层的计算量为：

**FLOPs = H \* W \* FH \* FW \* CO**

H和W为输出特征图的长和宽。

### 全局池化层

**全局池化层**（**Global Pooling
Layer**）是特化的池化层，全局池化层的过滤器大小跟输入特征图的大小完全一样，整个特征图经过池化之后变为一个标量（加上通道就是一个向量）。全局池化层分为**Global
Average Pooling**和**Global Max Pooling**。

全局池化层的计算量套用池化层的计算量公式得出:

**FLOPs = FH \* FW \* CO**

FH和FW为过滤器的大小，也是输入特征图的大小

### 反池化层

**反池化层（Unpooling Layer）**是上采样的一种方式，通常是Max反池化。

池化操作是不可逆的，通过max池化时记录下max池化时最大值的位置，在反池化时将最大值放回该位置，并将其余部分填0，反池化操作可以近似实现池化逆操作。

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562296083(1).png](/home/jimzeus/outputs/AANN/images/media/image157.png){width="5.0152777777777775in"
height="2.984722222222222in"}

### 常规卷积层

**卷积层（Convolution
Layer）**，进行卷积运算的层，由卷积核组成。CNN的重要组成部分。

2维卷积层的参数数量W（权重数量），B（偏置数量），P（参数总数）分别为：

**W = FH \* FW \* CI \* CO**

**B = CO**

**P = W + B = （FH \* FW \* CI + 1 ）\* CO**

其中FH和FW为卷积核（过滤器）的长和宽（通常是一致的，都写作K），CI为输入的通道数（输入特征图数），CO为输出的通道数（输出特征图数）

2维卷积层的计算量为：

**FLOPs = FH \* FW \* CI \* H \* W \* CO **

其中卷积乘法操作数量是**FH\*FW\*H\*W\*CI\*CO**

每个输入通道内的卷积累加操作数量是**(FH\*FW-1) \* H\*W\*CI\*CO**

输入通道间的累加操作数量是**(CI-1)\*H\*W\*CO**

偏置加法操作数量是**H\*W\*CO**。

这个计算量中：

-   FH\*FW表示对单个输入特征图的单个卷积核的一次卷积（不是一整张输入特征图）

-   乘以CI为输出特征图中每个像素的计算量

-   乘以H\*W表示每张输出特征图的计算量

-   再乘以CO（输出特征图数量）表示所有特征图的计算量

H和W分别为输出特征图的长和宽，如果Stride为1，则输入和输出特征图大小一样，否则输入特征图为输出特征图的S\*S倍。

### 本地卷积层

**本地卷积层（Locally-Convolutional
Layer）**，类似卷积层，区别在于卷积核不共享，也就是说对矩阵（图像）的每个局部块进行卷积运算的时候，都使用不同的卷积核。

因为参数不共享，本地卷积层的参数数量会远远大于卷积层，各个参数W（权重数量），B（偏置数量），P（参数总数）分别为：

**W = FH \* FW \* CI \* CO \* H \* W**

**B = CO**

**P = W + B = FH \* FW \* CI \* CO \* H \* W + CO**

各字母意思参考卷积层。

本地卷积层的计算量和普通的卷积层一致。

### Dropout层

Dropout用于在训练的时候随机的关闭一定比率的神经元，其效果是免于过拟合，这个比率是个超参数，称为**dropout
ratio**。Dropout层**只在训练中工作**，实际推理的网络的神经元会全部打开。

Dropout首次提出是在2012年的AlexNet中。

**参考**

12种主要的Dropout方法：如何应用于DNNs，CNNs，RNNs中的数学和可视化解释

[[https://zhuanlan.zhihu.com/p/146747803]{.underline}](https://zhuanlan.zhihu.com/p/146747803)

### Deconv（ZFNet）

**反卷积（Deconvolution，Deconv）**的主要目的是为了将feature
map+卷积核可视化，反卷积可能用在三个地方：

-   CNN可视化，比如在ZFNet论文中用到的可视化方法，用于分析理解网络

-   图片重建，比如用GAN生成图片

-   上采样（upsampling）

反卷积的具体实现请参考[[ZFNet]{.underline}](\l)

### Group Conv（AlexNet）

**Group
convolution**，**分组卷积**，或**群卷积**。通常的卷积过程是：输入特征图尺寸为H\*W\*CI，经过CO个卷积核（尺寸为FH\*FW\*CI\*CO），输出特征图尺寸为H\*W\*CO。而组卷积则是将输入通道分成若干组G，每组CI/G个通道，每个组内有CO/G个卷积核，尺寸为FH\*FW\*(CI/G)\*(CO/G)，每组输出CO/G个特征图。这样总的输入输出不变，

分组卷积可以减少参数及计算量，假设分为G组，采用组卷积之前的参数和计算量为：

**Params =（FH \* FW \* CI + 1 ）\* CO**

**FLOPs = FH \* FW \* CI \* H \* W \* CO**

而分成G组之后的参数和计算量为：

**Params =（FH \* FW \* CI/G + 1 ）\* CO/G \* G**

**FLOPs = FH \* FW \* （CI/G） \* H \* W \* （CO/G）\* G =
FH\*FW\*CI\*H\*W\*CO/G**

可以看到参数和计算量都变为原来的1/G

分组卷积最早的出现应该是AlexNet中，受限于硬件条件，前面的卷积层被分成了两组，分别对应两块GPU，减少训练时间，加快训练速度。

DW卷积（Depthwise Conv）是分组卷积的一个特殊情况，每个通道独立卷积。

分组卷积在后来的ResNeXt、ShuffleNet等也有出现，其目的也是为了减小计算量，类似于一种介于常规卷积和DW卷积之间的中间状态。

### Depthwise Separable卷积（Xception）

**Depthwise separable
卷积（深度可分离卷积）**是为了减少参数及计算量，并且能达到常规卷积的效果而出现的。Depthwise
Separable卷积将一个常规的卷积操作分成两层完成，前一层叫depthwise
convolution（DW卷积），后一层叫pointwise convolution（PW卷积）。

深度可分离卷积在主流CNN里首次出现是**Xception**和 **MobileNet
V1**，这两者都是Google的，推出时间也差不多。

#### Depthwise卷积层

**Depthwise卷积层**，又叫**DW卷积层**，**逐层卷积**。不同于常规卷积操作中每个卷积核是同时操作所有输入通道，**Depthwise
Convolution**的每个卷积核只负责一个输入通道，一个通道只被一个卷积核卷积。因此DW卷积的卷积核数、输出通道数CO和输入通道数CI是一样（如果有Same
padding则输出特征图的维度和输入特征图完全一致）。

![è¿éåå¾çæè¿°](/home/jimzeus/outputs/AANN/images/media/image158.png){width="4.79375in"
height="1.99375in"}

Depthwise Convolution完成后的Feature
map数量与输入层的通道数相同，无法扩展Feature
map。而且这种运算对输入层的每个通道独立进行卷积运算，没有有效的利用不同通道在相同空间位置上的feature信息。因此需要Pointwise
Convolution来将这些Feature map进行组合生成新的Feature map。

Depthwise Convoluation算是分组卷积的一种极端情况。

-   DW卷积层的参数数量为：

**W = FH \* FW \* CO**

**B = CO**

**P = W + B = (FH+FW+1) \* CO**

-   DW卷积层的计算量为：

**FLOPs = FH \* FW \* H \* W \* CO**

其中卷积乘法操作数量：**FH \* FW \* H \* W \* CO**

通道内卷积累加操作数量：**（FH \* FW -1）\* H \* W \* CO**

通道间累加操作：无

偏置累加操作数量：**H \* W \* CO**

#### Pointwise卷积层

**Pointwise卷积层**，又叫**PW卷积层**，**逐点卷积**。Pointwise卷积就是一个卷积核为1\*
1 \* CI \*
CO的普通卷积层，所以这里的卷积运算会将上一步的map在深度方向上进行加权组合，生成新的Feature
map。

![è¿éåå¾çæè¿°](/home/jimzeus/outputs/AANN/images/media/image159.png){width="5.154861111111111in"
height="2.188888888888889in"}

PW卷积层的参数和计算量参考常规卷积层为：

**FLOPs = CI \* CO \* H \* W**

#### 计算量

根据上面的公式，来计算下能省下多少计算力

假设输入大小为 56\*56\*128，输出为也为56\*56\*128，卷积核大小为3 \*
3，如果用常规卷积，则计算量为：

FH \* FW \* CI \* CO \* H \* W = 3\*3\*128\*128\*56\*56 = 462,422,016。

而如果使用depthwise
separable卷积（事实上这就是Mobilenet的一部分），计算量为：

DW\_FLOPs = FH\*FW\*H\*W\*CO = 2\*3\*3\*56\*56\*128 = 3,627,672

PW\_FLOPs = FH\*FW\*H\*W\*CI\*CO = 1\*1\*56\*56\*128\*128 = 51,380,224

FLOPs = DW\_FLOPs + PW\_FLOPs = 7,255,344 + 102,760,448 = 55,007,896

可以看到这里的Depthwise
separable卷积相对于常规卷积层，计算量只有其的1/8不到。

### 3\*3卷积核（VGGNet）

在早期的CNN中，可以看到一些大卷积核（11\*11，7\*7，5\*5），但在2014年VGG的论文中发现，两次3\*3卷积的感受野和一个5\*5卷积一样，三次3\*3卷积的感受野和一次7\*7卷积一样，但是计算量能减少很多。

假设输入特征图为H\*W，输入通道CI，输出通道CO，则计算量FLOPs如下：

一层5\*5的FLOPs=5\*5\*H\*W\*CI\*CO = 25K

两层3\*3的FLOPs=3\*3\*H\*W\*CI\*CO \*2 = 18K

一层7\*7的FLOPs=7\*7\*H\*W\*CI\*CO = 49K

三层3\*3的FLOPs=3\*3\*H\*W\*CI\*CO\*3 = 27K

可以看到计算量下降了不少，因此自VGG之后，CNN大都使用多3\*3卷积来替代大尺度的卷积核。

### 1\*1卷积核

N\*N卷积的目的一般是用来整合相邻区域内的信息，而1\*1的卷积核并不能实现这个目的，它的作用有2个：

-   改变维度：1\*1卷积核可以用于降维或者升维，比如一个H\*W\*CI的输入特征图，经过CO个1\*1\*CI个卷积核之后，形状变为H\*W\*CO。

-   跨通道的交互：类似于一个不同通道间同一位置的全连接，在ResNet、FCN等中出现，用于替代FC层。

1\*1卷积就是**PW卷积（Pointwise
Conv）**，本质上就是一个常规的卷积层，只不过其卷积核大小为1\*1。计算量为：

**FLOPs = CI \* CO \* H \* W**

### Spatial Separable卷积

**空间可分离卷积（Spatial Separable
Convolution，SS卷积）**也是一种将一个常规卷积层分解为2个卷积层，以减少计算量的方式。但是**空间可分离卷积**有其局限性，**因此在深度学习中使用的不多**。

SS卷积是将常规的N\*N卷积核在长和宽上分解为两个N\*1和1\*N的卷积核，分别作为一个单独的卷积层。

![https://cdn-images-1.medium.com/max/1600/1\*o3mKhG3nHS-1dWa\_plCeFw.png](/home/jimzeus/outputs/AANN/images/media/image160.png){width="5.084722222222222in"
height="2.607638888888889in"}

**SS卷积**和**深度可分离卷积**类似，都是把一个常规卷积层拆分成两个，区别在于拆分的维度不一样。常规卷积层中，每个输出特征图的每个像素，需要在三个维度上卷积（即**宽W、高H、输入通道CI**）。

-   深度可分离卷积将这三个维度拆分为**宽**和**高**先卷积，再在**通道**上融合。

-   SS卷积则是先在**宽**和**通道**上卷积，再在**高**和**通道**上卷积

至于计算量的节省，可以简单的通过公式得出，我们以常用的3\*3卷积核为例：

原先的计算量= 3\*3\*CI\*CO\*H\*W = 9 \* H \* W \* CI \* CO

SS卷积的计算量 = 1\*3\*CI\*CO\*H\*W + 3\*1\*CI\*CO\*H\*W = 6 \* H \* W
\* CI \* CO

可以看见，计算量的节省并不算非常大。

SS卷积最大的问题，就是并非所有的卷积核矩阵都可以表示为1\*N和N\*1的矩阵的点积。

### 带孔卷积

**Dilated
Convluation**，又叫**空洞卷积**、**带孔卷积**、**扩展卷积**。带孔卷积的目的是为了在不增加参数和计算量的情况下扩大感受野。其原理是在卷积的时候，跳着卷积特征图。

![](/home/jimzeus/outputs/AANN/images/media/image161.png){width="4.584722222222222in"
height="1.636111111111111in"}

上图中红点为卷积核卷积的位置，a为常规卷积，b为间隔为1的带孔卷积，c为间隔为3的带孔卷积。

### Bottleneck结构

Bottleneck结构也是CNN中减小计算量的方式，Bottleneck通常分成3个卷积层：先使用1\*1的卷积核（即PW卷积）对数据进行降维（减少通道），再进行常规的卷积，最后再用1\*1的卷积核进行升维。所谓瓶颈，指的就是中间层的通道数小于前后层。

![Bottleneck](/home/jimzeus/outputs/AANN/images/media/image162.jpeg){width="4.136111111111111in"
height="3.5409722222222224in"}

上面两个卷积过程的输入和输出都是H\*W\*256，左边常规卷积的计算量为：

**FLOPs = 3\*3\*256\*256\*H\*W = 589,824\*H\*W**

右边Bottleneck结构的计算量为：

**FLOPs =1\*1\*256\*64\*H\*W + 3\*3\*64\*64\*H\*W + 1\*1\*64\*256\*H\*W
= 69,632\*H\*W**

可以看到Bottleneck的计算量相较于常规卷积层下了一个数量级

### Residual Block（ResNet）

**残差结构（Residual
Block）**于2015年由微软的何凯明等人在**ResNet**中提出，其特点是在网络中增加了直连通道（被称作shortcut或skip
connection），

![https://cdn-images-1.medium.com/max/1200/1\*JCy4o3c6vs9jgAF35V0LOQ.png](/home/jimzeus/outputs/AANN/images/media/image163.png){width="2.326388888888889in"
height="1.3604166666666666in"}

传统的卷积网络或者全连接网络在信息传递的时候会存在信息丢失，损耗等问题，同时还有导致梯度消失或者梯度爆炸，导致很深的网络无法训练。残差结构在一定程度上解决了这个问题，通过直接将输入信息绕道传到输出，保护信息的完整性，整个网络只需要学习输入、输出差别的那一部分，简化学习目标和难度。

#### Residual bottleneck（ResNet）

在层数较深的实现中，ResNet使用了一种叫做**Residual bottleneck**的结构：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\af94ed7a3ced3e38f1814ec343204cf.png](/home/jimzeus/outputs/AANN/images/media/image164.png){width="4.169444444444444in"
height="2.0305555555555554in"}

图中左侧是浅层ResNet所用的**Residual block**，而右侧便是**residual
bottleneck**，先通过PW卷积降维，接着是一个3\*3的常规卷积，最后又用一个PW卷积升维，而shortcut连接了这三层的前后。

### Inverted Residual Block（MobileNet V2）

**Inverted Residual
Block**结构（**反残差，倒残差**）首次出现于MobileNetV2中。在层数比较深的ResNet中，会使用Residual
bottleneck结构，Inverted residual block结构与之类似，但有2点区别：

-   Residual bottleneck中间的3\*3卷积用的是常规卷积，而Inverted residual
    block中间的3\*3卷积用的是DW卷积

-   Residual
    bottleneck前面的1\*1卷积（PW卷积）作用是降维，而后面是升维（所以才叫"瓶颈"），而Inverted
    residual
    block正相反，前一个PW卷积升维，后一个降维。这么做的原因也和DW卷积有关：相较于常规卷积，DW卷积大幅度降低了计算量，因此没有普通bottleneck的需求（降维卷积减少计算量），而在高维提取特征效果更好。

见下图，上面是**Residual bottleneck**，下面是**Inverted Residual
Block**：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\846a9ec490185494b109306e7bc895d.png](/home/jimzeus/outputs/AANN/images/media/image165.png){width="5.163194444444445in"
height="1.5118055555555556in"}

### Linear bottleneck（MobileNet V2）

Linear bottleneck结构也是首次出现于MobileNet V2中，是一个基于MobileNet
V1中的Depthwise Separable卷积做的改进，其区别在于：

-   Linear
    bottleneck在DW卷积之前增加了一个PW卷积用于升维，其原因是DW卷积没法提升通道数，效果不够好。

-   Linear
    bottleneck去掉了第二个PW卷积的激活函数ReLU6，作者认为这个激活函数在高维空间能够有效的增加非线性，但在低维空间会破坏特征。

见下图：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\d1225f57f9b49690142d46738b1130b.png](/home/jimzeus/outputs/AANN/images/media/image166.png){width="4.954166666666667in"
height="1.4in"}

### Dense Block（DenseNet）

Dense Block是Residual
Block的极端化版本，首次出现在DenseNet中。在一个Dense
block中的所有层之间都有连接，下图是一个Dense block

![https://cdn-images-1.medium.com/max/1200/1\*l9\_9ySv8bhjj-Fz5vNVWLg.jpeg](/home/jimzeus/outputs/AANN/images/media/image167.jpeg){width="3.848611111111111in"
height="2.7305555555555556in"}

写成伪代码如下：

def dense\_block(x, f=32, d=5):\
l = x\
for i in range(d):\
x = conv(l, f)\
l = concatenate(\[l, x\])\
return l

### Inception结构（GoogleNet）

Inception结构首次出现于2014年谷歌推出的GoogLeNet中，其主要思想是**并联**，即若干不同大小的卷积层和池化层并联。Inception是CNN历史上的一个里程碑，在此之前，CNN只是更深的卷积层的简单堆叠。

Inception的主要思想是，物体在不同图片中占的比例会完全不同，可能是占据了整幅图片的特写，也可能只占据画面一角。因此在同一层使用不同大小的卷积核，用于感受不同的视野上的特征。

流行的Inception版本包括V1，V2，V3，V4和Inception-ResNet 。

**参考**

[https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202]{.underline}

#### Inception V1

V1结构是最早的版本，将1\*1，3\*3，5\*5的卷积层和3\*3的池化层并联在一起，其中为了降低3\*3和5\*5卷积层的计算量，先用1\*1卷积进行了降维。

![https://gss3.bdstatic.com/7Po3dSag\_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=aa07ecd5e9dde711f3df4ba4c686a57e/a50f4bfbfbedab647c2a1110fb36afc378311e81.jpg](/home/jimzeus/outputs/AANN/images/media/image168.jpeg){width="4.636111111111111in"
height="2.5388888888888888in"}

**论文**

Going Deeper with Convolutions

[[https://arxiv.org/pdf/1409.4842.pdf]{.underline}](https://arxiv.org/pdf/1409.4842.pdf)

#### Inception V2 V3

Inception V2主要做了以下改动:

-   将之前的5\*5卷积核拆成两层3\*3的卷积核，减少计算量（使用了VGG中出现的思想）

-   将之前的N\*N卷积核拆成N\*1和1\*N两层（即Spatial Separable卷积）

-   扩展了Inception结构中卷积结构的宽度，而非深度，以降低表现力瓶颈（Representational
    Bottleneck），如下图

![https://cdn-images-1.medium.com/max/1600/1\*DVXTxBwe\_KUvpEs3ZXXFbg.png](/home/jimzeus/outputs/AANN/images/media/image169.png){width="3.0729166666666665in"
height="2.436111111111111in"}

以上三个改动分别被应用在V2的三个模型中

Inception V3则有如下改动：

-   应用了V2中所有的三个改动

-   使用了RMSProp优化方法替代之前的SGD

-   辅助分类器中增加了BN层

-   Label Smoothing

**论文**

V2和V3用的是同一篇，Rethinking the Inception Architecture for Computer
Vision

[[https://arxiv.org/pdf/1512.00567.pdf]{.underline}](https://arxiv.org/pdf/1512.00567.pdf)

#### Inception V4 / Inception-ResNet

Inception V4主要的改动是增加了残差网络（ResNet），具体请自行参考论文。

**论文**

Inception-v4, Inception-ResNet and the Impact of Residual Connections on
Learning

[[https://arxiv.org/pdf/1602.07261]{.underline}](https://arxiv.org/pdf/1602.07261)

（Inception V4和Inception-ResNet也来自同一篇论文）

### ResNeXt结构（ResNeXt）

ResNeXt结构是Inception和ResNet思想的结合，有点类似**Inception-ResNet**，但区别在于各个path是一样的，下图左为ResNet结构，右为ResNeXt结构：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image170.png){width="3.946527777777778in"
height="2.3916666666666666in"}

### Fire Module（SqueezeNet）

Fire module是出现在SqueezeNet里的微结构，由两个部分组成，分别是：

-   squeeze层：由1\*1卷积组成，作用是降维

-   expand层：由1\*1卷积和3\*3卷积并联而成，类似Inception结构

Fire module不改变特征图的尺寸，只改变其通道数。

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\373ed48f55238384aa29548652dd163.png](/home/jimzeus/outputs/AANN/images/media/image171.png){width="4.3909722222222225in"
height="2.2847222222222223in"}

### SE结构（SENet）

SE结构相当于一个通道间的**注意力机制**，SE结构的操作分为三个步骤，Squeeze,
Excitation和Reweight（Scale）。

给定一个输入 x，其特征通道数为
c\_1，通过一系列卷积等一般变换后得到一个特征通道数为 c\_2
的特征。与传统的 CNN
不一样的是，接下来我们通过三个操作来重标定前面得到的特征。

![](/home/jimzeus/outputs/AANN/images/media/image172.png){width="5.759722222222222in"
height="1.2798611111111111in"}

首先是 **Squeeze 操作**，我们顺着空间维度来进行特征压缩（比如通过Global
Average
Pooling），将每个二维的特征通道变成一个实数，这个实数某种程度上具有全局的感受野，并且输出的维度和输入的特征通道数相匹配。它表征着在特征通道上响应的全局分布，而且使得靠近输入的层也可以获得全局的感受野，这一点在很多任务中都是非常有用的。

其次是 **Excitation
操作**，它是一个类似于循环神经网络中门的机制（注意力机制）。通过参数 w
来为每个特征通道生成权重，其中参数 w
被学习用来显式地建模特征通道间的相关性。

最后是一个 **Reweight 的操作**，我们将 Excitation
的输出的权重看做是进过特征选择后的每个特征通道的重要性，然后通过乘法逐通道加权到先前的特征上，完成在通道维度上的对原始特征的重标定。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image173.jpeg){width="5.173611111111111in"
height="2.83125in"}

上左图是将 SE 模块嵌入到 Inception
结构的一个示例。方框旁边的维度信息代表该层的输出。

这里我们使用 global average pooling 作为 Squeeze 操作。紧接着两个 Fully
Connected 层组成一个 Bottleneck
结构去建模通道间的相关性，并输出和输入特征同样数目的权重。我们首先将特征维度降低到输入的
1/16，然后经过 ReLu 激活后再通过一个 Fully Connected
层升回到原来的维度。这样做比直接用一个 Fully Connected 层的好处在于：

-   具有更多的非线性，可以更好地拟合通道间复杂的相关性；

-   极大地减少了参数量和计算量。

然后通过一个 Sigmoid 的门获得 0\~1 之间归一化的权重，最后通过一个 Scale
的操作来将归一化后的权重加权到每个通道的特征上。

除此之外，SE 模块还可以嵌入到含有 skip-connections 的模块中。上右图是将
SE 嵌入到 ResNet 模块中的一个例子，操作过程基本和 SE-Inception
一样，只不过是在 Addition 前对分支上 Residual
的特征进行了特征重标定。如果对 Addition
后主支上的特征进行重标定，由于在主干上存在 0\~1 的 scale
操作，在网络较深 BP
优化时就会在靠近输入层容易出现梯度消散的情况，导致模型难以优化。

**参考**

[[https://www.cnblogs.com/bonelee/p/9030092.html]{.underline}](https://www.cnblogs.com/bonelee/p/9030092.html)

### NASNet单元（TODO）

### AmoebaNet单元（TODO）

### SPP层（SPP-net）

**Spatial Pyramid Pooling
Layer，空间金字塔池化层**，是出现在SPP-net中的池化层。

卷积层是输入大小无关的，也就是说无论输入图片的长宽如何，都可以正常的走过整个卷积流程，但全连接层则需要固定输入的大小，因此传统的CNN输入的图片大小必须是固定的。

SPP层的目的是为了使得网络可以接收不同大小的图片（即同一图片中不同大小的区域），其原理是：针对卷积之后的特征图，以不固定大小的池化核进行池化，使得其产生固定大小的输出，如下图中的空间金字塔池化层使用了3个不固定大小的池化核，使得这三个级别的池化层的输出尺寸分别为1\*1，2\*2，4\*4，这样总的输出特征维度为（1+4+16）\*256=5376：

![https://images2015.cnblogs.com/blog/539316/201707/539316-20170716000938478-1529067244.png](/home/jimzeus/outputs/AANN/images/media/image174.png){width="3.839583333333333in"
height="2.6666666666666665in"}

**论文**

Spatial Pyramid Pooling in Deep Convolutional Networks for Visual
Recognition

[[https://arxiv.org/pdf/1406.4729]{.underline}](https://arxiv.org/pdf/1406.4729)

### RoI Pooling层（Fast R-CNN）

**RoI（Region of Interest） Pooling Layer（RoI池化层）**出现在**Fast
R-CNN**中，它受到了**SPP层**的启发，其实就是一个特化的SPP层：**RoI池化层**只有一个金字塔级别（在Fast
R-CNN中是7\*7）。假设图片大小为m\*n，则将poolling的核大小和stride都设为m/7和n/7，这样无论输入的图片大小为多少，出来的feature
map都是7\*7。

参考《[[研究方向：图像 \> 目标检测 \> R-CNN系列 \> Fast
R-CNN]{.underline}](\l)》

CNN结构
-------

CNN的基础网络基本上只有一种宏观结构。自深度学习介入到图像处理以来，最基础的**图像分类**任务的CNN的基本结构（同时也是很多其他更复杂的图像/视频处理网络的基础网络，即backbone）

几乎可以在任何一个图像分类网络中看到这种结构，其特点是：

1.  输入为h\*w\*3（高、宽、RGB三个颜色通道）

2.  通过若干微结构，使得特征图越来越小，同时channel越来越大

3.  通常这些微结构是重复的，其中最典型和基础的为：卷积层 + 池化层

4.  最后接一个微结构（最典型的情况是一个FC层），并变为若干分类（Softmax）。

    大致如下图所示：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image175.GIF){width="4.1666666666666664e-2in"
height="8.333333333333333e-2in"}![timg](/home/jimzeus/outputs/AANN/images/media/image176.jpeg){width="4.375in"
height="1.7486111111111111in"}

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image175.GIF){width="4.1666666666666664e-2in"
height="8.333333333333333e-2in"}

RNN结构
-------

循环神经网络单元，**Recurrent Neural
Network**，其被设计出来的原因是因为传统的神经网络无法处理时间序列数据，最根本的原因是传统NN没有记忆，而RNN解决了这个问题。

RNN可以分为狭义和广义两个概念，狭义的RNN指的是微结构内仅有一个tanh，如下图：

![](/home/jimzeus/outputs/AANN/images/media/image177.png){width="3.3520833333333333in"
height="1.2541666666666667in"}

而广义的RNN则指代所有包括时间序列的神经网络，包括LSTM、GRU等等各种变种。

近年来RNN逐渐被Transformer代替。

本节描述了RNN（广义）的神经网络结构。

### 通用结构

通常各种介绍RNN、LSTM等等的文章主要介绍的是各种RNN及变体的单元微结构：

![](/home/jimzeus/outputs/AANN/images/media/image178.png){width="1.0118055555555556in"
height="1.0215277777777778in"}

从图上看，这个微结构只有2个输入，这两个输入其实是两个输入向量，包括：

-   自身的前一个时间步骤的输出向量h~t-1~

-   当前时间步骤的输入向量X~t~

那么这个单一结构的单元如何在宏观上扩展，从而组成了一整个RNN网络？请看下图：

![](/home/jimzeus/outputs/AANN/images/media/image179.png){width="5.768055555555556in"
height="2.33125in"}

图中的每个图层表示了一个timestep（时间步骤），也就是说所有图层其实是同一个网络在不同时刻的表现。每个图层（也就是某一时刻），在不考虑时间步骤输入的情况下，就是一个全连接网络。不同于普通全连接层的是：

1.  这个网络处理的不是一个完整的输入（比如CNN中的图像），而只是一个完整输入（比如一个句子）的其中一个时间步骤（比如句子中的一个词）

2.  对每个单元来讲，除了输入的向量之外还有一个时间步骤的输入向量，**图中红线并不严谨**，应该是个隐层（前个时间步骤）到隐层（后一个时间步骤）的全连接

不以时间展开的RNN框架可以用以下结构图展示出来：

-   一个三维的输入层（一个时间步骤）

-   隐藏层包括四个RNN单元，与输入层全连接，也与自身（相邻时间步骤）全连接

-   一个二维的输出层，与隐藏层全连接

![](/home/jimzeus/outputs/AANN/images/media/image180.png){width="4.511805555555555in"
height="2.2930555555555556in"}

这是最基本的RNN宏结构，包括一个输入层(n\*d，n为样本数，d为向量维度）、一个输出层和一个隐藏层。这里面的参数包括了3个矩阵：

1.  U：输入层-\>隐藏层（d\*h，d为输入层向量维度，h为隐藏层单元个数）

2.  V：隐藏层-\>输出层（h\*q，h为隐藏层单元个数，q为输出层单元个数）

3.  W：隐藏层-\>隐藏层（不同时间步骤之间共享**同一个**参数矩阵，h\*h）

![](/home/jimzeus/outputs/AANN/images/media/image181.png){width="3.8333333333333335in"
height="1.5381944444444444in"}

对于给定的时刻t，隐藏层h，输入x之间的算法为：

![](/home/jimzeus/outputs/AANN/images/media/image182.png){width="2.954861111111111in"
height="0.4423611111111111in"}

其中ϕ为激活函数，一般是tanh，b为偏置。输出o的算法则为：

![](/home/jimzeus/outputs/AANN/images/media/image183.png){width="1.5638888888888889in"
height="0.41805555555555557in"}

### 变种结构

以下是RNN（包括LSTM、GRU等）宏观结构的几种时序方面的变种：

![](/home/jimzeus/outputs/AANN/images/media/image184.jpeg){width="5.768055555555556in"
height="1.8055555555555556in"}

由于没有时许，图中最左边并非是一个RNN，而另外四种则是RNN**宏观结构**的四种形态。下面一一介绍：

**参考**[[https://zhuanlan.zhihu.com/p/28054589]{.underline}](https://zhuanlan.zhihu.com/p/28054589)

#### N vs N

这就是最基本的RNN结构，即每一步都输出的通用结构，但是由于输入和输出必须完全完全等长，导致这种结构的实际应用很少，但也有一些问题适合用经典的RNN结构建模：

-   计算视频每一帧的分类标签。因为要对每一帧进行计算，因此输入和输出序列等长。

-   输入为字符，输出为下一个字符的概率。这就是著名的Char
    RNN（可以用来生成文章，诗歌，甚至是代码）。

![](/home/jimzeus/outputs/AANN/images/media/image185.jpeg){width="3.251388888888889in"
height="2.560416666666667in"}

#### N vs 1

输入一个序列，输出一个非时序的张量。这种情况仅在最后一步进行输出变换，这种结构通常用来处理序列分类问题。比如：

-   输入一段文字判别它所属的类别

-   输入一个句子判断其情感倾向

-   输入一段视频并判断它的类别

等等。

![](/home/jimzeus/outputs/AANN/images/media/image186.jpeg){width="3.2118055555555554in"
height="2.4625in"}

#### 1 vs N

输入不是序列而输出为序列的情况怎么处理？我们可以只在序列开始进行输入计算：

![](/home/jimzeus/outputs/AANN/images/media/image187.jpeg){width="3.1055555555555556in"
height="2.4277777777777776in"}

还有一种结构是把输入信息X作为每个阶段的输入：

![](/home/jimzeus/outputs/AANN/images/media/image188.jpeg){width="3.0506944444444444in"
height="2.638888888888889in"}

这种1 VS N的结构可以处理的问题有：

-   从图像生成文字（**image
    caption**），此时输入的X就是图像的特征，而输出的y序列就是一段句子

-   从类别生成语音或音乐等

#### N vs M

这是RNN通用结构最重要的一个变种，这种结构又叫**Encoder-Decoder**结构，或者**Seq2Seq**结构。

原始的N vs N
RNN要求序列等长，然而我们遇到的大部分问题序列都是不等长的，如机器翻译中，源语言和目标语言的句子往往并没有相同的长度。

为此，Encoder-Decoder结构先将输入数据编码成一个上下文向量c：

![](/home/jimzeus/outputs/AANN/images/media/image189.jpeg){width="2.7666666666666666in"
height="1.801388888888889in"}

得到c有多种方式，最简单的方法就是把Encoder的最后一个隐状态赋值给c，还可以对最后的隐状态做一个变换得到c，也可以对所有的隐状态做变换。

拿到c之后，就用另一个RNN网络对其进行解码，这部分RNN网络被称为Decoder。具体做法就是将c当做之前的初始状态h0输入到Decoder中：

![preview](/home/jimzeus/outputs/AANN/images/media/image190.jpeg){width="4.08125in"
height="1.6381944444444445in"}

还有一种做法是将c当做每一步的输入：

![](/home/jimzeus/outputs/AANN/images/media/image191.jpeg){width="4.0in"
height="2.2104166666666667in"}

由于这种Encoder-Decoder结构不限制输入和输出的序列长度，因此应用的范围非常广泛，比如：

-   机器翻译，Encoder-Decoder的最经典应用，事实上这一结构就是在机器翻译领域最先提出的

-   文本摘要，输入是一段文本序列，输出是这段文本序列的摘要序列。

-   阅读理解，将输入的文章和问题分别编码，再对其进行解码得到问题的答案。

-   语音识别，输入是语音信号序列，输出是文字序列。

-   视频摘要（Video Caption），输入是一段视频，输出是该视频的描述

**论文**

**Encoder-Decoder**：Learning Phrase Representations using RNN
Encoder--Decoder for Statistical Machine Translation

[[https://arxiv.org/pdf/1406.1078.pdf]{.underline}](https://arxiv.org/pdf/1406.1078.pdf)

**Seq2Seq**：Sequence to Sequence Learning with Neural Networks

[[https://arxiv.org/pdf/1409.3215.pdf]{.underline}](https://arxiv.org/pdf/1409.3215.pdf)

### 双向RNN

双向RNN，Bidirectional
RNN，顾名思义RNN在两个方向同时进行，它假设当前的输出不仅仅和之前的序列有关系，还和之后的序列有关系（比如预测一个缺失的词，需要同时根据上下文进行预测）。Bidirectional
RNN是一个相对简单的RNNs，由两个RNNs上下叠加在一起组成。输出由这两个RNNs的隐藏层的状态决定。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image192.jpeg){width="4.783333333333333in"
height="1.5770833333333334in"}

我们需要将从前往后和从后往前这两部分的结果拼接起来，如果他们都是1000\*1维的，则拼接起来就是2000\*1维的。

论文

[[https://www.di.ufpe.br/\~fnj/RNA/bibliografia/BRNN.pdf]{.underline}](https://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)

RNN微结构
---------

RNN微结构介绍的是单个RNN单元内部的结构

### 基本RNN单元

基本RNN的单元都带有循环，基本单元如下图：

![](/home/jimzeus/outputs/AANN/images/media/image193.png){width="1.6in"
height="1.6152777777777778in"}

可以看到相较于普通的神经网络（DNN/CNN），RNN增加了循环，其中A为RNN单元，xt表示输入，ht表示输出 ，t为表示不同的时间步骤。对于每个RNN单元来说，有2个输入：

-   一个是用户的输入，也是时间序列输入xt，比如**自然语言处理**任务中的每个词。

-   另一个是同一个单元的上一步的输出，也就是第t步的输出ht被作为t+1步的输入。

按时间展开之后如下：

![](/home/jimzeus/outputs/AANN/images/media/image194.png){width="4.769444444444445in"
height="1.2444444444444445in"}

所有的RNN（广义的RNN，包括LSTM、GRU等）都有类似的结构，但每个不同的RNN变种单元中的具体细节有些区别。我们这里的RNN特指最简单的RNN（或者叫标准RNN，Standard
RNN，Keras中被称为SimpleRNN），该RNN单元内部的结构如下图所示：

![](/home/jimzeus/outputs/AANN/images/media/image195.png){width="4.647916666666666in"
height="1.738888888888889in"}

相较于DNN的隐节点数学表达式为：

![2020-03-13 18-45-53屏幕截图
(复件)](/home/jimzeus/outputs/AANN/images/media/image196.png){width="1.4347222222222222in"
height="0.24166666666666667in"}

RNN隐节点的数学表达式为：

![2020-03-13
18-45-53屏幕截图](/home/jimzeus/outputs/AANN/images/media/image197.png){width="2.1625in"
height="0.2777777777777778in"}

### LSTM单元

**Long Short-Term Memory**
Cell，**长短期记忆网络**单元，LSTM算是广义RNN的一种，其网络结构也与标准RNN类似，区别在于其微结构不同。其微结构如下图：

![A LSTM neural
network.](/home/jimzeus/outputs/AANN/images/media/image198.png){width="4.266666666666667in"
height="1.6027777777777779in"}

图中各个组成标示如下：

![](/home/jimzeus/outputs/AANN/images/media/image199.png){width="4.429861111111111in"
height="0.825in"}

神经网络层 逐点操作 向量传递 向量拼接 向量复制

![](/home/jimzeus/outputs/AANN/images/media/image200.png){width="0.2423611111111111in"
height="0.15347222222222223in"}：sigmoid函数

![](/home/jimzeus/outputs/AANN/images/media/image201.png)：tanh函数

![](/home/jimzeus/outputs/AANN/images/media/image202.png)：向量的逐点相乘

![](/home/jimzeus/outputs/AANN/images/media/image203.png)：向量的逐点相加

相较于标准RNN，LSTM多了一个细胞状态（Cell
State）的时间步骤输出（同时也是输入），即图中上面那条横线，用Ct表示。如下图：

![](/home/jimzeus/outputs/AANN/images/media/image204.png){width="5.290972222222222in"
height="1.6333333333333333in"}

从图中可以看出，在一个时间步骤内，Ct-1（上一个时间步骤输出的Ct）变到Ct需要经过两次逐点操作，分别被称为：

-   **遗忘门**（**Forget gate
    layer**），用于去除不需要的信息，ft的输出在0-1之间（根据sigmoid函数），因此可以决定是忘记（0）还是记住（1），或者介于两者之间。

![](/home/jimzeus/outputs/AANN/images/media/image205.png){width="4.075694444444444in"
height="1.257638888888889in"}

-   **输入门**（**Input gate layer**），用于加入新的信息

![](/home/jimzeus/outputs/AANN/images/media/image206.png){width="4.075694444444444in"
height="1.257638888888889in"}

Ct-1到Ct的更新首先需要逐点乘以ft，以便"忘记"之前的某些信息，然后再逐点加上某些新的信息：

![](/home/jimzeus/outputs/AANN/images/media/image207.png){width="4.427083333333333in"
height="1.3666666666666667in"}

最后，我们需要根据Ct决定输出的ht：

![](/home/jimzeus/outputs/AANN/images/media/image208.png){width="4.256944444444445in"
height="1.3138888888888889in"}

**参考**

[[https://zhuanlan.zhihu.com/p/42717426]{.underline}](https://zhuanlan.zhihu.com/p/42717426)

### GRU单元

LSTM有很多变种，其中最著名的是**GRU（Gated recurrent
unit）**，其微结构如下：

![A gated recurrent unit neural
network.](/home/jimzeus/outputs/AANN/images/media/image209.png){width="5.166666666666667in"
height="1.5944444444444446in"}

GRU和LSTM均是采用门机制的思想改造RNN的神经元，和LSTM相比，GRU更加简单，高效，且不容易过拟合，但有时候在更加复杂的场景中效果不如LSTM，算是RNN和LSTM在速度和精度上的一个折中方案。

注意力机制（Attention）
-----------------------

**注意力（Attention）**的概念最早在九几年就提出来了，简单来说就是通过不同的权重，重点关注输入的不同部分，从而得到对应的输出。这就像是人类的视觉中通过快速扫描全局图像，获得需要重点关注的目标区域。

广义的注意力（Attention）机制，既不是一个具体的结构，也不是一个微结构，而更象是一个概念和抽象的微结构，在不同的网络中可能会有不同的表现形式。

-   **理解**

我们可以这样来看待Attention机制：将Source中的构成元素想象成是由一系列的\<Key,Value\>数据对构成，此时给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。即可以将其本质思想改写为如下公式：

![](/home/jimzeus/outputs/AANN/images/media/image210.png){width="5.768055555555556in"
height="0.3875in"}

也可以将Attention机制看作一种**软寻址（Soft
Addressing）**，Source可以看作存储器内存储的内容，元素由地址Key和值Value组成，当前有个Key=Query的查询，目的是取出存储器中对应的Value值，即Attention数值。通过Query和存储器内元素Key的地址进行相似性比较来寻址，之所以说是软寻址，指的不像一般寻址只从存储内容里面找出一条内容，而是可能从每个Key地址都会取出内容，取出内容的重要性根据Query和Key的相似性来决定，之后对Value进行加权求和，这样就可以取出最终的Value值，也即Attention值。所以不少研究人员将Attention机制看作软寻址的一种特例，这也是非常有道理的。

-   **计算过程**

至于Attention机制的具体计算过程，如果对目前大多数方法进行抽象的话，可以将其归纳为两个过程：第一个过程是根据Query和Key计算权重系数，第二个过程根据权重系数对Value进行加权求和。而第一个过程又可以细分为两个阶段：第一个阶段根据Query和Key计算两者的相似性或者相关性；第二个阶段对第一阶段的原始分值进行归一化处理；这样，可以将Attention的计算过程抽象为如下图展示的三个阶段。

![](/home/jimzeus/outputs/AANN/images/media/image211.png){width="3.2194444444444446in"
height="2.845138888888889in"}\
在第一个阶段，可以引入不同的函数和计算机制，根据Query和某个Key\_i，计算两者的相似性或者相关性，最常见的方法包括：求两者的向量点积、求两者的向量Cosine相似性或者通过再引入额外的神经网络来求值，即如下方式：

![](/home/jimzeus/outputs/AANN/images/media/image212.png){width="3.6118055555555557in"
height="0.7548611111111111in"}

第一阶段产生的分值根据具体产生的方法不同其数值取值范围也不一样，第二阶段引入类似SoftMax的计算方式对第一阶段的得分进行数值转换，一方面可以进行归一化，将原始计算分值整理成所有元素权重之和为1的概率分布；另一方面也可以通过SoftMax的内在机制更加突出重要元素的权重。即一般采用如下公式计算：

![](/home/jimzeus/outputs/AANN/images/media/image213.png){width="2.48125in"
height="0.4486111111111111in"}

第二阶段的计算结果a\_i即为value\_i对应的权重系数，然后进行加权求和即可得到Attention数值：

![](/home/jimzeus/outputs/AANN/images/media/image214.png){width="3.329861111111111in"
height="0.2902777777777778in"}

通过如上三个阶段的计算，即可求出针对Query的Attention数值，目前绝大多数具体的注意力机制计算方法都符合上述的三阶段抽象计算过程。

-   RNN

Attention机制本身和具体NN的类型（CNN、RNN）无关，Attetion机制在2014年**机器翻译**的论文**《Neural
Machine Translation by Jointly Learning to Align and
Translate》**中重新成为热门，论文将**RNN
Encoder-Decoder**和Attention机制结合在一起。

-   Soft & Hard Attention

在Kelvin Xu等人2015年关于**图像描述（Image Caption）**的论文**《Show,
Attend and Tell》**中介绍了**硬注意力（Hard
Attention）**和**软注意力（Soft Attention）**的概念。这其中Soft
Attention即是传统的Attention。

-   Global & Local Attention

在2015年的另一篇关于**机器翻译**的论文**《Effective Approaches to
Attention-based Neural Machine Translation》**中提出了Global
Attention和Local Attention的概念。

-   Self-Attention

在2017年Google关于NLP的著名论文**《Attention is All You
Need》**中，给出了**Self-Attention**的概念，并且据此提出了**Transformer**结构，成为后来各种**BERT**的基础。

**参考**

[[https://www.zhihu.com/question/68482809/answer/264632289]{.underline}](https://www.zhihu.com/question/68482809/answer/264632289)

[[https://zhuanlan.zhihu.com/p/31547842]{.underline}](https://zhuanlan.zhihu.com/p/31547842)

注意力机制：

[[https://blog.csdn.net/yimingsilence/article/details/79208092]{.underline}](https://blog.csdn.net/yimingsilence/article/details/79208092)

5种Attention：

[[http://www.360doc.com/content/19/0804/02/46368139\_852849046.shtml]{.underline}](http://www.360doc.com/content/19/0804/02/46368139_852849046.shtml)

Attention 综述：

[[https://zhuanlan.zhihu.com/p/62136754]{.underline}](https://zhuanlan.zhihu.com/p/62136754)

NLP中的Attetion机制：

[[https://zhuanlan.zhihu.com/p/53682800]{.underline}](https://zhuanlan.zhihu.com/p/53682800)

自然语言中的Attention：

[[https://blog.csdn.net/hahajinbu/article/details/81940355]{.underline}](https://blog.csdn.net/hahajinbu/article/details/81940355)

论文

An Attentive Survey of Attention Models

[[https://arxiv.org/pdf/1904.02874.pdf]{.underline}](https://arxiv.org/pdf/1904.02874.pdf)

### CNN中的Attention

CNN中的**SENet**就是一种简单的注意力机制，参考《[[网络构成 \>
DNN/CNN微结构 \> SE结构]{.underline}](\l)》和《[[研究方向:图像 \>
图像分类 \> SENet]{.underline}](\l)》

### RNN中的Attention (201409)

在普通的RNN
Encoder-Decoder模型中，输入序列X和输出序列Y之间的传递仅仅通过一个中间向量C实现。这就带来如下问题：

-   C的长度是固定的，如果输入序列X太长的话，特别是比训练集中最初的句子长度还长时，模型的性能急剧下降。

-   把输入X编码成一个固定的长度，对于句子中每个词都赋予相同的权重，这样做是不合理的，比如，在机器翻译里，输入的句子与输出句子之间，往往是输入一个或几个词对应于输出的一个或几个词。因此，对输入的每个词赋予相同权重，这样做没有区分度，往往使得模型性能下降。**（这个问题的影响程度存疑，在X足够短的情况下，这种偏重应该是可以通过Encoder和Decoder中的权重体现出来的**）

这篇关于机器翻译（Machine Translation）的论文主要应用了两个主要技术：

-   **双向RNN**，其中每个RNN单元为**GRU**。

-   Attention机制

如下图：

![](/home/jimzeus/outputs/AANN/images/media/image215.png){width="2.498611111111111in"
height="3.3513888888888888in"}

传统RNN的Encoder-Decoder机制中，将整个句子的特征向量C作为Decoder的输入：

![2020-03-18
17-22-21屏幕截图](/home/jimzeus/outputs/AANN/images/media/image216.png){width="2.495833333333333in"
height="0.3548611111111111in"}

而Attention模型是使用所有特征向量的加权和，通过对特征向量的权值的学习，我们可以使用对当前时间片最重要的特征向量的子集Ci：

![2020-03-18 17-22-21屏幕截图
(复件)](/home/jimzeus/outputs/AANN/images/media/image217.png){width="2.6194444444444445in"
height="0.41180555555555554in"}

其中：

![2020-03-18 17-22-21屏幕截图
(另一个复件)](/home/jimzeus/outputs/AANN/images/media/image218.png){width="1.8409722222222222in"
height="2.0930555555555554in"}

其中a即是**Similarity函数**（文中叫alignment
model，对齐模型，即原语言中的词和翻译语言中词的对齐），表示注意力权重。a的计算方式有很多种，最简单最常用的是点积计算。

**论文**

Neural Machine Translation by Jointly Learning to Align and Translate

[[https://arxiv.org/pdf/1409.0473.pdf]{.underline}](https://arxiv.org/pdf/1409.0473.pdf)

### Soft & Hard attention (201502)

2015年发表的论文《Show, Attend and Tell: Neural Image Caption Generation
with Visual Attention》，在Image
Caption中引入了Attention，当生成第i个关于图片内容描述的词时，用Attention来关联与i个词相关的图片的区域。论文中使用了两种Attention
Mechanism，即**软注意力**（**Soft
Attention，就是传统的Attention**）和**硬注意力**（**Hard
Attention**）。Soft
Attention是参数化的，因此可导，可以被嵌入到模型中去，直接训练。梯度可以经过Attention
Mechanism模块，反向传播到模型其他部分。

相反，Hard Attention是一个随机的过程。Hard
Attention不会选择整个encoder的输出做为其输入，Hard
Attention会依概率Si来采样输入端的隐状态一部分来进行计算，而不是整个encoder的隐状态。为了实现梯度的反向传播，需要采用蒙特卡洛采样的方法来估计模块的梯度。

两种Attention
Mechanism都有各自的优势，但目前更多的研究和应用还是更倾向于使用Soft
Attention，因为其可以直接求导，进行梯度反向传播。

**论文**

Show, Attend and Tell: Neural Image Caption Generation with Visual
Attention

[[https://arxiv.org/pdf/1502.03044.pdf]{.underline}](https://arxiv.org/pdf/1502.03044.pdf)

代码

[[https://github.com/kelvinxu/arctic-captions]{.underline}](https://github.com/kelvinxu/arctic-captions)

### Global & Local Attention (201508)

在2015年关于**机器翻译**的论文《Effective Approaches to Attention-based
Neural Machine Translation》中提出了**全局注意力**（**Global
Attention，即传统的Attention**）和**局部注意力**（**Local
Attention**）的概念。

Global Attention有一个明显的缺点就是，每一次，encoder端的所有hidden
state都要参与计算，这样做计算开销会比较大，特别是当encoder的句子偏长，比如，一段话或者一篇文章，效率偏低。

**Local Attention**可以视作是一种介于**Soft Attention**和**Hard
Attention**之间的一种Attention方式，即把两种方式结合起来：

![](/home/jimzeus/outputs/AANN/images/media/image219.png){width="3.5493055555555557in"
height="2.4451388888888888in"}

**Local
Attention**首先会为decoder端当前的词，预测一个source端对齐位置（aligned
position）pt，然后基于pt选择一个窗口，用于计算背景向量ct。

但是，在实际应用中，**Global Attention**应用更普遍，因为Local
Attention需要预测一个位置向量p，这就带来两个问题：

1、当encoder句子不是很长时，相对Global Attention，计算量并没有明显减小。

2、位置向量pt的预测并不非常准确，这就直接计算的到的local
Attention的准确率。

**论文**

Effective Approaches to Attention-based Neural Machine Translation

[[https://arxiv.org/pdf/1508.04025]{.underline}](https://arxiv.org/pdf/1508.04025)

**代码**

[[https://github.com/lmthang/nmt.matlab]{.underline}](https://github.com/lmthang/nmt.matlab)

### **Transformer & Self-Attention (201706)**

参考《[[研究方向:NLP \> NN：基于Transformer \>
Transformer]{.underline}](\l)》

### Non-Local Network (201711）

Non-local Network是**FAIR**在17年的论文，其所介绍的Non-local
network算是CV领域里**自注意力机制**的核心。

Non-Local的操作的公式如下：

![](/home/jimzeus/outputs/AANN/images/media/image220.png){width="2.3826388888888888in"
height="0.5833333333333334in"}

x：输入，i和j代表不同的位置，x~i~是一个向量 ，维数同x的channel数

y：输出，和原图大小一致

f：计算两个点相似关系的函数

g：映射函数，将一个点映射为一个向量

由于输入和输出形状保持一致，使得Non-Local可以很容易的插入CNN和RNN之中。

论文中f的形式包括以下四个：

**Gaussian：**![](/home/jimzeus/outputs/AANN/images/media/image221.png){width="1.2951388888888888in"
height="0.2375in"}

**Embedded
Gaussian：**![](/home/jimzeus/outputs/AANN/images/media/image222.png){width="1.76875in"
height="0.26458333333333334in"}

**Dot
Production：**![](/home/jimzeus/outputs/AANN/images/media/image223.png){width="1.61875in"
height="0.2152777777777778in"}

**Concatenation：**![](/home/jimzeus/outputs/AANN/images/media/image224.png){width="2.423611111111111in"
height="0.20555555555555555in"}

上述公式中的θ和φ为两个embedding：

**θ(x~i~) = W~θ~x~i~**

**φ(x~i~) = W~φ~x~j~**

以Embedded Gaussian为例的整个流程如下图：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image225.jpeg){width="5.670833333333333in"
height="4.084027777777778in"}

论文原文中的流程图更加抽象一点，并且多了一个维度------原文是针对视频的，因此除了表示图片长宽的H和W维度，还有个表示时间步骤的T维度：

![](/home/jimzeus/outputs/AANN/images/media/image226.png){width="3.888888888888889in"
height="4.108333333333333in"}

跟**FC层**的关系：

如果

-   任意两点的相似性仅跟两点的位置有关，即 f(x~i~,x~j~)=W~ij~

-   g是identity函数，即g(x~i~)=x~i~

-   归一化系数为1。归一化系数跟输入无关，全连接层不能处理任意尺寸的输入。

那么此时Non-Local变为FC层。

跟**Transformer**的关系：

如果f采用dot production，即f(xi,xj) = θ(x~i~)^T^φ(x~i~) ,而：

θ(x~i~) = W~θ~x~i~，可得θ(x) = W~θ~x

φ(x~i~) = W~φ~x~j~，可得φ(x) = W~φ~x

从而得到：

y=softmax(x^T^ W~θ~^T^ W~φ~ x)

此时Non-Local等同于**Transformer**，也就是说，Transformer可以被视作是Non-Local的一个特例。

**论文**

[[https://arxiv.org/pdf/1711.07971.pdf]{.underline}](https://arxiv.org/pdf/1711.07971.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/33345791]{.underline}](https://zhuanlan.zhihu.com/p/33345791)

[[https://zhuanlan.zhihu.com/p/102984842]{.underline}](https://zhuanlan.zhihu.com/p/102984842)

GAN结构
-------

GAN于2014年由Ian
Goodfellow提出，GAN的贡献主要是其理念及对应的网络结构。GAN是深度学习领域的一个重要组成部分。

**论文**

[[https://arxiv.org/pdf/1406.2661.pdf]{.underline}](https://arxiv.org/pdf/1406.2661.pdf)

综述**参考**

[[https://zhuanlan.zhihu.com/p/58812258]{.underline}](https://zhuanlan.zhihu.com/p/58812258)

[[https://zhuanlan.zhihu.com/p/110581201]{.underline}](https://zhuanlan.zhihu.com/p/110581201)

激活函数（Activation Function）
-------------------------------

对于某个隐藏节点的计算，该节点的激活值分为2步：先是一步线性变换，之后又被作用了一个函数，即激活函数。

激活函数（Activation
Function）的作用是，增加神经网络的非线性。如果没有激活层，因为所有层都是线性关系，输出都是输入的线性组合，所有的隐藏层都会变得没有意义。如果使用了激活函数，激活函数引入了非线性，则神经网络可以逼近任何非线性函数。

下图是**sigmoid、tanh、softplus**和**softsign**激活函数的曲线图（注意x和y轴不成比例，曲线图被拉高了）：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\81c8426c90abe69a991cafb1421d4b1.png](/home/jimzeus/outputs/AANN/images/media/image227.png){width="4.325in"
height="3.2604166666666665in"}

下面是各种激活函数的名称、公式、图像：

![](/home/jimzeus/outputs/AANN/images/media/image228.png){width="6.155555555555556in"
height="7.228472222222222in"}

![](/home/jimzeus/outputs/AANN/images/media/image229.png){width="5.7652777777777775in"
height="6.156944444444444in"}

**参考**

[[https://en.wikipedia.org/wiki/Activation\_function]{.underline}](https://en.wikipedia.org/wiki/Activation_function)

### sigmoid函数（logistic函数）

![](/home/jimzeus/outputs/AANN/images/media/image230.png){width="1.6958333333333333in"
height="0.6965277777777777in"}

也叫**Logistic函数**，输出范围在（0，1）之间，可以用于表示概率，或者用作数据的归一化（Normalization）。Sigmoid函数有2个问题：

-   软饱和性：当x趋于无穷时，f(x)的两侧导数逐渐趋向于0，导致反向传播的梯度非常小，网络很难得到有效训练，这种现象被称为**梯度消失**。

-   偏置现象：Sigmoid的输出均大于0，使得其输出不是0均值。

因此现在sigmoid经常用在网络最后一层，作为输出层二分类使用，很少用在隐藏层。

### tanh函数

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\368508be9f3401992c38df80b44a875.png](/home/jimzeus/outputs/AANN/images/media/image231.png){width="2.071527777777778in"
height="0.8708333333333333in"}

相较于sigmoid函数，tanh函数的输出均值为0，使得其收敛更快，但是tanh函数同样具有软饱和性，从而发生梯度消失。

### ReLU函数系列

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\c6d03f2c8235ec83e52229fd1e5e576.png](/home/jimzeus/outputs/AANN/images/media/image232.png){width="2.432638888888889in"
height="1.1145833333333333in"}

**线性整流单元、修正线性单元（Recitified Linear
Units）**，它的优点有几个：

-   在x\>0时不存在饱和问题，这让我们无需依赖逐层预训练，在正区间坚决了梯度消失问题

-   计算速度非常快，只需判断是否大于0

-   收敛速度非常快

缺点有几个：

-   然而随着训练推进，部分输入会落入硬饱和区，导致对应权重无法更新，这被称为"**神经元死亡**"。

-   与sigmoid类似，ReLU的输出均值也大于0，所以偏移现象和神经元死亡共同影响网络的收敛性。

ReLU函数有一系列变体，先上曲线图：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\543532983bb693732f843c4501e869a.png](/home/jimzeus/outputs/AANN/images/media/image233.png){width="4.127083333333333in"
height="2.9208333333333334in"}

#### ReLU6函数

类似ReLU函数，区别是当x\>6时函数值始终取6。

#### Leaky-ReLU函数

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\adad38cf3e9097e1917ce92e99384fb.png](/home/jimzeus/outputs/AANN/images/media/image234.png){width="3.084722222222222in"
height="0.9423611111111111in"}

**Leaky-ReLU**，为了避免ReLU在x\<0时的神经元死亡现象，添加了一个参数α，通常α为0.01。虽然理论上来看，Leaky-ReLU有ReLU所有的优点，而且还不会有Dead
ReLU问题，但实际使用中并不能证明ELU总是好于ReLU

#### ELU函数

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\d73e3654918082eaab59f1c1128964d.png](/home/jimzeus/outputs/AANN/images/media/image235.png){width="3.3513888888888888in"
height="1.1097222222222223in"}

ELU（Exponential Linear
Units）函数，它结合了sigmoid和ReLU函数，左侧软饱和，右侧无饱和。和ReLU一样，右侧线性部分使得ELU能缓解梯度消失，除此之外相较于ReLU还有两个优点：

-   左侧软饱和能让对ELU对输入变化或噪声更鲁棒。

-   ELU的输出均值接近于0，所以收敛速度更快。

相较于ReLU的缺点是计算量大。同Leaky
ReLU一样，虽然理论上优于ReLU，但实际使用中并不能证明ELU总是好于ReLU。

#### softReLU / softplus

也叫Smooth ReLU函数，也是对ReLU的平滑近似：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image236.png){width="1.525in"
height="0.22708333333333333in"}

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image237.png){width="1.1284722222222223in"
height="0.5645833333333333in"}

### softmax函数

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\06aba7e098f5ea3d704d999fb0023c6.png](/home/jimzeus/outputs/AANN/images/media/image238.png){width="1.7666666666666666in"
height="0.9638888888888889in"}

Softmax函数通常用在**分类问题**最后一层的激活函数，先对每个j进行指数幂运算变成非负，然后除以所有项之和归一化。其特点是输出层所有节点的激活值之和为1，用于表示属于该分类的概率。

### softsign函数

Softsign函数来自sign函数（即正为1，负为-1），只是在分母加了个1，公式为：

![](/home/jimzeus/outputs/AANN/images/media/image239.png){width="2.0055555555555555in"
height="0.7243055555555555in"}

就像 Tanh 一样，Softsign 是反对称、去中心、可微分，并返回-1 和 1
之间的值。其更平坦的曲线与更慢的下降导数表明它可以更高效地学习。另一方面，导数的计算比
Tanh 更麻烦。

损失函数（Loss Function）
-------------------------

监督式机器学习中，用于衡量预测值和真实值之间的误差的函数，被称为损失函数（Loss
Function），损失函数的值越小，说明预测值和真实值之间越接近，模型的预测能力越好。

简单的损失函数大致可分为两类：**分类问题**的损失函数，和**回归问题**的损失函数

**MSE**和**MAE**是最常用的回归损失函数，**交叉熵损失（cross entropy
loss）**则是最常用的分类问题损失函数。Softmax
loss只是交叉熵损失的另一个名字。

**参考**

[[https://gombru.github.io/2018/05/23/cross\_entropy\_loss/]{.underline}](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

### MSE（均方误差）

**均方误差MSE（Mean Squared
Error）**，或者叫**L2损失**，预测值yp和真实值y之差平方的平均值，通常用来做**回归问题**的损失函数。

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\2418703c7f833865e81360b778c08c4.png](/home/jimzeus/outputs/AANN/images/media/image240.png){width="2.0076388888888888in"
height="0.7361111111111112in"}

### RMSE

**均方根误差**，(**Root Mean Squred Error**)，是MSE开平方。

![](/home/jimzeus/outputs/AANN/images/media/image241.png){width="2.263888888888889in"
height="0.7194444444444444in"}

### MAE（平均绝对误差）

**平均绝对误差MAE（Mean Absolute
Error）**，通常用来做**回归问题**的损失函数

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\fa3ec6816f737cd831753150153d298.png](/home/jimzeus/outputs/AANN/images/media/image242.png){width="2.140972222222222in"
height="0.9in"}

由于均方误差（MSE）在误差较大点时的损失远大于平均绝对误差（MAE），它会给异常值赋予更大的权重，模型会全力减小异常值造成的误差，从而使得模型的整体表现下降。

所以当训练数据中含有较多的异常值时，平均绝对误差（MAE）更为有效。当我们对所有观测值进行处理时，如果利用MSE进行优化则我们会得到所有观测的均值，而使用MAE则能得到所有观测的中值。与均值相比，中值对于异常值的鲁棒性更好，这就意味着**平均绝对误差**对于异常值有着比**均方误差**更好的鲁棒性。

但MAE也存在一个问题，特别是对于神经网络来说，它的梯度在极值点处会有很大的跃变，及时很小的损失值也会长生很大的误差，这很不利于学习过程。为了解决这个问题，需要在解决极值点的过程中动态减小学习率。MSE在极值点却有着良好的特性，及时在固定学习率下也能收敛。MSE的梯度随着损失函数的减小而减小，这一特性使得它在最后的训练过程中能得到更精确的结果

### MAPE

**平均绝对百分比误差（Mean Absolute Percentage Error）**，公式如下：

![](/home/jimzeus/outputs/AANN/images/media/image243.png){width="2.2152777777777777in"
height="0.4534722222222222in"}

为误差占真实值的百分比，当真实值有数据为0时不可用。

### sMAPE

**对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）**：

![](/home/jimzeus/outputs/AANN/images/media/image244.png){width="2.6840277777777777in"
height="0.5166666666666667in"}

当真实值和预测值同时为0的时候不可用。

### MASE

### MSIS

### ND

### NRMSE

### OWA

### Cross-entropy Loss（交叉熵损失）

**交叉熵损失（Cross Entropy
loss）**通常用于做**分类问题**的损失函数。**Logistic
Loss（逻辑损失）**、**Softmax Loss**这些都是交叉熵损失的别名而已。

-   多分类问题

比如通常的**图像分类**任务，有若干可能的分类，而每张图片只预测一个分类，所有分类的预测值之和为1。多分类问题的损失函数，通常为：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\f4a39b5b6f1f0a05a941903ca253d2a.png](/home/jimzeus/outputs/AANN/images/media/image245.png){width="2.2180555555555554in"
height="0.7451388888888889in"}

其中n为类别数，而yi为真实值，y\^i为预测值。当y为**one-hot标签**时（分类任务通常都是如此），因为除了真实标签之外其余项都为0，这个函数只剩下了一项：

*loss=-log(y\^i) *

对应一个batch的损失，则为：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\4d3a11c4e91ffc69287d63491d7793b.png](/home/jimzeus/outputs/AANN/images/media/image246.png){width="2.6756944444444444in"
height="0.8090277777777778in"}

m为batch内样本数，n为类别数

-   二分类问题

而对于简单的二分类问题，比如判断图片中是否有枪支，其损失函数由上面的多分类问题的损失函数变为：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\5b70a4a43a4d0ce210deed0efb8f280.png](/home/jimzeus/outputs/AANN/images/media/image247.png){width="3.348611111111111in"
height="0.5305555555555556in"}

其中y为真实值，y\^为预测值，上式中的y和(1-y)相当于多分类中的yi，其和为1。当真实值为**是**的时候只有前一项，为**否**的时候则只有后一项。

-   多标签分类问题

多标签分类问题，在实际应用中有点介于**图像分类**和**目标检测**任务之间，需要在一张图片中得到若干物体的预测值，因此对于多标签分类问题，一张图片的各个类别预测值之和大于1（因为一张图片可能有若干物体）。

其**单个类别**的损失函数可以被视为一个二分类问题的损失函数：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\5b70a4a43a4d0ce210deed0efb8f280.png](/home/jimzeus/outputs/AANN/images/media/image247.png){width="3.348611111111111in"
height="0.5305555555555556in"}

对应每个batch的损失为：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\0ed36338d234dd108bec229b8071c4c.png](/home/jimzeus/outputs/AANN/images/media/image248.png){width="4.309027777777778in"
height="0.7909722222222222in"}

m为batch内样本数，n为类别数

### Softmax Loss

严格来说，并没有一个被称为softmax loss的学术定义。Softmax
loss指的就是softmax + cross entropy loss。

通常情况下，**分类问题**的最后一层FC层用softmax函数作为激活函数（或者有的人将之视为两层），FC层出来的是特征值，再经过softmax出来的是分类概率，概率被作为输入传给损失函数cross
entropy loss。通常情况下softmax和cross
entropy是成对出现的，这两个函数组合被定义为softmax loss。

### Triplet Loss（FaceNet）

### Contrastive Loss (TODO)

### Center Loss（2016）

Center loss是2016年的一篇论文，提出了一种新的损失函数center
loss，用于提高open-set情况下的人脸识别的准确率。

Close-set的人脸识别相当于一个分类问题，通常采用softmax
loss作为损失函数，这在close-set中性能良好，但在open-set情况下，模型性能会急剧下降。

一个直观的感觉是：如果模型学到的特征判别度更高，那么遇到没见过的数据时，泛化性能会比较好。为了使得模型学到的特征判别度更高，论文提出了一种新的辅助损失函数center
loss，之说以说是辅助损失函数是因为新提出的损失函数需要结合softmax交叉熵一起使用，而非替代后者。

文中对每个类别都定义了一个center，其维度与特征值的维度相同，这些center是可以被学习的。

Softmax loss的定义为：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561633636(1).png](/home/jimzeus/outputs/AANN/images/media/image249.png){width="3.066666666666667in"
height="0.8138888888888889in"}

Center loss的函数定义为：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561630819(1).png](/home/jimzeus/outputs/AANN/images/media/image250.png){width="2.4756944444444446in"
height="0.8583333333333333in"}

总的损失函数为softmax loss和center
loss的加权和，其中λ被用来平衡两个损失函数：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561639161(1).png](/home/jimzeus/outputs/AANN/images/media/image251.png){width="4.545138888888889in"
height="1.1229166666666666in"}

使用两个损失函数的加权和是因为：

-   如果没有center
    loss，各个类别的类内变化会比较大（即特征判别度低，feature不够discriminative，这样就不能很好的面对陌生数据），也就是说center
    loss用于保证特征值是discriminative的

-   而如果没有softmax loss，那么center和特征值都会降到0（这样center
    loss会非常小），各类别的区分度不够，也就是说softmax
    loss是用于保证特征值separable的。

文中使用了一个修改过的LeNet提取特征，这个LeNet输出的特征只有2个，这是为了能在二维坐标系中直观的表示出来，下图为通过传统的softmax
loss得到的10个类别的样本的特征值在二维坐标系中的分布（x轴和y轴分别为两个特征值），左边为训练集，右边为测试集：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\83f46b990091ec9fcaa7c556e6a22fd.png](/home/jimzeus/outputs/AANN/images/media/image252.png){width="4.420833333333333in"
height="2.720833333333333in"}

而以下是在不同λ的情况下，用组合损失函数习得的特征值分布：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561640075(1).png](/home/jimzeus/outputs/AANN/images/media/image253.png){width="4.638194444444444in"
height="3.38125in"}

而contrastive loss和triplet
loss，这两个损失函数的缺点是在样本空间变大的时候，样本对（sample
pair）和样本三元组（sample triplet）的量会变得非常巨大

**论文**

A Discriminative Feature Learning Approach for Deep Face Recognition

### A-Softmax Loss（SphereFace）

**A-Softmax Loss**是在SphereFace论文中提出的一种损失函数，A代表Angular。

对于一个2分类的情况，最后一个FC层的2维输出（没过softmax激活之前）分别是W1\*x+b1和W2\*x+b2，预测分类是这两者之中大的那个。也就是说softmax
loss函数的判断边界是![C:\\Users\\AW\\AppData\\Local\\Temp\\1561712765(1).png](/home/jimzeus/outputs/AANN/images/media/image255.png){width="1.3972222222222221in"
height="0.16319444444444445in"}。如果我们令![C:\\Users\\AW\\AppData\\Local\\Temp\\1561712917(1).png](/home/jimzeus/outputs/AANN/images/media/image256.png){width="0.9729166666666667in"
height="0.1451388888888889in"}且![C:\\Users\\AW\\AppData\\Local\\Temp\\1561712947(1).png](/home/jimzeus/outputs/AANN/images/media/image257.png){width="0.7604166666666666in"
height="0.1909722222222222in"}，则判断边界就变为![C:\\Users\\AW\\AppData\\Local\\Temp\\1561713407(1).png](/home/jimzeus/outputs/AANN/images/media/image258.png){width="1.7236111111111112in"
height="0.175in"}，其中![C:\\Users\\AW\\AppData\\Local\\Temp\\1561713596(1).png](/home/jimzeus/outputs/AANN/images/media/image259.png){width="0.19583333333333333in"
height="0.22361111111111112in"}为Wi和x的夹角，于是判断边界变为仅依赖于两个夹角![C:\\Users\\AW\\AppData\\Local\\Temp\\1561713701(1).png](/home/jimzeus/outputs/AANN/images/media/image260.png){width="0.23541666666666666in"
height="0.25416666666666665in"}和![C:\\Users\\AW\\AppData\\Local\\Temp\\1561713709(1).png](/home/jimzeus/outputs/AANN/images/media/image261.png){width="0.22847222222222222in"
height="0.2604166666666667in"}，直观的理解，就是x离哪个Wi更近，则被分为哪个类。

A-Softmax
Loss还加入了一个正整数m（m\>=1）来量化的控制判断边界，于是类别1的判断边界收缩为![C:\\Users\\AW\\AppData\\Local\\Temp\\1561714415(1).png](/home/jimzeus/outputs/AANN/images/media/image262.png){width="1.5145833333333334in"
height="0.17430555555555555in"}，类别2的为![C:\\Users\\AW\\AppData\\Local\\Temp\\1561718789(1).png](/home/jimzeus/outputs/AANN/images/media/image263.png){width="1.5631944444444446in"
height="0.15416666666666667in"}。更高的m会约束每个类别的角度θ变得更小，从而使得**inter-class**距离更大，而**intra-class**距离更小。我们可以得出m的下限，以满足对open-set人脸识别的准则：**最大的intra-class距离要小于最小的inter-class距离**。

下图中为这二分类神经网络在特征值为二维的情况下（为了方便显示），使用三种不同的损失函数在欧式空间和角空间得到的结果：最左边是原始的sotfmax
loss，中间两幅图是改进后的softmax
loss（即判断边界改为![C:\\Users\\AW\\AppData\\Local\\Temp\\1561713407(1).png](/home/jimzeus/outputs/AANN/images/media/image258.png){width="1.7236111111111112in"
height="0.175in"}），最右两幅图是增加了控制变量m的结果。

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561784057.png](/home/jimzeus/outputs/AANN/images/media/image264.png){width="5.231944444444444in"
height="1.2965277777777777in"}

A-Softmax
Loss普及到三维特征值，则为一个球面中，二个不同的向量形成的圆分别代表两个不同的分类。下图为二分类神经网络在特征值为二维及三维的情况下，三种不同类型的损失函数在欧式空间和角空间形成的分类结果：两种颜色表示不同分类，上面三个是特征值为2D，下面三个是特征值为3D，最左边两个是使用欧式边缘损失函数，比如contrastive
loss，triplet loss（facenet），center loss；中间两个是改进后的softmax
loss；最右边两个则是A-Softmax Loss的情形：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1561823213.png](/home/jimzeus/outputs/AANN/images/media/image265.png){width="4.651388888888889in"
height="4.235416666666667in"}

### Focal Loss (RetinaNet)

Focal
Loss是在RetinaNet中被提出来的，是为了应对在单步检测网络（比如SSD和YOLO）中存在的问题：

-   由于正负样本比例严重失衡（网络产生的bbox预测大部分为false
    negative），大量的negative example的损失淹没了positive的loss。

-   大多negative
    example不在前景和背景的过渡区域上，分类很明确(这种易分类的negative称为easy
    negative)，训练时对应的背景类score会很大，换个角度看就是单个example的loss很小，反向计算时梯度小。梯度小造成easy
    negative
    example对参数的收敛作用很有限，我们更需要loss大的对参数收敛影响也更大的example，即hard
    positive/negative example。

这篇论文调整了loss function，通过增加权重系数来解决这个问题。

传统的二分类交叉熵损失函数CE为：

![fl
(第3个复件)](/home/jimzeus/outputs/AANN/images/media/image266.png){width="3.1555555555555554in"
height="0.6131944444444445in"}

如果定义p~t~为：

![fl
(另一个复件)](/home/jimzeus/outputs/AANN/images/media/image267.png){width="2.5284722222222222in"
height="0.7402777777777778in"}

则有：

![fl
(复件)](/home/jimzeus/outputs/AANN/images/media/image268.png){width="2.277083333333333in"
height="0.21458333333333332in"}

通过在CE上增加权重系数，Focal Loss函数FL被定义为：

![fl](/home/jimzeus/outputs/AANN/images/media/image269.png){width="2.2090277777777776in"
height="0.26875in"}

其中γ是个大于0的值， α~t~ 是个\[0，1\]间的小数，γ和α~t~都是固定值，不参与训练。

-   无论是前景类还是背景类，p~t~ 越大，权重(1-p~t~)就越小。也就是说easy
    example可以通过权重进行抑制。换言之，当某样本类别比较明确些，它对整体loss的贡献就比较少；而若某样本类别不易区分，则对整体loss的贡献就相对偏大。这样得到的loss最终将集中精力去诱导模型去努力分辨那些难分的目标类别，于是就有效提升了整体的目标检测准度。

-   α~t~用于调节positive和negative的比例，前景类别使用 α~t~时，对应的背景类别使用1- α~t~；

-   γ的作用也是抑制easy example对整体loss的贡献

-   γ和 α~t~的最优值是相互影响的，所以在评估准确度时需要把两者组合起来调节。作者在论文中给出γ=2、 α~t~=0.25时，ResNet-101+FPN作为backbone的结构有最优的性能。

假设一个easy正样本，p为0.8（p~t~=p=0.8），另外有个hard正样本，p为0.4（p~t~=p=0.8），在CE中，两者对最终loss的贡献大约是1：4（log0.8约等于1/4的log0.4），而在FL中，通过(1-p~t~)^γ^，两者的贡献比例被放大为：(1-0.8)^2^：(1-0.4)^2^\*4
，约等于1：36

**论文**

[[https://arxiv.org/pdf/1708.02002.pdf]{.underline}](https://arxiv.org/pdf/1708.02002.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/59910080]{.underline}](https://zhuanlan.zhihu.com/p/59910080)

优化方法（Optimizer）
---------------------

### 梯度下降法（GD）

**梯度下降法（Gradient
Descent，GD）**，最基础的参数优化算法，计算数据集中样本的梯度，然后执行决策（沿着梯度相反的方向更新参数），重复这个步骤从而逐渐靠近最优解。

根据每次更新前计算的样本数量不同，GD可以分为：

#### SGD

**Stochastic GD**，**随机梯度下降法**，也叫One-Sample
GD，每次计算一个随机样本，然后更新参数。

这种算法的优点包括：

-   易于理解和实现

-   快速的更新频率使得在某些问题上学习速度很快

-   随机样本的噪声使得模型更易避开局部最小值

缺点：

-   更新次数太过频繁导致计算量更大

-   随机样本的噪声同样会导致不易稳定在全局最小值

**混淆注意！**

由于实际应用中较少使用One-Sample GD，而经常使用Mini-Batch
GD，**SGD**这个词经常被用于指代**Mini-Batch GD**。

#### BGD

Batch Gradient Descent，每次计算所有的样本，然后更新参数。

这种算法的优点是：

-   计算的效率比SGD更高

-   更少的更新次数带来更稳定的收敛

-   计算更加并行，方便利用硬件的并行能力

缺点是：

-   会落入局部极小值的点出不来

-   需要额外的算力来将所有样本的loss相加

-   对内存要求太高，通常需要将整个训练集加载到内存

-   对于大型数据集来说，训练速度会变得很慢

#### MBGD

MBGD，Mini-batch
GD，介于SGD和BGD之间，将训练集随机分割为若干个Batch，每次计算一个Batch，然后更新参数。

可以看出来MBGD综合了SGD和BGD的优缺点，优点是：

-   比SGD计算更有效率，更快的收敛速度

-   比BGD更鲁棒的收敛，避免了局部极小值

-   在利用了硬件并行算力的同时又无须将整个训练集加载进内存

    缺点：

<!-- -->

-   需要一个额外的超参数"mini-batch size"

-   也综合了SGD和BGD的缺点，但总体来说比两者更平衡

    MBGD也是最常用的GD方法，同时也是最常用的优化方法，经常被误称为**SGD**。

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562834695(1).png](/home/jimzeus/outputs/AANN/images/media/image270.png){width="3.2020833333333334in"
height="2.5236111111111112in"}

### 动量法（Momentum）

Momentum，动量法，公式如下：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562833620(1).png](/home/jimzeus/outputs/AANN/images/media/image271.png){width="1.1694444444444445in"
height="0.33194444444444443in"}

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562833628(1).png](/home/jimzeus/outputs/AANN/images/media/image272.png){width="1.051388888888889in"
height="0.1875in"}

相较于SGD，增加了一个惯性的概念![C:\\Users\\AW\\AppData\\Local\\Temp\\1562842552(1).png](/home/jimzeus/outputs/AANN/images/media/image273.png){width="0.22291666666666668in"
height="0.1451388888888889in"}，动量法给人的感觉就好像是小球在函数曲面上滚动，滚动方向，也就是每一次更新的方向都是由惯性（之前的梯度方向残留）和受力方向（本次的梯度方向）组合而成。而相较于此，之前的GD则像是小球每次更新到一个新的位置之后立刻静止，只受当前的梯度方向影响。

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562834612(1).png](/home/jimzeus/outputs/AANN/images/media/image274.png){width="3.259027777777778in"
height="2.5756944444444443in"}

### NAG

Nesterov Accelerated
Gradient，NAG和动量法类似，也是"历史梯度+当前梯度"的方法，其区别在于其"当前梯度"采用的不是当前点的梯度，而是当前点在历史梯度上前进了之后的超前点的梯度，见下图：

![è¿éåå¾çæè¿°](/home/jimzeus/outputs/AANN/images/media/image275.jpeg){width="3.6569444444444446in"
height="1.9673611111111111in"}

与之相比，动量法如下：

![è¿éåå¾çæè¿°](/home/jimzeus/outputs/AANN/images/media/image276.jpeg){width="4.124305555555556in"
height="1.6631944444444444in"}

上两图中，A为起始点，AB为起始点的梯度方向，B为当前点，下一个点的位置分别为D（上图，NAG）和C（下图，动量法），而AD/AC都是AB和另一个向量之和，区别在于下图的动量法使用的是当前点B的梯度，而上图的NAG使用的是超前点C的梯度方向CD。

### AdaGrad

在有关学习率的技巧中，有一种被称为学习率衰减的方法，随着学习的进行，使学习率逐渐减小。而AdaGrad方法则进一步发展了这个办法，针对每一个参数，适当的调整学习率，于此同时进行学习。

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562842385(1).png](/home/jimzeus/outputs/AANN/images/media/image277.png){width="2.145138888888889in"
height="1.2069444444444444in"}

公式中W表示权重，![C:\\Users\\AW\\AppData\\Local\\Temp\\1562842472(1).png](/home/jimzeus/outputs/AANN/images/media/image278.png){width="0.2048611111111111in"
height="0.21805555555555556in"}表示梯度，![C:\\Users\\AW\\AppData\\Local\\Temp\\1562842514(1).png](/home/jimzeus/outputs/AANN/images/media/image279.png){width="0.1423611111111111in"
height="0.18472222222222223in"}表示学习率，新的变量h保存了之前所有的梯度的平方和。这样有以下效果：

-   变动大的参数的学习率下降得更快

-   随着梯度的更新，学习率将逐渐变小

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562842881(1).png](/home/jimzeus/outputs/AANN/images/media/image280.png){width="2.842361111111111in"
height="2.202777777777778in"}

### RMSprop

AdaGrad有一个问题，就是学习率总会逐渐地变小（因为h会持续的变大），导致最终学习率会降到0而无法学习，RMSprop相较于AdaGrad的改进就是添加了一个衰减率，使得h不会因为累积而变得太大。

### Adam

Adam方法融合了Momentum和AdaGrad两种方法：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562842915(1).png](/home/jimzeus/outputs/AANN/images/media/image281.png){width="3.1708333333333334in"
height="2.5055555555555555in"}

标准化方法（Normalization）
---------------------------

**标准化**，或者**规范化**，**归一化**（**Normalization**）是神经网络中一种常见的处理方式，可以将数据标准化，其目的是为了增强网络的泛化能力，降低网络的**过拟合**。这个词本身的定义来源于统计学。

Normalization的目的是将数据标准化。具体来说，是将数据按行、按列、或者某种维度，映射到一个特定的区间（比如-1到1）或者某种特定分布里（比如均值为0,方差为1），使得样本间、批次间、或者该维度间的差距变小，更加趋同。

因此Normalization指代两种概念，一种仅指映射的算法，比如L1N、L2N、min-max
normalization、Z-score
normalization，另一种则是指代更具体的方法，不仅包括算法，还包括所取的维度，比如：

-   BN：其标准化的维度A为单个批次内，从而使得批次间的数据更加趋同。

-   LN：其标准化的维度为单个样本内的不同通道，缩小样本间的差异。

-   IN：标准化的维度为单个通道内的，缩小通道间的差异。

-   GN：介于IN和LN之间的一种标准化方案

Normalization对于数据的处理还可以按照处理的对象（以及所适用的场景）分为两种处理过程：

-   第一种Normalization是指对于某个多特征的机器学习数据集来说，将**输入数据**进行预处理时进行的操作，对数据集的各个特征分别进行处理，主要包括采用的算法为**min-max
    normalization、Z-score normalization、
    log函数转换、atan函数转换**等。

-   第二种Normalization则一般应用在深度学习中，对于每个样本缩放到单位范数（每个样本的范数为1），主要采用的算法为**L1-normalization、L2-normalization、Z-score
    normalization**等。

换句话说，第一种类型的Normalization的目的是**缩小特征间的差异**，也叫**Feature
Normalization**，或者**Feature
Scaling**。在将输入的多种特征进行趋同化和无量纲化，比如将两个输入特征，第一个高度特征，范围是-10000米到+30000米，另一个角度特征，范围是0度到3度之间，如果不进行标准化，则这两个特征之间的数值差异太大。

而深度学习中通常所指的Normalization通常指的是第二种，其目的是**缩小不同维度间的特征值之间的差异**。

综上所述：

-   Normalization

    -   算法：min-max、z-score、L1N、L2N等

    -   场景和处理对象

        -   机器学习中，输入数据预处理：各种算法都有

        -   深度学习中，特征值的不同维度：一般用z-score

            -   BN：图像处理常用（CNN）

            -   LN：NLP常用（RNN、Transformer）

            -   IN

            -   GN

**参考**

一文搞定深度学习中的规范化BN,LN,IN,GN,CBN

[[https://zhuanlan.zhihu.com/p/115949091]{.underline}](https://zhuanlan.zhihu.com/p/115949091)

### LRN (AlexNet)

**局部响应归一化层（Local Response Normalization），**LRU的定义如下：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\15f715597a7cb04271acde5d43e3e5a.png](/home/jimzeus/outputs/AANN/images/media/image282.png){width="5.768055555555556in"
height="1.2375in"}

具体不做解释，可以参考AlexNet的论文。现今的神经网络中已经很少用LRN，标准化大都通过BN层等来实现。

### Min-max Normalization

对原始数据线性变换，映射到\[0,1\]区间，公式如下：

x\* = (x-min) / (max-min)

### Z-score Normalization

也叫标准差标准化，这种方法根据原始数据的均值（mean）μ和标准差（standard
deviation）σ进行数据的标准化。经过处理的数据符合标准正态分布，即均值为0，标准差为1，其转化函数为：

x\* = (x - μ ) / σ

这种标准化算法普遍应用在深度学习中不同的Normalization中。

### L1 Normalization

L1
Normalization，L1标准化，即向量中的每个元素除以**向量的L1范数**（向量各元素的绝对值之和）

### L2 Normalization

L2
Normalization，L2标准化，即向量中的每个元素除以**向量的L2范数**（向量各元素的平方和的平方根）。

### Batch Normalization (201502)

**Batch
Normalization**，**BN**，**BatchNorm**，**批标准化**，是目前常用的标准化方法，尤其在CNN中。优点是可以增加学习速度，不那么依赖初始值，抑制过拟合。

BN是对每个batch中各个样本的同一特征值做**标准差标准化（Z-score）**，即使数据满足标准正态分布（均值为0方差为1）。而在CNN中，由于有channel的存在，是对每个batch中各个样本的各个channel的同一特征值做**标准差标准化**。

BN的提出是基于小批量随机梯度下降（mini-batch
SGD）的。随机梯度下降的缺点是对参数比较敏感，较大的学习率和不合适的初始化值均有可能导致训练过程中发生梯度消失或者梯度爆炸的现象的出现。BN的出现则有效的解决了这个问题。

**参考**

[[https://zhuanlan.zhihu.com/p/54171297]{.underline}](https://zhuanlan.zhihu.com/p/54171297)

[[https://www.cnblogs.com/guoyaohua/p/8724433.html]{.underline}](https://www.cnblogs.com/guoyaohua/p/8724433.html)

**论文**

[[https://arxiv.org/pdf/1502.03167.pdf]{.underline}](https://arxiv.org/pdf/1502.03167.pdf)

### Layer Normalization (201607)

BN并不适用于RNN等动态网络和batchsize较小的时候。**Layer
Normalization（LN，LayerNorm，层标准化）**的提出有效的解决BN的这两个问题，LayerNorm也是个比较常用的标准化方法，通常用在NLP中（RNN和Transformer）。

LN和BN不同点是归一化的维度是互相垂直的，如下图所示。图中 N表示样本轴， C表示通道轴，F是每个通道的特征数量。BN如右侧所示，它是取不同样本的同一个通道的特征做归一化；LN则是如左侧所示，它取的是同一个样本的不同通道做归一化。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image283.jpeg){width="4.118055555555555in"
height="1.698611111111111in"}

左为LN，右为BN

参考

[[https://zhuanlan.zhihu.com/p/54530247]{.underline}](https://zhuanlan.zhihu.com/p/54530247)

**论文**

[[https://arxiv.org/pdf/1607.06450.pdf]{.underline}](https://arxiv.org/pdf/1607.06450.pdf)

### Instance Normalization (201607)

对于[图像风格迁移](https://zhuanlan.zhihu.com/p/55948352)这类的注重每个像素的任务来说，每个样本的每个像素点的信息都是非常重要的，于是像[BN](https://zhuanlan.zhihu.com/p/54171297)这种每个批量的所有样本都做归一化的算法就不太适用了，因为BN计算归一化统计量时考虑了一个批量中所有图片的内容，从而造成了每个样本独特细节的丢失。同理对于[LN](https://zhuanlan.zhihu.com/p/54530247)这类需要考虑一个样本所有通道的算法来说可能忽略了不同通道的差异，也不太适用于图像风格迁移这类应用。

所以这篇文章提出了**Instance
Normalization（IN）**，一种更适合对单个像素有更高要求的场景的归一化算法（IST，GAN等）。IN的算法非常简单，计算归一化统计量时考虑单个样本，单个通道的所有元素。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image284.jpeg){width="5.409027777777778in"
height="1.8458333333333334in"}

IN（右）和BN（中）以及LN（左）的不同从图中可以非常明显的看出。

**参考**

[[https://zhuanlan.zhihu.com/p/56542480]{.underline}](https://zhuanlan.zhihu.com/p/56542480)

**论文**

[[https://arxiv.org/pdf/1607.08022.pdf]{.underline}](https://arxiv.org/pdf/1607.08022.pdf)

### Group Normalization (201803)

**Group Normalization（GN）**是针对**Batch Normalization（BN）**在batch
size较小时错误率较高而提出的改进算法，因为BN层的计算结果依赖当前batch的数据，当batch
size较小时（比如2、4这样），该batch数据的均值和方差的代表性较差，因此对最后的结果影响也较大。

Group Normalization（GN）是何恺明提出的一种归一化策略，它是介于[Layer
Normalization](https://zhuanlan.zhihu.com/p/54530247)（LN）和 [Instance
Normalization](https://zhuanlan.zhihu.com/p/56542480)（IN）之间的一种折中方案，下图最右。它通过将通道数据分成几组计算归一化统计量，因此GN也是和批量大小无关的算法，因此可以用在batchsize比较小的环境中。作者在论文中指出GN要比LN和IN的效果要好。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image285.jpeg){width="5.524305555555555in"
height="1.4770833333333333in"}

图中从左至右依次是：BN、LN、IN、GN

**参考**

[[https://zhuanlan.zhihu.com/p/56613508]{.underline}](https://zhuanlan.zhihu.com/p/56613508)

[[https://blog.csdn.net/u014380165/article/details/79810040]{.underline}](https://blog.csdn.net/u014380165/article/details/79810040)

**论文**

[[https://arxiv.org/pdf/1803.08494.pdf]{.underline}](https://arxiv.org/pdf/1803.08494.pdf)

### Switchable Normalization (201806) 

虽然BN、LN、IN这些归一化方法往往能提升模型的性能，但是当你接收一个任务时，具体选择哪个归一化方法仍然需要人工选择，这往往需要大量的对照实验或者开发者优秀的经验才能选出最合适的归一化方法。本文提出了Switchable
Normalization（SN），它的算法核心在于提出了一个可微的归一化层，可以让模型根据数据来学习到每一层该选择的归一化方法，亦或是三个归一化方法的加权和，如下图所示。所以SN是一个任务无关的归一化方法，不管是LN适用的RNN还是IN适用的图像风格迁移（IST），SN均能用到该应用中。作者在实验中直接将SN用到了包括分类，检测，分割，IST，LSTM等各个方向的任务中，SN均取得了非常好的效果。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image286.jpeg){width="5.213888888888889in"
height="1.2805555555555554in"}

**参考**

[[https://zhuanlan.zhihu.com/p/57807576]{.underline}](https://zhuanlan.zhihu.com/p/57807576)

**论文**

[[https://arxiv.org/pdf/1806.10779.pdf]{.underline}](https://arxiv.org/pdf/1806.10779.pdf)

研究方向：图像
==============

本章介绍了NN在图像这个研究方向的概念、分支、使用的技术、技术的演进、使用的数据集等等。

在NN兴起之前，人工智能各个领域被很多非NN的技术占据，整体而言，在所有这些研究方向里，大致呈现几个趋势：

1.  其他技术渐渐被NN取代，有的早期实现是NN和其他技术共存，随着发展，**NN**渐渐替代了其他部分的非NN技术，在很多现有的前沿技术已经全由NN构成，输入输出全部实现**端到端**。

2.  在各个数据集的准确率在逐渐上升

3.  有很多技术被发展出来用于在相同效率的情况下减少计算量

目前在业界有2种相反的路线观点，一种观点认为，只要有足够的算力，所有的问题都可以用足够强大的端到端神经网络解决。另一个观点认为，无论算力发展到多强大，NN仍然不是**强人工智能**的最终解决方案，因此仍然需要部分人工的介入。

推荐一个网站：[[www.paperswithcode.com]{.underline}](http://www.paperswithcode.com)
。该网站中包含了很多的研究方向：有论文有代码有排行榜，并且根据网站介绍，该网站是自动化更新的。

目前的深度学习，最大应用方向（一多半的论文都在这个方向）毫无疑问是CV（计算机视觉），剩下的应用里，NLP（自然语言处理）又占了一大部分，其余包括医疗、打游戏、机器人、音乐等等。从总体比例来看，差不多"NN研究共一石，CV独得八斗，NLP得一斗，其他方向共分一斗"。

而CV其中又划分了很多的子类及更小的细分方向，其中几个比较大的，也是实用性比较强的子类为：

-   **图像理解**

-   **视频理解**

-   **人脸识别（及相关）**

-   **行人识别（及相关）**

**图像理解**里包括了几个基本且主要的方向：

-   图像分类：Image Classification，解决了What的问题

-   目标检测：Object Detection，解决了What和Where（边界框）

-   图像分割：Image Segmentation，解决了What和Where（像素级别），分为：

    -   语义分割：Semantic Segmentation，按同类物体分割

    -   实例分割：Instance Segmentation，按单个物体分割

    -   全景分割：Panoptic Segmentation，前景实例分割，背景语义分割

-   图像描述：Image Caption，根据图像生成描述

-   图像问答：Image
    QA，图像理解的终极王者，给定图像和问题，NN给出回答，比如"图中几个人的衣服分别是什么颜色"

图像分类（MNIST，ImageNet）为NN的发源地。在大多数研究中，CNN是作为一个基本结构出现的（一般称为Backbone，骨干网络，基础网络），比如在目标检测（Object
Detection）中的Faster R-CNN网络结构，其中用到了2个子CNN网络。

对于每一个具体的实现而言，包括以下几个因素：

-   **数据集**：数据集有很多种，每个数据集基本框定了可以进行的运算（比如著名的ImageNet和LFW）

-   **任务**：在同一个数据集中可能可以实现若干个研究任务（比如LFW可以进行人脸检测和人脸识别），对于一个任务，最好有一个对应的衡量标准（Metrics，比如ImageNet图像分类任务，衡量标准是前一和前五的准确率），衡量标准可能没有。

-   **网络结构**：网络结构是网络的基础，一个创新的网络结构通常会有一篇**论文**来描述，而一个网络结构也可能通过不同实现达到N种研究目的（比如FaceNet实现人脸验证和人脸识别）

-   **实现**：网络结构的具体实现，可能是代码、超参数设置、或者更具体的预训练好的权重

因此对于一个具体的网络，你可能可以找到：

-   一个预训练好的，针对某一数据集，某一目的的，可以直接使用的网络

-   一个开源的网络结构，没有权重，可能有训练方法或者超参数

-   一篇论文，其中描述了网络结构，没有任何开源实现

以下各个网络模型后的时间表示其论文发表的时间，该时间晚于其提出时间。

图像分类（Image Classification）
--------------------------------

在各个研究方向中，**图像分类（Image
Classification）**是比较特别的一支。整个神经网络的复兴，就是起源于2012年AlexNet在ImageNet图像分类赛中的君临天下，此后CNN和图像分类也成了神经网络/深度学习的最基础的分支。

对于基础CNN网络（也就是单一的CNN，没有额外的复合架构）的技术演进，图像分类的准确率也是验证其性能的最主要手段。而很多其他的研究方向，要么就是采用基础CNN网络的变体，要么就是采用包含了基础CNN的复合架构。

因此这一节中描述了关于图像分类的CNN，都是著名且基础的CNN。之所以**著名**，是因为这些网络模型的提出带有自己的创新发明，或者是独特的层结构，或者是创新的思想。之所以**基础**，是因为这些都是传统的卷积神经网络，虽然主要的工作目的是图像分类，但是可以被整合在完成其他功能的更复杂的神经网络中，作为其中的一部分（通常被称为其backbone，骨干网络，基础网络）。

这些基础CNN的大致结构都差不多：前面的卷积层+后面的分类器，分类器通常由全局池化层+FC层+Softmax三部分组成。

### 衡量标准

#### Top-N错误率

**top-N error
rate**，对图像分类的预测结果是按照概率排序的，top-1错误率表示仅看第一个时的错误率，top-5表示预测结果中排前五的包含正确标签的错误率。top-1和top-5是常用的两个标准。

#### TPR / FPR

可参考《[[概念定义 \> NN相关 \> TP/FP/TN/FN]{.underline}](\l)》

**TPR（True Positive
Ratio）**，也叫**查全率**，**召回率（Recall）**，**hit
rate**，或者**Sensitivity**，等于TP/(TP+FN)，用于衡量所有实际为阳性的样本的检测准确率。

**FPR（False Positive
Ratio）**，等于FP/(FP+TN)，等于1-TNR，用于衡量所有实际为阴性样本的检测错误率。

#### ROC曲线 / AUC值

**TPR**和**TNR**通常在不同阈值的情况负相关，TPR越高，TNR越低。因此就有了**ROC曲线**（**ROC
Curve**），以FPR为x轴，TPR为y轴，ROC曲线越靠近（0，1）点，说明分类器能力越强。

![](/home/jimzeus/outputs/AANN/images/media/image287.png){width="2.990972222222222in"
height="2.2118055555555554in"}

也可以用**AUC（Aera Under
Curve）**，即ROC曲线下的面积来衡量分类器的能力。

### 数据集

以下列出部分用于图像分类的数据集，或者多用途数据集的图像分类部分，此外**目标检测**和**图像分割**的数据集通常也可以用作图像分类任务，请自行参考。

#### MNIST

MNIST=Modified National Institute of Standard and Technology。

经典的手写数字低分辨率（28\*28）图片数据集，图像分类领域的Hello
World，包括超过6万张训练图像和1万张测试图像。

#### CIFAR-10

CIFAR=Canadian Institute For Advanced Research。

分为CIFAR-10和CIFAR-100。由Alex Krizhevsky（AlexNet的作者）2009年推出。

CIFAR-10包括了6万张分辨率为32\*32的图片，被分为10类，每类6000张图片。这10类为：飞机、小轿车、鸟、猫、鹿、狗、青蛙、马、船和卡车。

CIFAR-100类似，但被分为了100类。

#### ImageNet

这应该是整个神经网络/计算机视觉界最有名的数据集。

由李飞飞创建的庞大的可视化数据集，超过1400万图片被人工标注出包含什么物体，至少100万图片包含了标注框（box），超过2万个类别。可以被用于衡量/训练图像分类任务，以及目标检测任务。

官网：

[[http://image-net.org]{.underline}](http://image-net.org)

下载：

[[http://image-net.org/download.php]{.underline}](http://image-net.org/download.php)

ImageNet中的图片是按照WordNet3.0进行组织的，对每一个图片类（synset），用wnid（WordNet
ID）标示，比如"军装"的wnid是n03763968。

-   synset的页面：

http://www.image-net.org/synset?wnid=\[wnid\]

-   synset的直接子类：

<http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=>\[wnid\]

-   synset的所有子类（直接和间接）：

<http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=>\[wnid\]&full=1

-   synset的名称：

http://www.image-net.org/api/text/wordnet.synset.getwords?wnid=\[wnid\]

-   synset包含的所有图片的URL：

<http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=>\[wnid\]

-   synset包含的所有图片的图片名和URL映射

<http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid=>\[wnid\]

-   所有synset的wnid和名称的映射：

[[http://image-net.org/archive/words.txt]{.underline}](http://image-net.org/archive/words.txt)

-   所有synset的wnid列表

[[http://www.image-net.org/api/text/imagenet.synset.obtain\_synset\_list]{.underline}](http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list)

-   所有synset的wnid和解释的映射

[[http://image-net.org/archive/gloss.txt]{.underline}](http://image-net.org/archive/gloss.txt)

-   WordNet中的所有的synset父类和子类的关系映射（wnid-wnid）

[http://www.image-net.org/archive/wordnet.is\_a.txt]{.underline}

#### COCO

**C**ommon **O**bject
**Co**ntext，微软发布的物体识别数据集，包括超过33万张图片（其中20万张被标注）和80个物体类别，包括目标检测，图像分割，关键点检测等任务。

#### Pascal VOC

Visual Object
Classes，目标检测和图像分类分类数据集，包括超过11万张图片和20个物体类别，2010年推出。

#### Object365

旷视2019年发布的通用目标检测数据集，包括63万张图片，其中的物体分为365个类别，高达1000万的框数。

### 代码

图像分类作为最基础的神经网络，几乎在所有NN的通用框架中都有实现：

-   darknet: darknet/cfg

-   keras: keras/applications

-   tensorflow: tensorflow/python/keras/applications

-   pytorch: torchvision/models

-   mxnet: mxnet/gluon/model\_zoo/vision

-   gluon: gluoncv/model\_zoo

### LeNet（1998）

1998年Yann
LeCun设计的CNN，用于识别手写文字的图片，CNN的开山鼻祖，也确定了基础CNN的大致结构。现在常用的简化改进过的LeNet-5包含5个层（2个卷积+池化层，2个FC层，1个Softmax层）。

> ![图片包含 屏幕截图
> 描述已自动生成](/home/jimzeus/outputs/AANN/images/media/image288.png){width="2.0069444444444446in"
> height="3.6729166666666666in"}

### AlexNet（2012）

2012年**Hinton**的学生**Alex
Krizhevsky**设计的CNN，当年ImageNet（ILSVRC2012）分类竞赛的冠军，大幅提高了对ImageNet的识别率，从而使得深度神经网络成为一门显学。

AlexNet网络作者是多伦多大学的Alex Krizhevsky等人。Alex
Krizhevsky是Hinton的学生。在ILSVRC-2010竞赛中，AlexNet取得了37.5%（top-1）和17%（top-5）的错误率。并且在ILSVRC-2012竞赛中获得了15.3%（top-5）的错误率，
获得第二名的方法错误率 是
26.2%，可以说差距是非常的大了，足以说明这个网络在当时给学术界和工业界带来的冲击之大。

AlexNet网络结构在整体上类似于LeNet，都是先卷积然后在全连接（之后很多的CNN都采取这种形式）：

-   AlexNet有60 million个参数和65000个 神经元

-   包括五层卷积（其中部分后跟max池化层），三层全连接网络

-   最终的输出层是1000通道的softmax

-   为了防止过拟合，采用了**dropout**技术（当时是新技术）

-   AlexNet利用了两块GPU进行计算，提高了运算效率

> ![图片包含 文字, 地图
> 描述已自动生成](/home/jimzeus/outputs/AANN/images/media/image289.png){width="5.009027777777778in"
> height="1.5694444444444444in"}

上图是AlexNet的网络结构，前半部分分为两个相同的结构，这是因为AlexNet是在两个GPU上进行运算的，这俩部分分别对应两个GPU。

-   输入层为224\*224\*3（图片大小+RGB）

-   第一层卷积核为11\*11\*3，48\*2个，stride=4，padding=0，输出为55\*55\*48\*2，后接Max池化层，池化核3\*3，stride=2，输出为27\*27\*48\*2。

-   第二层卷积核为5\*5\*48，128\*2个，stride=1，padding=2，输出为27\*27\*128\*2，后接Max池化层，池化核3\*3，stride=2，输出为13\*13\*128\*2。

-   第三层卷积核为3\*3\*128，192\*2个，stride=1，padding=1，输出为13\*13\*192\*2。

-   第四层卷积核为3\*3\*192，192\*2个，stride=1，padding=1，输出为13\*13\*192\*2.

-   第五层卷积核为3\*3\*192，128\*2个，stride=1，padding=1，输出为13\*13\*128\*2，后接Max池化层，池化核3\*3，stride=2，输出为7\*7\*128\*2.

-   第一层4096的全连接层，激活函数为ReLU，dropout ratio为0.5

-   第二层4096的全连接层，激活函数为ReLU，dropout ratio为0.5

-   最后一层输出为1000的全连接层，激活函数为Softmax

**论文**

ImageNet Classification with Deep Convolutional Neural Networks

[[https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf]{.underline}](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

**参考**

[[https://blog.csdn.net/luoluonuoyasuolong/article/details/81750190]{.underline}](https://blog.csdn.net/luoluonuoyasuolong/article/details/81750190)

### ZFNet（201311）

2013 ImageNet（ILSVRC2013）图像分类赛的冠军，作者是Zeiler &
Fergus，在CNN的结构上相较于AlexNet只是调整了参数（卷积核大小、stride等等），并没有什么特别突出的地方。其论文的亮点是可视化技术的应用（ZFNet的论文pdf文件大小是其他论文大小的10倍，就是因为其中图片太多）

下图为ZFNet的结构：

> ![图片包含 地图, 文字
> 描述已自动生成](/home/jimzeus/outputs/AANN/images/media/image290.png){width="5.051388888888889in"
> height="1.2951388888888888in"}

在ZFNet中，为了实现对CNN的可视化，每一层所谓**convnet结构**（包括卷积、ReLU激活、max池化三层）都附加有一层**deconvnet结构**，deconvnet的三个步骤分别对应convnet的三个步骤：

-   **反池化（unpooling）**：池化操作是不可逆的，通过池化时记录下max池化时最大值的位置（被称为Switchs），在反池化时将最大值放回该位置，并将其余部分填0，反池化操作可以近似实现池化逆操作。

-   **反ReLU激活**：同ReLU激活函数

-   **反卷积**：卷积网使用学习得到的卷积核与输入做卷积得到特征图，为了实现逆过程，反卷积网使用该卷积核的转置作为卷积核，与反ReLU之后的特征图进行卷积计算

参考下图：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\633a5c889358aa54e618ea06961a4b8.png](/home/jimzeus/outputs/AANN/images/media/image291.png){width="4.276388888888889in"
height="5.284722222222222in"}

**论文**

Visualizing and Understanding Convolutional Networks

[[https://arxiv.org/pdf/1311.2901]{.underline}](https://arxiv.org/pdf/1311.2901)

**参考**

[[https://blog.csdn.net/qq\_36673141/article/details/78551932]{.underline}](https://blog.csdn.net/qq_36673141/article/details/78551932)

### VGGNet（201409）

**VGGNet**是由牛津大学的视觉**Visual Geometry Group**和**Google
Deepmind**一起研发的CNN，是2014年ImageNet（ILSVRC2014）分类赛的第二名（冠军是**GoogLeNet**）。

VGGNet有权重的层（卷积层或者全连接层）叠加到了13层、16层或者19层，在当时是很深的网络，也称VGG13、VGG16或者VGG19。

VGG的贡献主要有如下:

-   反复使用3\*3的卷积来替代之前的大卷积核，认为2个3\*3的卷积可以起到5\*5的卷积的效果，而计算量会大幅减少。这种思想对后来的网络有很大的影响。具体参考《网络构成 \>
    DNN/CNN微结构 \> 3\*3卷积核》

-   发现**LRN层**的作用不大

> ![](/home/jimzeus/outputs/AANN/images/media/image292.png){width="3.0819444444444444in"
> height="1.8479166666666667in"}

**论文**

Very Deep Convolutional Networks for Large-scale Image Recognition

[[https://arxiv.org/pdf/1409.1556]{.underline}](https://arxiv.org/pdf/1409.1556)

**参考**

[[https://blog.csdn.net/whz1861/article/details/78111606]{.underline}](https://blog.csdn.net/whz1861/article/details/78111606)

### GoogLeNet（201409）

Google开发的CNN，是2014年ImageNet（ILSVRC2014）分类赛的冠军，其特点是**Inception结构**，使得网络不仅在纵向上有深度，在横向上也有广度。

Inception结构的目的是感受不同大小视野上的特征，具体请参考《[[网络构成 \>
DNN/CNN微结构 \>
Inception结构]{.underline}](\l)》，以下是GoogLeNet的结构：

![141544\_FfKB\_876354.jpg](/home/jimzeus/outputs/AANN/images/media/image293.jpeg){width="2.1319444444444446in"
height="9.502083333333333in"}

**论文**

Going Deeper with Convolutions

[[https://arxiv.org/pdf/1409.4842.pdf]{.underline}](https://arxiv.org/pdf/1409.4842.pdf)

**参考**

[[https://my.oschina.net/u/876354/blog/1637819]{.underline}](https://my.oschina.net/u/876354/blog/1637819)

[[https://www.cnblogs.com/Allen-rg/p/5833919.html]{.underline}](https://www.cnblogs.com/Allen-rg/p/5833919.html)

### ResNet（201509）

**ResNet**是微软开发的CNN，2015年ImageNet（ILSVRC2015）分类赛的第一名，其特点是通过**残差结构（Residual
Block）**解决了反向传播梯度消失的问题，使得比以前的网络有更深的深度。请参考《[[网络构成
\> DNN/CNN微结构 \> Residual block]{.underline}](\l)》。

ResNet论文中的几个ResNet变种如下：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\6616e2e285dfea4359212f2b7404102.png](/home/jimzeus/outputs/AANN/images/media/image294.png){width="5.768055555555556in"
height="2.126388888888889in"}

这些变种通常被称为ResNet-50、ResNet-101、ResNet-152等。

**论文**

Deep Residual Learning for Image Recognition

[[https://arxiv.org/pdf/1512.03385]{.underline}](https://arxiv.org/pdf/1512.03385)

**参考**

[[https://blog.csdn.net/u013181595/article/details/80990930]{.underline}](https://blog.csdn.net/u013181595/article/details/80990930)

[[http://baijiahao.baidu.com/s?id=1598536455758606033&wfr=spider&for=pc]{.underline}](http://baijiahao.baidu.com/s?id=1598536455758606033&wfr=spider&for=pc)

### SqueezeNet（201602）

SqueezeNet在保持了AlexNet对ImageNet数据集的准确率的情况下，将参数数量压缩到了其1/50，并且将模型大小限制在了0.5M之内（使用了Deep
compression技术）。

**Fire
module**是SqueezeNet所提出的的微结构，由squeeze层（1\*1卷积，降维）和expand层（3\*3卷积和1\*1卷积并联，升维）组成，fire
module不改变特征图大小，只改变通道数。

SqueezeNet的网络结构由一个卷积层开始，中间若干**fire
module**（升维）和**池化层**（缩小尺寸），最后一个卷积层加一个softmax，结构如下：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\05002b9a119d06a489455dacc5d036f.png](/home/jimzeus/outputs/AANN/images/media/image295.png){width="4.425694444444445in"
height="3.102777777777778in"}

参考《[[网络构成 \> DNN/CNN微结构 \> Fire Module]{.underline}](\l)》

**论文**

SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters and \<0.5Mb
Model Size

[[https://arxiv.org/pdf/1602.07360]{.underline}](https://arxiv.org/pdf/1602.07360)

参考:

[[https://blog.csdn.net/csdnldp/article/details/78648543]{.underline}](https://blog.csdn.net/csdnldp/article/details/78648543)

[[https://blog.csdn.net/u011995719/article/details/78908755]{.underline}](https://blog.csdn.net/u011995719/article/details/78908755)

代码：

[[https://github.com/DeepScale/SqueezeNet]{.underline}](https://github.com/DeepScale/SqueezeNet)

### DenseNet（201608）

2016年发表的DenseNet吸收了ResNet的特点，是一种具有密集连接的CNN，在一个dense
block中的任何两层都有直接的连接，各层的输出在后面的层的输入中以通道的维度进行合并：

![图片包含 屏幕截图
描述已自动生成](/home/jimzeus/outputs/AANN/images/media/image296.png){width="5.768055555555556in"
height="0.7875in"}

每个Dense Block中的任意两层都有连接，而每两个Dense
block之间由于尺寸不同，有一个Transition层（包括一个1\*1卷积和一个2\*2的average池化）来实现下采样到下一个dense
block的尺寸，具体的结构如下：

![è¿éåå¾çæè¿°](/home/jimzeus/outputs/AANN/images/media/image297.png){width="5.768055555555556in"
height="2.691666666666667in"}

**论文**

Densely Connected Convolutional Networks

[[https://arxiv.org/pdf/1608.06993]{.underline}](https://arxiv.org/pdf/1608.06993)

**参考**

[[https://blog.csdn.net/blank\_tj/article/details/82563810]{.underline}](https://blog.csdn.net/blank_tj/article/details/82563810)

[[https://blog.csdn.net/sigai\_csdn/article/details/82115254]{.underline}](https://blog.csdn.net/sigai_csdn/article/details/82115254)

### Xception（201610）

**Xception**是在**Inception
V3**的基础上做的改进，其主要特点是加入了**Depthwise
Separable卷积**的思想。

参考《[[网络构成 \> DNN/CNN微结构 \> Depthwise
Separable卷积]{.underline}](\l)》

**论文**

Xception: Deep Learning with Depthwise Separable Convolutions

[[https://arxiv.org/pdf/1610.02357]{.underline}](https://arxiv.org/pdf/1610.02357)

参考:

[[https://zhuanlan.zhihu.com/p/50897945]{.underline}](https://zhuanlan.zhihu.com/p/50897945)

[[https://blog.csdn.net/u014380165/article/details/75142710]{.underline}](https://blog.csdn.net/u014380165/article/details/75142710)

[[https://www.leiphone.com/news/201708/KGJYBHXPwsRYMhWw.html]{.underline}](https://www.leiphone.com/news/201708/KGJYBHXPwsRYMhWw.html)

### ResNeXt (201611)

**ResNeXt**是[**ResNet**](https://zhuanlan.zhihu.com/p/42706477)和[**Inception**](https://zhuanlan.zhihu.com/p/42704781)的结合体，不同于[**Inception
v4**](https://zhuanlan.zhihu.com/p/42706477)（**Inception-ResNet**结构）的是，ResNeXt不需要人工设计复杂的Inception结构细节，而是每一个分支都采用相同的拓扑结构。ResNeXt的本质是[**分组卷积（Group
Convolution）**](https://zhuanlan.zhihu.com/p/50045821)，通过变量基数（Cardinality）来控制组的数量。组卷积是普通卷积和**深度可分离卷积（Depthwise
Seperable Conv）**的一个折中方案，即每个分支产生的Feature
Map的通道数为 n (n\>1)。下图左为ResNet结构，右为ResNeXt结构：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image170.png){width="4.376388888888889in"
height="2.652083333333333in"}

**参考**

[[https://zhuanlan.zhihu.com/p/51075096]{.underline}](https://zhuanlan.zhihu.com/p/51075096)

**论文**

[[https://arxiv.org/pdf/1611.05431.pdf]{.underline}](https://arxiv.org/pdf/1611.05431.pdf)

### MobileNet V1（201704）

MobileNet
V1是谷歌推出的致力于移动设备上使用的轻量化网络，其主要特点和Xception一样，也是**深度可分离卷积（Depthwise
Separable Convolution）**。

MobileNet的结构如下，可以看出来是连续的深度可分离卷积的堆叠：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\ab8dce5c730433cfac4d12fe0393d50.png](/home/jimzeus/outputs/AANN/images/media/image298.png){width="3.88125in"
height="4.4847222222222225in"}

MobileNets作为轻量级网络，其主要特点是在不显著降低准确率的情况下，大幅降低参数和计算量，这在不同的应用场景中都有体现。

图像分类：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\34d1ca7fe21cda0bb7a94c89325dc05.png](/home/jimzeus/outputs/AANN/images/media/image299.png){width="3.451388888888889in"
height="1.1215277777777777in"}

目标检测：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\4622d23e6533717cc47fd495f3bab4d.png](/home/jimzeus/outputs/AANN/images/media/image300.png){width="2.747916666666667in"
height="1.8333333333333333in"}

人脸识别：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\b1a0656e90bc86bc0a53ef9993e6985.png](/home/jimzeus/outputs/AANN/images/media/image301.png){width="3.442361111111111in"
height="1.39375in"}

**论文**

MobileNets: Efficient Convolutional Neural Networks for Mobile Vision

Applications

[[https://arxiv.org/pdf/1704.04861]{.underline}](https://arxiv.org/pdf/1704.04861)

**参考**

[[https://blog.csdn.net/u011974639/article/details/79199306]{.underline}](https://blog.csdn.net/u011974639/article/details/79199306)

[[https://blog.csdn.net/mzpmzk/article/details/82976871]{.underline}](https://blog.csdn.net/mzpmzk/article/details/82976871)

### NasNet (201707)

NasNet是Google通过自动架构搜索生成的网络。

具体请参考《[[元学习 \> AutoML \> 算法：NAS \>
NasNet]{.underline}](\l)》

**论文**

[[https://arxiv.org/pdf/1707.07012.pdf]{.underline}](https://arxiv.org/pdf/1707.07012.pdf)

### ShuffleNet V1 (201707)

在[ResNeXt](https://zhuanlan.zhihu.com/p/51075096)中，分组卷积作为传统卷积核深度可分离卷积的一种折中方案被采用。这时大量的对于整个Feature
Map的Pointwise卷积成为了ResNeXt的性能瓶颈。一种更高效的策略是在组内进行Pointwise卷积，但是这种组内Pointwise卷积的形式不利于通道之间的信息流通，为了解决这个问题，ShuffleNet
v1中提出了**通道洗牌（channel shuffle）**操作。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image302.jpeg){width="4.6125in"
height="1.7041666666666666in"}

上图中左为普通分组卷积，右为channel
shuffle，使得第二层卷积的输入为第一层卷积的各组输入。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image303.jpeg){width="4.3625in"
height="2.4347222222222222in"}

上图(a)为一个普通的带有残差结构的深度可分离卷积（[MobileNet](https://zhuanlan.zhihu.com/p/50045821), [Xception](https://zhuanlan.zhihu.com/p/50897945)）。ShuffleNet
v1的结构图(b)，(c)。其中(b)不需要降采样，(c)是需要降采样的情况。

**参考**

[[ShuffleNet旷世官方解读]{.underline}](https://mp.weixin.qq.com/s?__biz=MzA3NjIzMTk0NA==&mid=2651646559&idx=1&sn=9a4ff0a61f022de9dff6369bf1913896&scene=21#wechat_redirect)

[[https://zhuanlan.zhihu.com/p/51566209]{.underline}](https://zhuanlan.zhihu.com/p/51566209)

**论文**

[[https://arxiv.org/pdf/1707.01083.pdf]{.underline}](https://arxiv.org/pdf/1707.01083.pdf)

**代码**

[[https://github.com/megvii-model/ShuffleNet-Series]{.underline}](https://github.com/megvii-model/ShuffleNet-Series)

### SENet（201709）

SENet是最后一届ImageNet（2017）分类赛的冠军。

SENet可以被视为是一种注意力模型。通常的卷积的计算方式，对每个channel来说，都是以同样卷积核心乘以各个输入channel的feature
map再相加。SENet网络的创新点在于关注channel之间的关系，希望模型可以自动学习到不同channel特征的重要程度。为此，SENet提出了Squeeze-and-Excitation
(SE)模块，如下图所示：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image304.jpeg){width="5.6715277777777775in"
height="2.3229166666666665in"}

SE模块首先对卷积得到的特征图进行Squeeze操作，得到channel级的全局特征，然后对全局特征进行Excitation操作，学习各个channel间的关系，也得到不同channel的权重，最后乘以原来的特征图得到最终特征。本质上，SE模块是在channel维度上做attention或者gating操作，这种注意力机制让模型可以更加关注信息量最大的channel特征，而抑制那些不重要的channel特征。另外一点是SE模块是通用的，这意味着其可以嵌入到现有的网络架构中。

具体SE模块请参考《[[网络构成 \> DNN/CNN微结构 \>
SE结构]{.underline}](\l)》

**论文**

Squeeze-and-exitation networks

[[https://arxiv.org/pdf/1709.01507.pdf]{.underline}](https://arxiv.org/pdf/1709.01507.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/65459972]{.underline}](https://zhuanlan.zhihu.com/p/65459972)

**代码**

[[https://github.com/hujie-frank/SENet]{.underline}](https://github.com/hujie-frank/SENet)

### MobileNet V2（201801）

相对于MobileNet V1，MobileNet V2主要引入了两个改动：

-   **Linear
    bottleneck**：在DW卷积前增加一个PW卷积，去掉DW后的PW卷积的ReLU

-   **Inverted Residual block**：在层数比较深的ResNet中会使用Residual
    bottleneck结构，Inverted residual
    block结构与之类似，但将常规卷积改为了DW卷积。

整个网络的结构如下：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\3eda9dc26a6ae779f79a9aa1175d078.png](/home/jimzeus/outputs/AANN/images/media/image305.png){width="4.129861111111111in"
height="3.16875in"}

**论文**

MobileNetV2: Inverted Residuals and Linear Bottlenecks

[[https://arxiv.org/pdf/1801.04381]{.underline}](https://arxiv.org/pdf/1801.04381)

**参考**

[[https://zhuanlan.zhihu.com/p/33075914]{.underline}](https://zhuanlan.zhihu.com/p/33075914)

### MNasNet (201807)(TODO)

**论文**

[[https://arxiv.org/pdf/1807.11626.pdf]{.underline}](https://arxiv.org/pdf/1807.11626.pdf)

### ShuffleNet V2 (201807)

在ShuffleNet
v2的文章中作者指出现在普遍采用的FLOPs评估模型性能是非常不合理的，因为一批样本的训练时间除了看FLOPs，还有很多过程需要消耗时间，例如文件IO，内存读取，GPU执行效率等等。作者从内存消耗成本，GPU并行性两个方向分析了模型可能带来的非FLOPs的行动损耗，进而设计了更加高效的**ShuffleNet
v2**。ShuffleNet
v2的架构和[**DenseNet**](https://zhuanlan.zhihu.com/p/42708327)有异曲同工之妙，而且其速度和精度都要优于DenseNet。

文中总结了设计高性能网络的四点规则：

1.  使用输入通道和输出通道相同的卷积操作；

2.  谨慎使用分组卷积；

3.  减少网络分支数；

4.  减少element-wise操作。

根据以上几条原则，设计出了ShuffleNet V2的单元结构，下图中，a是ShuffleNet
v1的普通单元，b是v1的降采样单元，c是v2的普通单元，d是v2的降采样单元。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image306.jpeg){width="5.214583333333334in"
height="3.053472222222222in"}

**参考**

[[ShuffleNet
V2旷世官方解读]{.underline}](https://mp.weixin.qq.com/s?__biz=MzA3NjIzMTk0NA==&mid=2651647996&idx=1&sn=71abac18f177126daa89593398f2267d&scene=21#wechat_redirect)

[[https://zhuanlan.zhihu.com/p/51566209]{.underline}](https://zhuanlan.zhihu.com/p/51566209)

[[https://zhuanlan.zhihu.com/p/48261931]{.underline}](https://zhuanlan.zhihu.com/p/48261931)

**论文**

[[https://arxiv.org/pdf/1807.11164.pdf]{.underline}](https://arxiv.org/pdf/1807.11164.pdf)

代码：

[[https://github.com/megvii-model/ShuffleNet-Series]{.underline}](https://github.com/megvii-model/ShuffleNet-Series)

### EfficientNet (201905)(TODO)

EfficientNet是当前最强的图像分类CNN。

**论文**

[[https://arxiv.org/pdf/1905.11946.pdf]{.underline}](https://arxiv.org/pdf/1905.11946.pdf)

**参考**

EfficientNet详解。凭什么EfficientNet号称当今最强？

[[https://zhuanlan.zhihu.com/p/104790514]{.underline}](https://zhuanlan.zhihu.com/p/104790514)

EfficientNet-可能是迄今为止最好的CNN网络

[[https://zhuanlan.zhihu.com/p/67834114]{.underline}](https://zhuanlan.zhihu.com/p/67834114)

代码：

[[https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet]{.underline}](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

### RegNet (TODO)

**论文**

**参考**

何恺明团队最新力作RegNet：超越EfficientNet，GPU上提速5倍，这是网络设计新范式

[[https://zhuanlan.zhihu.com/p/122278712]{.underline}](https://zhuanlan.zhihu.com/p/122278712)

代码：

目标检测（Object Detection）
----------------------------

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image307.jpeg){width="5.434722222222222in"
height="2.7506944444444446in"}

如果说图像分类是NN最先应用的任务，目标检测（物体检测）就是第二个。如果Image
Classification是Black Sabbath，Object Detection就是Judas
Priest（不了解金属乐的请忽略该句）。

相较于图像分类，**目标检测（Object
Detection）**任务的难度在于，它需要给出图片中若干个物体的类别及其位置（X,Y,H,W）。或者说，目标检测的任务有2个，一个是分类，一个是定位。

在2012年之前，物体检测一直是**计算机视觉（CV）**加上**机器学习（ML）**的天下。当时主要的研究方向是如何设计高质量的特征提取器和高效的分类器和回归器。当时特征提取器比较有代表性的算法有**HoG**，**SIFT**等，而分类任务几乎是**SVM**的天下。

早期的物体检测基本上也遵循四步走的流程：

-   Selective Search选取候选区域；

-   特征提取器提取特征；

-   分类器和回归器预测类别和位置四要素；

-   non maximum suppression合并检测框。

2012年，Hinton团队的[AlexNet](https://zhuanlan.zhihu.com/p/42692344)在ILSVRC大赛中将精度提高了约10个百分点。从此，计算机视觉的各个方向都开始考虑使用深度学习解决他们的问题。

卷积网络作为特征提取器首先引起了业内研究者的注意，2014年，使用深度学习解决物体检测问题的开山之作[**R-CNN**](https://zhuanlan.zhihu.com/p/42731634)应运而生，论文的一作Ross
B.
Girshick（RBG）不仅是该方向的鼻祖，而且其一系列的论文也引领了物体检测的发展方向。R-CNN的目的也很单纯，只是将特征提取器简单的换成了CNN。

-   骨干网络

同时在2014年，网络模型方向诞生了两个非常经典的网络结构，一是牛津大学计算机视觉组的[**VGG**](https://zhuanlan.zhihu.com/p/42695549)，另外一个是谷歌公司的[**GoogLeNet**](https://zhuanlan.zhihu.com/p/42704781)。VGG使用了当时最流行的深度学习框架Caffe，并非常有先见之明的开源了其训练好的网络模型。VGG也作为骨干网络成为了之后3-4年的检测算法的主要使用骨干网络，代表算法便是RBG的**R-CNN系列**的三篇文章。随着数据量的增大和人们对高精度的追求，骨干网络更深的深度成为了一个最容易想到的方向。很幸运，2016年深度学习领域的另外一尊大神，我国广东省2003年的高考理科状元何恺明的**[残差网络](https://zhuanlan.zhihu.com/p/42706477)ResNet**使用short-cut解决了深度学习中的退化问题，因为其无限深度的能力成为了近几年物体检测算法骨干网络主要使用的算法，经典算法包括**[R-FCN](https://zhuanlan.zhihu.com/p/42858039)，[Mask
R-CNN](https://zhuanlan.zhihu.com/p/42745788)**以及[**YOLOv3**](https://zhuanlan.zhihu.com/p/42865896)等。

-   端到端模型

物体检测的一个非常重要的优化方向是优化传统方法四步走的流程，2015年RBG的[Fast
R-CNN](https://zhuanlan.zhihu.com/p/42738847)使用softmax替代了SVM，进而将特征提取和分类模型的训练合二为一，算是第一个端到端的物体检测算法。在[R-FCN](https://zhuanlan.zhihu.com/p/42858039)中使用了更为快速的投票机制替代了Fast
R-CNN中的softmax，因为softmax前往往要接最少一层全连接，这也成了制约Fast
R-CNN速度的一个重要瓶颈。[YOLOv3](https://zhuanlan.zhihu.com/p/42865896)则是使用C路sigmoid的多标签模型增强了对覆盖样本的检测能力。

同是在2015年，RBG和何恺明强强联手，推出了使用**RPN**替代了**Selective
Search**的[Faster
R-CNN](https://zhuanlan.zhihu.com/p/42741973)算法。Faster
R-CNN因为其最高的算法精度和在显卡环境下的近实时的速度性能，也成了今年最为流行的算法之一。Faster
R-CNN因其巧妙的设计也是深度学习面试官最爱问的算法之一。

**Faster
R-CNN**系列虽然在实现上实现了端到端训练，但是其两步走（候选区域提取+位置精校）的策略也被一些人诟病。2016年，Joseph
Redmon提出了更为革命性的[**YOLO**](https://zhuanlan.zhihu.com/p/42772125)系列算法。不同于R-CNN系列分两步走的策略，YOLO是单次检测检测的算法，YOLO可以看做是高精度的RPN。其更彻底的端到端训练将物体检测的速度大幅提升，在非顶端显卡环境下也实现了实时检测。

-   降采样池化

无论是**Selective
Search**还是**RPN**，得到的候选区域在尺寸和比例上都是不固定的，由此输入到网络中得到的Feature
Map大小是不同的，最后展开成的特征向量长度也不固定，在目前的开源框架下，暂不支持变长的特征向量作为输入。在**SPP-net**中，作者提出了金字塔池化的方式，通过多尺度分bin的形式得到长度固定的特征向量，在**Fast
R-CNN**中将其简化为单尺度并命名为**ROI Pooling**。[**Mask
R-CNN**](https://zhuanlan.zhihu.com/p/42745788)发现当**ROI
Pooling**应用到语义分割任务中会存在若干个像素的偏移误差，由此设计了更为精确的**ROIAlign**。

-   锚点

**Faster
R-CNN**最大的特点是在**RPN**网络中引入了**锚点**机制，对锚点一个更好的解释是**先验框**，即对检测框的先验假设。在早期阶段，锚点是根据开发者的经验手动写死的。在**YOLOv2**中，作者在训练集对锚点进行了**k-means聚类**，进而产生了一组更优代表性的锚点。**DSSD**中锚点的设置则是根据聚类的结果分析得到的。

-   小尺寸物体检测困难

在所有的检测算法中都普遍存在着小尺寸物体检测困哪的问题。究其原因，是因为在深层网络中随着语义信息的增强，位置信息也越来越弱，这是深度网络的固有问题。[**SSD**](https://zhuanlan.zhihu.com/p/42795805)率先提出使用各个阶段的Feature
Map都参与损失函数的计算，在**FPN**中则是通过将各个阶段的Feature
Map融合到一起的方式，融合的方式有[**FPN**](https://zhuanlan.zhihu.com/p/42745788)中从小尺寸向大尺寸融合的双线性插值上采样算法，也是目前最为广泛使用的融合方法；[DSSD](https://zhuanlan.zhihu.com/p/42795805)\[11\]则是通过反卷积得到不仅将小尺寸Feature
Map上采样，而且包含语义信息的Feature
Map；而[YOLOv2](https://zhuanlan.zhihu.com/p/42861239)采用的是中的大尺寸向小尺寸融合的space\_to\_depth()算法。而[YOLOv3](https://zhuanlan.zhihu.com/p/42865896)则是接合了FPN和锚点机制的思想，为不同深度的Feature
Map赋予了不同比例，不同尺寸的锚点。

[YOLOv2](https://zhuanlan.zhihu.com/p/42861239)中采用的另外一个解决方案则是在训练过程中，不同批使用不同尺寸的输入图像。

-   半监督学习

系列算法中一个非常有商业前景的方向便是通过半监督学习的方式增加模型可处理的类别。半监督学习即是通过少量的带标签数据和大量的无标签数据，将模型的能力扩展到无标签数据中。[**YOLO9000**](https://zhuanlan.zhihu.com/p/42861239)通过WordTree融合了80类的检测数据集**COCO**和9418类的分类数据集**ImageNet**，生成了可以检测9418类物体的模型。[**MaskX
R-CNN**](https://zhuanlan.zhihu.com/p/42749621)则是通过权值迁移函数融合了80类的分割数据COCO和3000类的检测数据集Visual
Gnome，生成了可以分割3000类物体的模型。

-   物体检测和语义分割

近几年物体检测和语义分割的距离越来越小，双方都在汲取对方的算法来获得灵感和优化算法。最典型的算法便是[Mask
R-CNN](https://zhuanlan.zhihu.com/p/42745788)中融合了分类，检测和分割的三任务模型。[**DSSD**](https://zhuanlan.zhihu.com/p/42795805)使用反卷积进行上采样也非常有意思。

最后预测一下未来一段时间物体检测的发展方向：

1.  小尺寸物体检测困难至今尚未有效解决，更有效的多尺度Feature
    Map，或者针对小尺寸物体的特定算法是研究的一个热点和难点；

2.  半监督学习：能否将语义分割任务扩展到ImageNet类别中，提升非子类或父类物体的无监督学习能力是一大热点；

3.  嵌入式平台的物体检测算法：目前最快的YOLOv3的实时运行依然依赖GPU环境，能否将检测算法实时的应用到嵌入式平台，例如手机，扫地机器人，无人机等都是有急切需求的场景；

4.  特定领域的物体检测算法：目前在单一领域发展较靠前的是场景文字检测算法。但在一些特定的场景中，例如医学，安检，微生物等依然很有研究前景，也是比较容易有研究成果和应用场景的方向。

总体来讲，当前的目标检测网络大致可以分为两类，一步检测和两步检测。

**两步检测（Multi-shot
detector）**先利用候选区域网络找出可能的候选目标（通常数量固定），这步被称为region
proposal，再用第二个网络来预测每个候选框的置信度（confidence）并修改边界框。早期的R-CNN系列就是两步检测架构。

**一步检测（Single-shot
detector）**最著名的即是YOLO和SSD，也可以分为两类：基于锚点（即anchor
box）的检测和基于关键点的检测。

**参考**

[[https://zhuanlan.zhihu.com/p/43211392]{.underline}](https://zhuanlan.zhihu.com/p/43211392)

two/one-stage,anchor-based/free目标检测发展及总结：一文了解目标检测

[[https://zhuanlan.zhihu.com/p/100823629]{.underline}](https://zhuanlan.zhihu.com/p/100823629)

### 衡量标准

#### IoU

**交并比，Intersection over
Union**，用于衡量两个区域的重合度，在目标检测任务中，这两个区域指的是**真实边界框**和**预测边界框**。其定义相当于TP/(TP+FP+FN)，图示如下：

![](/home/jimzeus/outputs/AANN/images/media/image308.png){width="2.3722222222222222in"
height="1.8027777777777778in"}

基于以上的定义，IoU应而可以衡量目标检测任务中**单个预测**的正确与否，比如在PASCAL
VOC数据集的测评标准中，IoU大于0.5的预测被视为正确。而如果对同一个目标有多次预测（即产生了若干个预测边界框），则第一个预测被视为真阳性，其它的预测被视为假阳性。

#### AP

**精度均值，Average
Precision**，P-R曲线中的Precision在不同Recall（实际是不同置信度导致的）上的均值。

有了IoU推出的TP/FP/TN/FN作为前提，就可以引出目标检测中的Precision和Recall的定义，复习一下前文介绍的这两个概念：

-   Precision：TP/(TP+FP)，所有被检测为阳性的样本的检测正确率

-   Recall：TP/(TP+FN)，所有实际为阳性的样本的检测正确率

在某个IoU阈值（这里指**某个**IoU，注意下图中的曲线是在单个IoU之下画出来的）之下，根据**不同的置信度(confidence或score)**可以统计出一组不同的Precision-Recall（通常来说，这两个值负相关，即Precision越高，Recall越低），如果将Recall作为横坐标，Precision作为纵坐标，可形成一条Precision-Recall曲线：

![](/home/jimzeus/outputs/AANN/images/media/image309.png){width="3.5972222222222223in"
height="2.138888888888889in"}

AP指的是在不同Recall值的情况下，Precision的平均值。在PASCAL
VOC数据集中，是将Recall分别0，0.1，0.2...0.9，1.0的11种情况下的Precision取平均值，得到AP。

#### mAP

**平均精度均值，mean Average Precision**，AP在不同类别上的均值。

参考AP的定义，mAP指的是不同类别的AP的平均值。也就是说，A指的是同一类别不同置信度的平均，m指的是不同类别的平均。

在某些语境下（比如COCO数据集中），AP就表示mAP，比如AP@.75，表示在IoU为0.75的情况下，所有类别的Precision的平均值。

#### 阈值：置信度和IoU

置信度的阈值调整通常会使得Precision和Recall此消彼长，即：

-   置信度阈值升高，则FP和TP降低，通常导致Precision升高，Recall降低。

-   置信度阈值降低，则FP和TP升高，通常导致Precision降低，Recall升高。

而IoU的阈值调整则使得整个P-R曲线和bounding box可用性此消彼长，即：

-   IoU阈值降低，则P-R曲线整体向右上移动（Precision和Recall都更优），而bounding
    box可用性变差。

-   IoU阈值升高，则P-R曲线整体向左下移动（Precision和Recall都更差），而bounding
    box可用性变好。

但IoU阈值通常只取几个固定典型值（0.5, 0.75,
0.95等），因为对于特定任务来说，过差的IoU没有意义（想一下IoU为0.1，可能识别出来的汽车bounding
box中只有某一角是车的一部分）

### 数据集

#### PASCAL VOC

PASCAL（Pattern Analysis, Statistical Modelling and Computational
Learning），VOC（Visual Object Classes）。

PASCAL VOC竞赛目标主要是目标检测。第一届PASCAL
VOC举办于2005年，然后每年一届，于2012年终止。其提供的数据集里包括了20类的物体：

-   person

-   bird, cat, cow, dog, horse, sheep

-   aeroplane, bicycle, boat, bus, car, motorbike, train

-   bottle, chair, dining table, potted plant, sofa, tv/monitor

PASCAL VOC的主要2个任务是(按照其官方网站所述，实际上是5个)：

-   分类： 对于每一个分类，判断该分类是否在测试照片上存在（共20类）；

-   检测：检测目标对象在待测试图片中的位置并给出矩形框坐标（bounding
    box）；

-   Segmentation:
    对于待测照片中的任何一个像素，判断哪一个分类包含该像素（如果20个分类没有一个包含该像素，那么该像素属于背景）；

-   （在给定矩形框位置的情况下）人体动作识别；

-   Large Scale Recognition（由ImageNet主办）。

VOC数据集主要包括VOC2007和VOC2012两部分。

官方网页：

[[http://host.robots.ox.ac.uk/pascal/VOC/]{.underline}](http://host.robots.ox.ac.uk/pascal/VOC/)

排行榜：

[[http://host.robots.ox.ac.uk:8080/leaderboard/main\_bootstrap.php]{.underline}](http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php)

**参考**

[[https://zhuanlan.zhihu.com/p/33405410]{.underline}](https://zhuanlan.zhihu.com/p/33405410)

[[https://blog.csdn.net/u013832707/article/details/80060327]{.underline}](https://blog.csdn.net/u013832707/article/details/80060327)

[[https://blog.csdn.net/mzpmzk/article/details/88065416]{.underline}](https://blog.csdn.net/mzpmzk/article/details/88065416)

##### 格式

对于目标检测来说，每一张图片对应一个xml格式的标注文件。下面是其中一个xml文件的示例：

\<annotation\>

\<folder\>VOC2007\</folder\>

\<filename\>001264.jpg\</filename\>

\<source\>

\<database\>The VOC2007 Database\</database\>

\<annotation\>PASCAL VOC2007\</annotation\>

\<image\>flickr\</image\>

\<flickrid\>194748336\</flickrid\>

\</source\>

\<owner\>

\<flickrid\>Hannibal. Or maybe just Rex.\</flickrid\>

\<name\>Minky Paw\</name\>

\</owner\>

\<size\>

\<width\>500\</width\>

\<height\>375\</height\>

\<depth\>3\</depth\>

\</size\>

\<segmented\>0\</segmented\>

\<object\>

\<name\>motorbike\</name\>

\<pose\>Unspecified\</pose\>

\<truncated\>0\</truncated\>

\<difficult\>0\</difficult\>

\<bndbox\>

\<xmin\>69\</xmin\>

\<ymin\>25\</ymin\>

\<xmax\>447\</xmax\>

\<ymax\>348\</ymax\>

\</bndbox\>

\</object\>

\</annotation\>

在这个xml格式中：

-   bndbox是一个轴对齐的矩形，它框住的是目标在照片中的可见部分。

-   truncated表明这个目标因为各种原因没有被框完整（被截断了），比如说一辆车有一部分在画面外。

-   occluded是说一个目标的重要部分被遮挡了（不管是被背景的什么东西，还是被另一个待检测目标遮挡）。

-   difficult表明这个待检测目标很难识别，有可能是虽然视觉上很清楚，但是没有上下文的话还是很难确认它属于哪个分类；标为difficult的目标在测试成绩的评估中一般会被忽略。

VOC数据集的目录结构如下，VOC2012和VOC2007一样：

-   Annotations：存放上面提到的目标检测的标识xml

-   ImageSets：存放各种数据集的划分，txt格式，内容是图片文件名

    -   Main：目标检测相关的数据集划分，例如train.txt中为所有的训练数据集图片，cat\_val.txt为所有猫分类的验证数据集图片（这种单分类训练集划分里，每行最后的1/-1应为该类的正负样本）

    -   Action：人体动作识别数据集的划分

    -   Layout：

    -   Segmentation：分割训练数据集的划分

-   JPEGIMages：存放所有原始的图片文件

-   SegmentationClass：语义分割掩图图片

-   SegmentationObject：实例分割掩图图片

#### COCO

COCO的 全称是Common Objects in
COntext，是微软团队提供的一个可以用来进行图像识别的数据集。MS
COCO数据集中的图像分为训练、验证和测试集。

官网：

[[http://cocodataset.org/\#home]{.underline}](http://cocodataset.org/#home)

**参考**

[[https://zhuanlan.zhihu.com/p/29393415]{.underline}](https://zhuanlan.zhihu.com/p/29393415)

##### 格式

COCO的目录格式如下：

-   annotation：标识

-   common：

-   images：原始图片

    -   train2014：2014训练集

    -   val2014：2014验证集

-   LuaAPI：

-   MatlabAPI：

-   PythonAPI：

-   results：

### 代码

目标检测CNN的作者很多都来自Facebook（Ross
Girshick，何恺明等），YOLOv1也有Facebook的参与，因此目标检测的专业库里，FB的Detectron2涵盖了较多的实现。

**代码**

[[https://github.com/facebookresearch/Detectron2]{.underline}](https://github.com/facebookresearch/Detectron2)

### R-CNN系列

传统的CNN最主流的应用场景是图像分类（Image
Classification），通过若干层的卷积层提取出图像的特征，最后通过全连接层得到若干种分类的可能性。而在目标检测（Object
Detection）任务中，R-CNN系列是早期主要的技术演进路线：

**R-CNN** -\> **SPP-net** -\> **Fast R-CNN** -\> **Faster R-CNN** -\>
**R-FCN**

目标检测的一个最简单直观的办法就是：对图片的不同子区域进行类似图像分类CNN所进行的操作，但是这个办法的问题在于计算量太大。

因此R-CNN系列神经网络登上了历史的舞台，先直观的了解一下这几个神经网络的处理时间，单位是**秒**：

![https://cdn-images-1.medium.com/max/1600/1\*4gGddZpKeNIPBoVxYECd5w.png](/home/jimzeus/outputs/AANN/images/media/image310.png){width="4.042361111111111in"
height="1.9958333333333333in"}

R-CNN系列的特点是所谓two-stage方法：先通过region proposal（selective
search算法，或者CNN）确定若干候选框，然后对这些候选框进行分类（确定类别）和回归（确定边界）。

#### R-CNN（201311）

为了减少需要通过CNN的图像的数量，R-CNN的流程如下：

1.  首先通过region proposal（具体是**selective
    search**算法）选取了2000个候选区域（**RoI，Region of Interest**）。

2.  将这些候选区域缩放为统一的大小（227\*227），然后送入CNN分类器

3.  这个CNN分类器提取出4096个特征，这些特征被分别送入两个地方：

    a.  N个SVM分类器决定其所属物体类别（每一个类别对应一个SVM，用于判断是否属于该类别）；

    b.  一个岭回归器（ridge regression），用来修正物体的位置。

4.  使用贪心的非极大值抑制（NMS）合并候选区域，得到输出结果

Selective search算法通过颜色和纹理的不断合并得到候选区域。

下图为R-CNN的结构图：

![](/home/jimzeus/outputs/AANN/images/media/image311.jpeg){width="4.363888888888889in"
height="1.3777777777777778in"}

R-CNN的主要问题是其计算量仍然很大，2000个RoI都需要走一遍CNN，导致其对单张图片的操作时间达到了40多秒，完全无法用于实时检测。

Selective Search 的区域的合并规则是：

-   优先合并颜⾊相近的

-   优先合并纹理相近的

-   优先合并合并后总⾯积⼩的

-   合并后，总⾯积在其BBOX中所占⽐例⼤的优先合并

下图是通过Selective Search得到的一组候选区域

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image312.jpeg){width="4.583333333333333in"
height="1.7131944444444445in"}

**参考**

[[https://zhuanlan.zhihu.com/p/42731634]{.underline}](https://zhuanlan.zhihu.com/p/42731634)

**论文**

**R-CNN**：Rich feature hierarchies for accurate object detection and
semantic segmentation Tech report (v5)

[[https://arxiv.org/pdf/1311.2524]{.underline}](https://arxiv.org/pdf/1311.2524)

**selective search**：

[[http://www.huppelen.nl/publications/selectiveSearchDraft.pdf]{.underline}](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)

#### SPP-net（201406）

在R-CNN系列网络演进的过程中，**SPP（Spatial Pyramid
Pooling，空间金字塔池化）**的思想对其影响很大。SPP层具体请参考《[[网络构成
\> DNN/CNN微结构 \> SPP层]{.underline}](\l)》节，SPP-net的结构如下：

![https://img-blog.csdn.net/20170618165250909?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdjFfdml2aWFu/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center](/home/jimzeus/outputs/AANN/images/media/image313.png){width="4.384722222222222in"
height="2.7423611111111112in"}

SPP层是介于特征层（卷积层的最后一层）和全连接层之间的一种pooling结构。

**参考**

[[https://zhuanlan.zhihu.com/p/42732128]{.underline}](https://zhuanlan.zhihu.com/p/42732128)

**论文**

Spatial Pyramid Pooling in Deep Convolutional Networks for Visual
Recognition

[[https://arxiv.org/pdf/1406.4729]{.underline}](https://arxiv.org/pdf/1406.4729)

#### Fast R-CNN（201504）

**Fast
R-CNN**是在**R-CNN**的基础上，结合了SPP层的特点，做出了改进，流程如下：

1.  通过region proposal在图像上选出1000-2000个RoI

2.  对整个图像卷积一次（而不是每个ROI都做一次卷积）

3.  在卷积后的特征图上找到对应的RoI，并将它们的特征图送入RoI
    Pooling（RoI池化层，即SPP层），输出固定维度的特征值。

4.  从RoI池化层输出的特征值被送入FC层，之后接两个输出

    a.  FC层+Softmax函数作为分类器，用于区分物体类别

    b.  FC层+边界框回归器，用于确定物体位置

相较于R-CNN，做出的改动有：

1.  region proposal放在CNN之后，只卷积一次 ，从整体特征图中获得RoI特征图

2.  不缩放图像，而是用RoI Pooling层获得特征值

3.  最后的分类器由SVM变为FC层(Softmax)

结构图如下：

![](/home/jimzeus/outputs/AANN/images/media/image314.jpeg){width="4.221527777777778in"
height="1.2819444444444446in"}

**参考**

[[https://zhuanlan.zhihu.com/p/42738847]{.underline}](https://zhuanlan.zhihu.com/p/42738847)

**论文**

Fast R-CNN

[[https://arxiv.org/pdf/1504.08083]{.underline}](https://arxiv.org/pdf/1504.08083)

代码：

[[https://github.com/rbgirshick/fast-rcnn]{.underline}](https://github.com/rbgirshick/fast-rcnn)（已废弃，现在Detectron2中）

#### Faster R-CNN（201506）

Fast R-CNN的操作时间降到了2.3秒，但是其中有2秒左右的时间都是在进行region
proposal。因此想要进一步提高速度，就要找出一个更高效率的RoI搜索算法，这个算法就是**RPN**。

**RPN（Region Proposal Network）**是Faster
R-CNN所使用的RoI搜索器，是一个CNN，具体在Faster
R-CNN论文中使用的是**ZFNet**或**VGG**。RPN将第一个CNN的输出作为输入，吐出256维的特征值，这个特征值被送入两个独立的FC层，一个用于做有物体/无物体的分类器，另一个则用于预测边框。

RPN结构如下：

![](/home/jimzeus/outputs/AANN/images/media/image315.jpeg){width="4.407638888888889in"
height="1.2125in"}

这样整个Faster R-CNN都是由CNN实现的，实现了端到端，整个Faster
R-CNN的流程：

-   对整张图像进行卷积，得到特征图

-   卷积特征输出到RPN，得到候选框的特征信息

-   对候选框中提取出的特征，分别送去做分类和边框预测

相较于Fast R-CNN，Faster
R-CNN的改进就是使用**RPN**替代了之前的**selective search**作为**region
proposal**算法，因此大幅提高了速度。整体结构如下：

![](/home/jimzeus/outputs/AANN/images/media/image316.jpeg){width="4.227083333333334in"
height="1.2965277777777777in"}

**论文**

Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
Networks

[[https://arxiv.org/pdf/1506.01497]{.underline}](https://arxiv.org/pdf/1506.01497)

**参考**

[[https://zhuanlan.zhihu.com/p/31426458]{.underline}](https://zhuanlan.zhihu.com/p/31426458)

[https://zhuanlan.zhihu.com/p/24916624]{.underline}

[[https://zhuanlan.zhihu.com/p/42741973]{.underline}](https://zhuanlan.zhihu.com/p/42741973)

#### R-FCN（201605）

**论文**

[[https://arxiv.org/pdf/1605.06409.pdf]{.underline}](https://arxiv.org/pdf/1605.06409.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/42858039]{.underline}](https://zhuanlan.zhihu.com/p/42858039)

### YOLO系列

**You Only Look
Once**，YOLO之前的目标检测方法（主要是R-CNN系列）都分成两步，先通过regional
proposal产生若干可能包含物体的边界框，再通过分类器判断是具体类别，以及回归器来精确定位物体的边界。而正如其名所示，YOLO的特点是通过一次卷积过程完成了整个定位和分类的过程。

YOLO的准确率（**定位误差**和**召回率**）比R-CNN系列低，优点是速度快（高一个数量级），以及**假阳性（FP）**更低。YOLO也是一个系列，包括V1、V2和V3。

代码：

原作者redmon版本：[[https://github.com/pjreddie/darknet]{.underline}](https://github.com/pjreddie/darknet)

Alexy工程化版本：[[https://github.com/AlexeyAB/darknet]{.underline}](https://github.com/AlexeyAB/darknet)

第三方团队Pytorch复现版：[[https://github.com/ultralytics/yolov3]{.underline}](https://github.com/ultralytics/yolov3)

#### YOLOv1（201506）

相较于R-CNN系列，YOLO V1的优势在于：

-   计算速度更快（45FPS，能达到Faster R-CNN的若干倍）

-   假阳性（False Positive）更低

-   由于其从整体着眼，泛化能力更强，在艺术作品中有更好的效果

但其缺点也很明显：

-   准确率其实更低，且边界框更不准。

-   由于其原理限制，能预测的物体数量有限

YOLO将图像分成S\*S个格子，如果一个物体的中心落在某格子内，则该格子负责预测该物体，每个格子可以预测B个物体，每个物体用5个值用来描述：x,y,w,h,confidence，（x,y）是该物体的中心位置，w和h是物体的宽和高，confidence是该物体的置信度。此外每个格子有C个值来描述分类，每个值表示一个物体分类（注意是每个格子，不是每个物体）。

因此YOLO的输出张量为S\*S\*(B\*5+C)，通常S为7，B为2，C则为20（PASCAL
VOC数据集），整个输出张量为7\*7\*30。

以下为YOLO的网络结构，24层卷积+2层全连接：

![](/home/jimzeus/outputs/AANN/images/media/image317.png){width="5.768055555555556in"
height="2.3680555555555554in"}

首先用前20层卷积层+1个平均池化层+1个全连接层，在ImageNet上进行分类训练，最后达到top-5有88%的正确率。

接着将后4层卷积层+2层全连接层接到之前训练好的20层卷积层之后，形成完整的网络，同时将输入分辨率从224\*224提升到448\*448。

因为在图片的大部分格子中没有物体，这部分权重占比太大，导致网络训练发散，为了避免这种情况，在训练中给有物体的格子和没有物体的格子的损失分配了不同的权重（分别为5和0.5）。

对于不同大小的物体，同样的宽高误差显然有不同的意义，更小的物体对宽高误差更敏感。为了应对这种情况，在损失函数中使用w和h的平方根而非本身来衡量误差。

以下是损失函数，其中：

-   第一行是中心点坐标的误差（S\*S\*B\*2）

-   第二行是宽和高的误差（S\*S\*B\*2）

-   第三行是有物体的格子的置信度误差

-   第四行是没有物体的格子的置信度的误差

-   最后一行是类别的误差（S\*S\*B\*C）

![](/home/jimzeus/outputs/AANN/images/media/image318.png){width="4.177777777777778in"
height="2.845833333333333in"}

**论文**

You Only Look Once: Unified, Real-Time Object Detection

[[https://arxiv.org/pdf/1506.02640]{.underline}](https://arxiv.org/pdf/1506.02640)

#### YOLOv2（201612）

YOLO V2的出现是为了解决YOLO
V1的两个主要的问题：定位准确率较低，以及较低的识别率（TPR）。目的是在不增加网络复杂度的情况下提高这些值，为此文章中使用了一系列手段来实现，或者说，YOLO
V2和YOLO
V1的区别就是一系列细节，下表中显示了各种不同手段应用在YOLO上之后带来的mAP上的变化：

![](/home/jimzeus/outputs/AANN/images/media/image319.png){width="5.768055555555556in"
height="2.25in"}

这些手段包括：

**Batch Normalization**

BN可以提高模型收敛速度，且可以降低模型的过拟合，YOLOv2中每个卷积层后面都增加了BN，且不再使用dropout。

**高分辨率分类器**

High-resolution
classifier，每个目标检测网络都会使用经ImageNet预训练的分类器，YOLOv1在训练的时候先在224\*224的图片上训练分类器，再在448\*448的图片上训练目标检测，这意味着网络需要同时应用低分辨率到高分辨率，以及分类到检测的两个转变。对于YOLOv2，在这两者之间增加了10个epoch的448\*448的图片分类训练做fine
tune，然后在转到448\*448的目标检测做fine-tune。

**带先验框的卷积**

锚点（anchor box）即先验框，这个概念是Faster
R-CNN中提出的。YOLOv1直接预测每个边界框的坐标，而YOLOv2则借鉴了先验框的概念，预测相对于先验框的偏移值，这样可以更好的学习。并且通过设置不同长宽比的先验框，对于不同长宽比的物体可以有很高的预测结果（这也是YOLOv1
TPR低的一个原因）。

YOLOv2去除了v1中的全连接层，改用卷积层和先验框来预测边界框。为了使检测所用的特征图分辨率更高，移除了一个池化层。并且不使用448\*448而是416\*416作为输入，这使得全部下采样（总步长32）之后得到的特征图大小为13\*13，这样只有一个中心位置，便于预测中心点在中间的大物体。

YOLOv1将图像分成7\*7的格子（也即其最终特征图的大小），每格可以有2个边界框（每个边界框有一个置信度来表示是否有物体，但这两个边界框共享一个分类预测），也就是说YOLOv1最多只能预测98个物体。

YOLOv2则大大提高了预测物体的数量，首先有13\*13个预测位置，并且每个位置有若干个先验框，每个先验框都有其对应的置信度（有无物体）和分类预测。与SSD不同，YOLO使用置信度来分辨有无物体，而SSD将背景作为一个单独的分类，因此SSD的分类预测数量是比实际的类别数量多1。

![](/home/jimzeus/outputs/AANN/images/media/image320.png){width="4.045833333333333in"
height="2.1118055555555557in"}

**维度聚类**

Dimension clusters，在Faster
R-CNN和SSD中先验框的维度（长和宽）都是手动设定的，带有一定的主观性。如果选取的先验框维度比较合适，那么模型更容易学习，从而做出更好的预测。因此，YOLOv2采用**k-means聚类**方法对训练集中的边界框做了聚类分析。因为设置先验框的主要目的是为了使得预测框与ground
truth的IOU更好，所以聚类分析时选用box与聚类中心box之间的IOU值作为距离指标：

![](/home/jimzeus/outputs/AANN/images/media/image321.png){width="3.6569444444444446in"
height="0.33541666666666664in"}

下图为在VOC和COCO数据集上的聚类分析结果，随着聚类中心数目的增加，平均IOU值（各个边界框与聚类中心的IOU的平均值）是增加的，但是综合考虑模型复杂度和召回率，作者最终选取5个聚类中心作为先验框，其相对于图片的大小如右侧图所示：

![](/home/jimzeus/outputs/AANN/images/media/image322.png){width="3.4458333333333333in"
height="1.8in"}

**新的网络**

YOLOv2采用了一个新的基础网络用作特征提取，被称为Darknet-19，包括19个卷积层和5个最大池化层：

![](/home/jimzeus/outputs/AANN/images/media/image323.png){width="2.98125in"
height="3.545138888888889in"}

Darknet-19与VGG16模型设计原则是一致的，主要采用3\*3卷积，采用2\*2的maxpooling层之后，特征图维度降低2倍，而同时将特征图的channles增加两倍。与NIN(Network
in Network)类似，Darknet-19最终采用global
avgpooling做预测，并且在3\*3卷积之间使用1\*1卷积来压缩特征图channles以降低模型计算量和参数。Darknet-19每个卷积层后面同样使用了batch
norm层以加快收敛速度，降低模型过拟合。在ImageNet分类数据集上，Darknet-19的top-1准确度为72.9%，top-5准确度为91.2%，但是模型参数相对小一些。使用Darknet-19之后，YOLOv2的mAP值没有显著提升，但是计算量却可以减少约33%。

**直接位置预测**

Direct location prediction，直接Anchor
Box回归导致模型不稳定，对应公式也可以参考
Faster-RCNN论文，该公式没有任何约束，中心点可能会出现在图像任何位置，这就有可能导致回归过程震荡，甚至无法收敛。

针对这个问题，作者在预测位置参数时采用了强约束方法：

       1）对应 Cell
距离左上角的边距为（Cx，Cy），σ定义为sigmoid激活函数，将函数值约束到［0，1］，用来预测相对于该Cell
中心的偏移（不会偏离cell）；

       2）预定Anchor（文中描述为bounding box
prior）对应的宽高为（Pw，Ph），预测 Location 是相对于Anchor的宽高
乘以系数得到

**细粒度特征**

YOLOv2的输入图片大小为416\*416，经过5次maxpooling之后得到13\*13大小的特征图，并以此特征图采用卷积做预测。13\*13大小的特征图对检测大物体是足够了，但是对于小物体还需要更精细的特征图（Fine-Grained
Features）。因此SSD使用了多尺度的特征图来分别检测不同大小的物体，前面更精细的特征图可以用来预测小物体。YOLOv2提出了一种**passthrough**层来利用更精细的特征图。YOLOv2所利用的Fine-Grained
Features是26\*26大小的特征图（最后一个maxpooling层的输入），对于Darknet-19模型来说就是大小为26\*26\*512的特征图。passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上。前面的特征图维度是后面的特征图的2倍，passthrough层抽取前面层的每个2\*2的局部区域，然后将其转化为channel维度，对于26\*26\*512的特征图，经passthrough层处理之后就变成了13\*13\*2048的新特征图（特征图大小降低4倍，而channles增加4倍，图6为一个实例），这样就可以与后面的13\*13\*1024特征图连接在一起形成13\*13\*3072的特征图，然后在此特征图基础上卷积做预测。在YOLO的C源码中，passthrough层称为reorg
layer。在TensorFlow中，可以使用tf.extract\_image\_patches或者tf.space\_to\_depth来实现passthrough层

**多尺度训练**

由于YOLOv2模型中只有卷积层和池化层，所以YOLOv2的输入可以不限于416\*416大小的图片。为了增强模型的鲁棒性，YOLOv2采用了多尺度输入训练策略，具体来说就是在训练过程中每间隔一定的iterations之后改变模型的输入图片大小。由于YOLOv2的下采样总步长为32，输入图片大小选择一系列为32倍数的值输入图片。最小为320\*320，此时对应的特征图大小为10\*10，而输入图片最大为608\*608，对应的特征图大小为19\*19,在训练过程，每隔10个iterations随机选择一种输入图片大小，然后只需要修改对最后检测层的处理就可以重新训练。

采用Multi-Scale
Training策略，YOLOv2可以适应不同大小的图片，并且预测出很好的结果。在测试时，YOLOv2可以采用不同大小的图片作为输入，在VOC
2007数据集上的效果如下图所示。可以看到采用较小分辨率时，YOLOv2的mAP值略低，但是速度更快，而采用高分辨输入时，mAP值更高，但是速度略有下降，对于544\*544,mAP高达78.6%。注意，这只是测试时输入图片大小不同，而实际上用的是同一个模型（采用Multi-Scale
Training训练）。 

![](/home/jimzeus/outputs/AANN/images/media/image324.png){width="3.3513888888888888in"
height="1.9708333333333334in"}

**论文**

YOLO9000: Better, Faster, Stronger

[[https://arxiv.org/pdf/1612.08242]{.underline}](https://arxiv.org/pdf/1612.08242)

参考:

[[https://blog.csdn.net/shanlepu6038/article/details/84778770]{.underline}](https://blog.csdn.net/shanlepu6038/article/details/84778770)

[[https://blog.csdn.net/anqian123321/article/details/82627332]{.underline}](https://blog.csdn.net/anqian123321/article/details/82627332)

#### YOLOv3（201804）

YOLOv3在YOLOv2的基础上做了一些改动，包括如下：

**新的基础网络Darknet-53**

相较于上一版本中的Darknet19，Darknet53通过加入残差结构，大大增加了网络深度。此外YOLOv3中是完全没有**池化层**和**全连接层**的，张量的尺寸变化是通过改变卷积核的步长来是实现的。

![](/home/jimzeus/outputs/AANN/images/media/image325.png){width="3.8680555555555554in"
height="5.696527777777778in"}

**多尺度检测**

Predictions across
scales，YOLOv2中使用passthrough结构来进行检测细粒度特征，在v3中得到了进一步发扬光大，吸取了FPN的特点，YOLOv3输出3个不同尺度的特征图，分别是13\*13，26\*26，52\*52，

**论文**

YOLOv3: An Incremental Improvement

[[https://arxiv.org/pdf/1804.02767]{.underline}](https://arxiv.org/pdf/1804.02767)

**代码**第三方团队在Pytorch上复现的YOLOv3

[[https://github.com/ultralytics/yolov3]{.underline}](https://github.com/ultralytics/yolov3)

#### YOLOv4（202004）

YOLOv3之后其作者Joseph
Redmon宣布退出CV研究（因为他觉得自己的开源算法被用在军事上），于是YOLOv4的作者Alexey就"接手"了之后的开发。Alexey跟Redmon认识，而且之前就做了很多YOLO的工作，主要是一些工程化方面的东西，比如移植到Windows、fix
bug、增加模型等等，Alexey的github库也是除官方之外最权威的。

YOLOv4中没有之前几个版本的YOLO的创新性的改进，更多的贡献是在工程化的方面的，在YOLOv3之上测试并应用了很多新出来的技术（数据增强、激活函数、Loss函数、dropout等等）。

**论文**

[[https://arxiv.org/pdf/2004.10934.pdf]{.underline}](https://arxiv.org/pdf/2004.10934.pdf)

**代码**Alexey版本的各个YOLO

[[https://github.com/AlexeyAB/darknet]{.underline}](https://github.com/AlexeyAB/darknet)

**参考**

[[https://www.zhihu.com/question/390194081]{.underline}](https://www.zhihu.com/question/390194081)

[[https://zhuanlan.zhihu.com/p/135899403]{.underline}](https://zhuanlan.zhihu.com/p/135899403)

#### YOLOv5（Ultralytics）

YOLOv5并不是一篇论文。

如果说YOLOv3是官方（Joseph
Redmon），YOLOv4是半官方（AlexeyAB），那YOLOv5就是民间的了，其作者是Ultralytics，这家公司之前就一直在做YOLO上的各种工作，包括移植到Pytorch上，之前对YOLOv3的移植在github上也获得了各种移植版本中最高的star：[[https://github.com/ultralytics/yolov3]{.underline}](https://github.com/ultralytics/yolov3)

事实上v5这个称号也是它自封的，和YOLOv4一样，YOLOv5也是在YOLOv3之上加了各种tricks形成的，但是并没有发表论文，具体的tricks目前尚未看到介绍。

和YOLOv4最大的区别在于它是Pytorch实现的。

**代码**

[[https://github.com/ultralytics/yolov5]{.underline}](https://github.com/ultralytics/yolov5)

### SSD（201512）

SSD同YOLO一样，是个一步检测的框架，其特点在于：

-   相较于YOLO，提供了更高的准确性，可以达到和两步检测方法（比如Faster
    R-CNN）差不多的水准（SSD出来的时候YOLOv2尚未出来）

-   SSD的关键是用一组缺省的边界框来预测类别置信度和边界，这些缺省边界框是通过将小的卷积核应用在特征图上得出的

-   在不同尺度的特征图上得到缺省边界框，以及不同宽高比的边界框

-   以上设计使得整个网络是端到端的，易于训练，高准确率，即便是在低分辨率的图上，进一步提高了速度/准确性

SSD使用一个传统的图像分类网络（这里是VGG）作为基础网络，最后用于分类的层被去掉，此外全连接层FC6和FC7被替换为卷积层，还增加了以下**辅助结构**来产生目标检测结果：

-   通过在不同尺寸的**特征图**（feature
    map，即不同卷积层）上追加**特征层（feature
    layer）**来实现对不同尺寸物体的检测：在大的（即靠前的）特征图上预测小物体，在小的（靠后的）特征图上预测大的物体。

-   这些特征层可以产生固定数量的预测。特征层是卷积层，一个m\*n\*p的特征层（p为通道数）所对应的卷积核为3\*3\*p。对于每个预测位置（m\*n个）的每个通道（p个），会产生某个类别的置信度，或者对位置的预测（x,y,w,h）

-   每个预测位置可以产生不同长宽比的若干个（k个）先验框，与单个先验框相关的输出有c+4个，c为类别数，表示每个类别的置信度，4为坐标，分别为中心位置（x,y）和长宽（w,h）。因此共有(c+4)\*k个卷积核被应用到特征图的每个预测位置上，从而对于每个特征层，有p=(c+4)\*k个通道。这个先验框（default
    box）的想法来自于Faster R-CNN中的锚点（anchor box）

下图为SSD的结构，以及YOLO的结构

![](/home/jimzeus/outputs/AANN/images/media/image326.png){width="5.768055555555556in"
height="3.0520833333333335in"}

再用简单的描述总结一下：

-   基础网络上的N层（特征图）可以产生若干预测，越靠前的层预测越小的物体，越靠后的层预测越大的物体；

-   每层有m\*n个位置（特征图大小）；

-   每个位置可以产生k个预测（不同长宽比的先验框）；

-   每个先验框包括c+4个输出（c为类别数），4对应每个类别的置信度和中心坐标（x,
    y）和长宽（w, h）。

每个特征图总共有m\*n\*k\*(c+4) 个输出

**先验框匹配策略**

在训练过程中，首先要确定训练图片中的ground
truth（真实目标）与哪个先验框来进行匹配，与之匹配的先验框所对应的边界框将负责预测它。在Yolo中，ground
truth的中心落在哪个单元格，该单元格中与其IOU最大的边界框负责预测它。但是在SSD中却完全不一样，SSD的先验框与ground
truth的匹配原则主要有两点。

-   对于图片中每个ground
    truth，找到与其IOU最大的先验框，该先验框与其匹配，这样，可以保证每个ground
    truth一定与某个先验框匹配。通常称与ground
    truth匹配的先验框为正样本，反之，若一个先验框没有与任何ground
    truth进行匹配，那么该先验框只能与背景匹配，就是负样本。

-   第二个原则是，对于剩余的未匹配先验框，若某个ground
    truth的IOU大于某个阈值（一般是0.5），那么该先验框也与这个ground
    truth进行匹配。这个原则简化了学习过程。而这意味着某个ground
    truth可能与多个先验框匹配，这是可以的。但是反过来却不可以，因为一个先验框只能匹配一个ground
    truth，如果多个ground
    truth与某个先验框IOU大于阈值，那么先验框只与IOU最大的那个先验框进行匹配。第二个原则一定在第一个原则之后进行，仔细考虑一下这种情况，如果某个ground
    truth所对应最大IOU小于阈值，并且所匹配的先验框却与另外一个ground
    truth的IOU大于阈值，那么该先验框应该匹配谁，答案应该是前者，首先要确保某个ground
    truth一定有一个先验框与之匹配。但是，这种情况我觉得基本上是不存在的。由于先验框很多，某个ground
    truth的最大IOU肯定大于阈值，所以可能只实施第二个原则既可以了。

**损失函数**

损失函数定义为位置误差（loc）与置信度误差（conf）的加权和：

![](/home/jimzeus/outputs/AANN/images/media/image327.png){width="3.736111111111111in"
height="0.6180555555555556in"}

其中N是先验框的正样本数量。这里xpij∈{1,0}为一个指示参数，当xpij=1时表示第i个先验框与第j个ground
truth匹配，并且ground
truth的类别为p。c为类别置信度预测值。l为先验框的所对应边界框的位置预测值，而g是ground
truth的位置参数。对于位置误差，其采用Smooth L1 loss，定义如下

![](/home/jimzeus/outputs/AANN/images/media/image328.png){width="4.205555555555556in"
height="1.5013888888888889in"}

![](/home/jimzeus/outputs/AANN/images/media/image329.png){width="3.3847222222222224in"
height="0.7201388888888889in"}

**数据增广**

采用**数据扩增**（**Data
Augmentation**）可以提升SSD的性能，主要采用的技术有水平翻转（horizontal
flip），随机裁剪加颜色扭曲（random crop & color
distortion），随机采集块域（Randomly sample a
patch）（获取小目标训练样本）等。

**论文**

SSD: Single Shot MultiBox Detector

[[https://arxiv.org/pdf/1512.02325]{.underline}](https://arxiv.org/pdf/1512.02325)

**参考**

[[https://blog.csdn.net/xiaohu2022/article/details/79833786]{.underline}](https://blog.csdn.net/xiaohu2022/article/details/79833786)

### FPN (201612)(TODO)

**特征金字塔网络**（**Feature Pyramid Network**）

**论文**

[[https://arxiv.org/abs/1612.03144]{.underline}](https://arxiv.org/abs/1612.03144)

**参考**

代码：

### RetinaNet（201708）

RetinaNet的主要贡献是**Focal Loss**，参考《[[网络构成 \> 损失函数 \>
Focal Loss]{.underline}](\l)》

**论文**

[[https://arxiv.org/pdf/1708.02002.pdf]{.underline}](https://arxiv.org/pdf/1708.02002.pdf)

### MaskX R-CNN (201711)(TODO)

**论文**

[[https://arxiv.org/pdf/1711.10370.pdf]{.underline}](https://arxiv.org/pdf/1711.10370.pdf)

### CenterNet (201904)(TODO)

**论文**

[[https://arxiv.org/pdf/1904.07850.pdf]{.underline}](https://arxiv.org/pdf/1904.07850.pdf)

代码：

[[https://github.com/xingyizhou/CenterNet]{.underline}](https://github.com/xingyizhou/CenterNet)

### EfficientDet (201911)(TODO)

EfficientDet是以EfficientNet为backbone实现的目标检测网络

**论文**

[[https://arxiv.org/pdf/1911.09070.pdf]{.underline}](https://arxiv.org/pdf/1911.09070.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/129016081]{.underline}](https://zhuanlan.zhihu.com/p/129016081)

代码：

[[https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch]{.underline}](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

图像分割（Image Segmentation）
------------------------------

**图像分类**识别整个图片，**目标检测**识别图片中的物体，**图像分割**则是更进一步的任务------它的精细度到了单个像素，图像分割的目的是找出图片中各个物体的轮廓。

图像分割有几个子任务：

-   语义分割（Semantic Segmentation）：按照物体的类别分割

-   实例分割（Instance Segmentation）：按照单个物体进行分割

-   全景分割（Panoptic Segmentation）：前景实例分割+背景语义分割

请注意区别**语义分割（Semantic Segmentation）**和**实例分割（Instance
Segmentation）**的区别，语义分割按类别区分，而实例分割按个体区分，例如，当图像中有多只猫时，语义分割会将多只猫整体的所有像素预测为"猫"这个类别。与此不同的是，实例分割需要区分出哪些像素属于第一只猫、哪些像素属于第二只猫。下图中左下为语义分割，右下为实例分割：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562316402(1).png](/home/jimzeus/outputs/AANN/images/media/image330.png){width="4.054166666666666in"
height="3.1902777777777778in"}

一般的图像分割架构可以被认为是一个**编码器-解码器**网络，编码器通常是一个预训练的分类网络（VGG、ResNet等），后面接一个解码器网络，它的任务是将编码器学到的低分辨率的可判别特征投射到高分辨率的像素空间，以获得清晰的类别区分。

**参考**

[[https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html]{.underline}](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html)

上文翻译：[[https://blog.csdn.net/qq\_20084101/article/details/80432960]{.underline}](https://blog.csdn.net/qq_20084101/article/details/80432960)

### FCN（201411）

**FCN（全卷积网络，Fully Convolutional
Network）**在近几年的语义分割论文中有开创性的意义，之后的几篇语义分割相关的网络都是基于FCN的理论基础。FCN是一个**编码器-解码器**网络，编码器是一个预训练的分类网络（VGG、ResNet等），后面接一个解码器网络。

FCN这个概念（网络全都是卷积构成）在其他的研究方向也有应用。

FCN的特点包括：

-   分类网络中的全连接层被替换为全卷积层以保持特征图维度（FC输出形状是一维，卷积层输出形状是二维）

-   输入给**解码器**的特征是由**编码器**的不同阶段合并而成，这些阶段的语义特征有着不同的粗糙程度。

-   低分辨率的语义特征的**上采样**是通过经双线性插值滤波器初始化的**反卷积**实现的

-   从VGG、AlexNet等分类器网络进行知识迁移来实现语义细分

下图可以看到不同粗糙程度的阶段有不同的上采样倍数（FCN-32s、FCN-16s、FCN-8s）

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562303334(1).png](/home/jimzeus/outputs/AANN/images/media/image331.png){width="5.768055555555556in"
height="2.213888888888889in"}

**参考**

[[https://www.cnblogs.com/xuanxufeng/p/6249834.html]{.underline}](https://www.cnblogs.com/xuanxufeng/p/6249834.html)

**论文**

Fully Convolutional Networks for Semantic Segmentation

[[https://arxiv.org/pdf/1605.06211]{.underline}](https://arxiv.org/pdf/1605.06211)

### SegNet（201511）

SegNet
的新颖之处在于解码器对其较低分辨率的输入特征图进行上采样的方式。具体地说，解码器使用了在相应编码器的最大池化步骤中计算的池化索引来执行非线性上采样。这种方法消除了学习上采样的需要。经上采样后的特征图是稀疏的，因此随后使用可训练的卷积核进行卷积操作，生成密集的特征图。我们将我们所提出的架构与广泛采用的
FCN 以及众所周知的 DeepLab-LargeFOV，DeconvNet
架构进行比较。比较的结果揭示了在实现良好的分割性能时所涉及的内存与精度之间的权衡。

简单地说，SegNet的特点是：

-   上采样使用了**反池化层**

其网络架构如下：

![C:\\Users\\AW\\AppData\\Local\\Temp\\1562309334(1).png](/home/jimzeus/outputs/AANN/images/media/image332.png){width="5.768055555555556in"
height="2.120138888888889in"}

**论文**

SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image
Segmentation

[[https://arxiv.org/pdf/1511.00561]{.underline}](https://arxiv.org/pdf/1511.00561)

### DeepLab系列(TODO)

#### DeepLab（201412）

#### DeepLabV2（201606）

#### DeepLabV3（201706）

#### DeepLabV3+（201802）

### Mask R-CNN（201703）

以**Faster R-CNN**
为基础，在现有的边界框识别分支基础上添加一个并行的预测目标掩码的分支。Mask
R-CNN 很容易训练，仅仅在 Faster R-CNN 上增加了一点小开销，运行速度为
5fps。此外，Mask R-CNN
很容易泛化至其他任务，例如，可以使用相同的框架进行姿态估计。我们在 COCO
所有的挑战赛中都获得了最优结果，包括实例分割，边界框目标检测，和人关键点检测。在没有使用任何技巧的情况下，Mask
R-CNN 在每项任务上都优于所有现有的单模型网络，包括 COCO 2016
挑战赛的获胜者。

Mask R-CNN的特点：

-   在**Faster R-CNN** 上添加辅助分支以执行语义分割

-   对每个实例进行的 **RoIPool** 操作被修改为 **RoIAlign** ，它避免了特征提取的空间量化，因为在最高分辨率中保持空间特征不变对于语义分割很重要。

-   **Mask R-CNN** 与 [Feature Pyramid
    Networks](https://arxiv.org/abs/1612.03144)（类似于PSPNet，它对特征使用了金字塔池化）相结合，在 [MS
    COCO](http://mscoco.org/) 数据集上取得了最优结果。

**论文**

Mask R-CNN

[[https://arxiv.org/pdf/1703.06870]{.underline}](https://arxiv.org/pdf/1703.06870)

### PointRend（201912）(TODO)

**论文**

PointRend: Image Segmentations as Rendering

[[https://arxiv.org/pdf/1912.08193.pdf]{.underline}](https://arxiv.org/pdf/1912.08193.pdf)

人脸识别和建模（Face Recognition）
----------------------------------

**Facial Recognition and
Modelling**，人脸相关的神经网络，大致有以下几个部分，这些概念之间有交叉的部分，同样的技术和网络也可能用在不同部分上。

-   **人脸检测（Face
    Detection）**：找出图片中的人脸，输入是图片，输出是人脸的位置和大小。通常情况下其它人脸相关的深度学习会需要**人脸检测**作为第一步。**人脸检测**也可以单独作为一个模块使用（比如拍照时候的自动对焦）。

-   **人脸对齐（Face
    Alignment）**：将人脸图片标准化，包括2D对齐（歪-\>正）和3D对齐（侧脸-\>正脸）。**人脸对齐**的输入是人脸的图片，输出是对齐后的人脸。现在有些神经网络已经不需要人脸对齐这个步骤，而是直接用未对齐的人脸图片做端到端的计算。

-   **人脸验证（Face
    Verification）**：验证两张图片中的人脸是否为同一人（1：1），输入是通常是两张人脸图片（对齐或未对齐），输出是为同一人的概率。具体步骤包括人脸图片的特征提取，以及两个特征值的比对。

-   **人脸识别（Face
    Identification）**：识别出图片中人脸是谁（1：N）。输入是一个人脸图片，以及一个特征值数据库，输出为该人脸是谁（或者谁都不是）的概率向量**？**

-   **人脸聚类（Face
    Clustering）**：找出图片中哪些人脸是同一个人（非监督学习）

-   **表情识别（Facial Expression Recognition）**：识别人脸的表情

相较于以上的概念，**Face
Recognition**倒是一个相对不严格的定义，有时候指代**Face
Identification**，有时候指代**人脸验证**和**Face
Identification**，有时候包括前面**人脸检测**和**人脸对齐**环节，有时候则不包括。

**人脸检测**和**人脸对齐**是其它部分的前导环节，之后通过不同的特征提取手段，在后来的一些人脸相关的NN上，人脸对齐这个阶段有时候会被忽略，直接将检测出来的人脸图片拿去提取特征。

早期的人脸相关的研究，有很多非神经网络的特征提取方法，比如**LBP、HOG、SIFT**等。而后渐渐增大神经网络在整个模型中的比例，这在早期的论文中可以明显的看到NN的部分占比越来越大，现在基本都是用神经网络完成端到端的工作了。

相对于**close-set**人脸识别的数据，对**Open-set**人脸识别的数据，有一个准则，**即最大的intra-class距离要小于最小的inter-class距离**（这个准则应该可以普及到所有open-set分类问题）。这可以通俗的理解为，close-set的数据必然会落入某个分类，因此类与类之间不能有空隙，不然不知道该分入哪个类，而open-set的数据则得留下足够大的空隙，不然应该被标记为"unknown"的陌生数据也会落入某个已知分类。

比较近期的论文（FaceNet，Center
loss，SphereFace，CosFace，ArcFace等）大都是在发掘不同的损失函数，以使得提取出来的特征值不仅是可分离的（Separable），而且特征判别度高（Discriminative），以提高在open-set的情况下的识别能力。这些损失函数要么是通过增加提高类间的**欧式边缘（Euclidean
margin）**，要么是通过增加类间的**角度边缘（angular margin）**来实现。

Github上的一个人脸相关的资料仓库：

[[https://github.com/ChanChiChoi/awesome-Face\_Recognition]{.underline}](https://github.com/ChanChiChoi/awesome-Face_Recognition)

**参考**

[https://blog.csdn.net/tkyjqh/article/details/70139718]{.underline}

[https://paperswithcode.com/task/face-verification]{.underline}

### 数据集

人脸相关数据集介绍

#### LFW

**Labelled Face in the
Wild**，LFW人脸数据库主要用来研究非受限情况下的人脸识别问题。LFW
主要是从互联网上搜集图像，一共含有13000
多张人脸图像，这些人脸图片有着不同的表情、朝向、光照，分别属于5000多个人，每张图像都被标识出对应的人的名字，其中有1680
人对应不只一张图像。因此LFW被广泛用于衡量Face Verification的准确性。

LFW可以随机生成一个pairs.txt文件，包含了6000对图片的ID，其中3000对为同一个人，格式为"名字
n1 n2"，另外3000对为不同的人，格式为"名字1 n1 名字2 n2"

网站：[[http://vis-www.cs.umass.edu/lfw/]{.underline}](http://vis-www.cs.umass.edu/lfw/)

#### YTF（Youtube Faces）

采集自Youtube的脸部视频数据集，包括3000多段视频，其中有1595个人。

官网：[[https://www.cs.tau.ac.il/\~wolf/ytfaces/]{.underline}](https://www.cs.tau.ac.il/~wolf/ytfaces/)

#### CelebA

CelebA是**CelebFaces
Attribute**的缩写，意即名人人脸属性数据集，其包含10,177个名人身份的202,599张人脸图片，每张图片都做好了特征标记，包含人脸bbox标注框、5个人脸特征点坐标以及40个属性标记。这些属性包括"秃头"、"眼镜"、"双下巴"等等。

CelebA由香港中文大学开放提供，广泛用于人脸相关的计算机视觉训练任务，可用于人脸属性标识训练、人脸检测训练以及landmark标记等。

官方网址：[[http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html]{.underline}](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

CelebFaces+

#### FDDB

FDDB（Face Detection Data Set and
Benchmark），FDDB数据集主要用于约束人脸检测研究，该数据集选取野外环境中拍摄的2845个图像，从中选择5171个人脸图像。是一个被广泛使用的权威的人脸检测平台。

[[http://vis-www.cs.umass.edu/fddb/index.html]{.underline}](http://vis-www.cs.umass.edu/fddb/index.html)

#### WIDER

WIDER
FACE是香港中文大学的一个提供更广泛人脸数据的人脸检测基准数据集，由YangShuo，
Luo Ping ，Loy ，Chen Change ，Tang Xiaoou收集。

1.  它包含32203个图像和393703个人脸图像，在尺度，姿势，闭塞，表达，装扮，关照等方面表现出了大的变化。

2.  WIDER
    FACE是基于61个事件类别组织的，对于每一个事件类别，选取其中的40%作为训练集，10%用于交叉验证（cross
    validation），50%作为测试集。

3.  和PASCAL VOC数据集一样，该数据集也采用相同的指标。

4.  和MALF和Caltech数据集一样，对于测试图像并没有提供相应的背景边界框。

-   格式

Annotation包中的wider\_face\_train\_bbox\_gt.txt和wider\_face\_val\_bbox\_gt.txt用于标记脸的位置大小等信息，格式如下：

文件路径

Bbox数量

Bbox1信息

Bbox2信息

...

Bbox信息格式为：x1, y1, w, h, blur, expression, illumination, invalid,
occlusion, pose

官网：

[[http://shuoyang1213.me/WIDERFACE/index.html]{.underline}](http://shuoyang1213.me/WIDERFACE/index.html)

#### AFW

Annotated Face in the
Wild，AFW数据集是使用Flickr（雅虎旗下图片分享网站）图像建立的人脸图像库，包含205个图像，其中有473个标记的人脸。对于每一个人脸都包含一个长方形边界框，6个地标和相关的姿势角度。数据库虽然不大，额外的好处是作者给出了其2012
CVPR的论文和程序以及训练好的模型。

[http://www.ics.uci.edu/\~xzhu/face/]{.underline}

#### AFLW

Annotated Facial Landmarks in the
Wild，AFLW人脸数据库是一个包括多姿态、多视角的大规模人脸数据库，而且每个人脸都被标注了21个特征点。此数据库信息量非常大，包括了各种姿态、表情、光照、种族等因素影响的图片。AFLW人脸数据库大约包括25000已手工标注的人脸图片，其中59%为女性，41%为男性，大部分的图片都是彩色，只有少部分是灰色图片。该数据库非常适合用于人脸识别、人脸检测、人脸对齐等方面的研究，具有很高的研究价值。

[[https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/]{.underline}](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)

#### IJB-A

IJB-A是一个用于人脸检测和识别的数据库，包含24327个图像和49759个人脸。

[[https://www.nist.gov/itl/iad/image-group/ijb-dataset-request-form]{.underline}](https://www.nist.gov/itl/iad/image-group/ijb-dataset-request-form)

#### MegaFace

MegaFace资料集包含一百万张图片，代表690000个独特的人。所有数据都是华盛顿大学从Flickr（雅虎旗下图片分享网站）组织收集的。这是第一个在一百万规模级别的面部识别算法测试基准。
现有脸部识别系统仍难以准确识别超过百万的数据量。为了比较现有公开脸部识别算法的准确度，华盛顿大学在去年年底开展了一个名为"MegaFace
Challenge"的公开竞赛。这个项目旨在研究当数据库规模提升数个量级时，现有的脸部识别系统能否维持可靠的准确率。

[http://megaface.cs.washington.edu/dataset/download.html]{.underline}

### 代码

这里介绍了一些开源的实现，因为都是工程方面的改进，因此没有相关论文

#### 超轻量级人脸检测（201910）

Ultra Light Fast Generic Face
Detector是一个开源的大小只有1MB的人脸检测模型。

介绍及代码如下：

[[https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB]{.underline}](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

### DeepFace（201406）

DeepFace是FaceBook
2014年推出的一篇关于人脸验证的论文。被认为是神经网络使用在人脸识别领域的开山之作。

DeepFace算法分成四个步骤：人脸检测、人脸对齐、特征提取（CNN）、分类。这其中特征提取部分完全由神经网络实现。DeepFace所使用的人脸对齐算法比较复杂，并且在之后的研究中发现，跳过这个步骤，直接将检测到的人脸图片送到特征提取模块，效果也不差，因此在DeepFace之后的论文中大都没有使用人脸对齐这个部分。

在人脸识别领域，DeepFace相较于之前的方法，是**只**使用CNN提取特征，而不是LBP（局部二值模式），HOG等方法。

整个流程的如下：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\7f16cee0fdfee1580c224d6fd100656.png](/home/jimzeus/outputs/AANN/images/media/image333.png){width="5.768055555555556in"
height="1.7569444444444444in"}

第一步人脸检测，第二步人脸对齐，后面是8层CNN：卷积、最大池化、卷积，然后是3层本地连接层，再接2层全连接层。五层连接层的参数数量占了整个网络参数量的95%。

**论文**

**参考**

[https://blog.csdn.net/stdcoutzyx/article/details/46776415]{.underline}

[https://blog.csdn.net/hh\_2018/article/details/80576290]{.underline}

[[https://blog.csdn.net/hh\_2018/article/details/80581612]{.underline}](https://blog.csdn.net/hh_2018/article/details/80581612)

### DeepID系列（2014）

**Deep
ID**是香港中文大学的汤晓欧（也是商汤的创始人）推出的人脸识别算法，一共出了DeepID、DeepID2、DeepID2+、DeepID3几个版本。

#### DeepID（2014）

DeepID的亮点包括：

-   将特征的维数大幅度降低，只用了160维特征

-   使用同一个人脸图片的不同patch（即部分）作为测试集训练，提高了准确性

-   有个Multi-scale的概念，将最后两层卷积层和160维DeepID层连接

-   分别使用了**联合贝叶斯（Joint Bayesian）**和NN作为最后的分类器

DeepID所用的网络整体框架如下：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\c33cffd1a917d859299a8349a3adf0d.png](/home/jimzeus/outputs/AANN/images/media/image335.png){width="4.3180555555555555in"
height="3.2055555555555557in"}

最前面是39\*31\*3（或31\*31\*3）的输入，接着是4层卷积层+3层最大池化层，然后是一个吐出160维特征值的全连接层（被称为DeepID层），最后一层是一个根据类别数量决定的全连接+Softmax层。

注意图中最后一个隐藏层（即生成160维特征值的FC层）同时连接了第三个池化层和第四个卷积层，作者认为这样提高了对多种尺寸特征的感受，并且避免了这个特征FC层因为神经元太少而成为信息传播的瓶颈。

**论文**

Deep Learning Face Representation from Predicting 10,000 Classes

**参考**

[https://blog.csdn.net/a\_1937/article/details/40425397]{.underline}

[https://blog.csdn.net/stdcoutzyx/article/details/42091205]{.underline}

#### DeepID2（201406）

**论文**

Deep Learning Face Representation by Joint Identification-Verification

[[https://arxiv.org/pdf/1406.4773]{.underline}](https://arxiv.org/pdf/1406.4773)

参考:

[[https://blog.csdn.net/hh\_2018/article/details/80540067]{.underline}](https://blog.csdn.net/hh_2018/article/details/80540067)

#### DeepID3

### FaceNet（201503）

FaceNet是谷歌推出的人脸识别算法/结构，这个结构可以用来进行人脸的验证、识别和聚类。

**论文**

FaceNet: A Unified Embedding for Face Recognition and Clustering

[[https://arxiv.org/pdf/1503.03832]{.underline}](https://arxiv.org/pdf/1503.03832)

代码：

[https://github.com/davidsandberg/facenet]{.underline}

**参考**

[https://blog.csdn.net/stdcoutzyx/article/details/46687471]{.underline}

[https://medium.com/intro-to-artificial-intelligence/one-shot-learning-explained-using-facenet-dff5ad52bd38]{.underline}

### MTCNN（201604）

MTCNN（Multi-Task
CNN）是一个比较著名的人脸检测+人脸对齐网络，它主要包括了3个级联的CNN，分别简称P-Net、R-Net和O-Net，MTCNN的整体流程如下：

-   首先将输入图片缩放到不同的大小，以建立一个图像金字塔

-   用一个简单的全卷积网络Proposal-Net（P-Net）来找出候选区域。

-   接着从P-Net中选出的候选区域被送到更复杂一点的Refine-Net（R-Net），R-Net会拒绝掉大量的伪候选区域，

-   最后从R-Net的输出会进入到最复杂的Output-Net（O-Net），O-Net的工作和P-Net差不多，也是从输入滤除掉更多的伪候选区域，并对人脸框进行回归，除此之外还要给出脸部landmark的位置。

训练的时候，每个网络最后都分成三个分支，分别对应：人脸判定（是否是人脸）、人脸框回归（确定位置）、脸部特征（左眼、右眼、鼻子、左嘴角、右嘴角）定位，实际使用时候不用全部使用（应该是关闭P-Net和R-Net的脸部特征定位）。

**论文**

Joint Face Detection and Alignment using Multi-task Cascaded
Convolutional Networks

[[https://arxiv.org/pdf/1604.02878]{.underline}](https://arxiv.org/pdf/1604.02878)

代码：

[[https://github.com/kpzhang93/MTCNN\_face\_detection\_alignment]{.underline}](https://github.com/kpzhang93/MTCNN_face_detection_alignment)

**参考**

[[https://blog.csdn.net/u014380165/article/details/78906898]{.underline}](https://blog.csdn.net/u014380165/article/details/78906898)

### CenterLoss（2016）

参考《[[网络构成 \> 损失函数 \> Center Loss]{.underline}](\l)》

### SphereFace（201704）

**SphereFace**提出了一种基于角边缘（angular
margin）分离的损失函数**A-Softmax
Loss**，用于使得特征值更加discriminative（判别度高），具体参考《[[网络构成
\> 损失函数 \> A-Softmax Loss]{.underline}](\l)》。

**论文**

SphereFace: Deep Hypersphere Embedding for Face Recognition

[[https://arxiv.org/pdf/1704.08063]{.underline}](https://arxiv.org/pdf/1704.08063)

代码：

[[https://github.com/wy1iu/sphereface]{.underline}](https://github.com/wy1iu/sphereface)

### FacePoseNet（201708）

简称**FPN**

### CosFace（201801）

### ArcFace/InsightFace（201801）

### SeqFace（201803）

### MobileFaceNets（201804）

**MobileFaceNet**是一个轻量化的网络，是在**MobileNet
V2**的基础上，针对人脸识别做了些改动。

最主要的改动是：大部分基础CNN最后都有一个**global avg
pooling（全局平均池化层）**，但研究发现在人脸识别任务中，这层的存在反而降低了准确性，MobileFaceNet论文给出了一个解释：因为人脸识别任务中，图片中间部分的重要性高于周围。为此MobileFaceNet将这层全局平均池化层改为全局DW卷积层（Global
Depthwise Conv）

MobileFaceNet结构如下：

![C:\\Users\\AW\\AppData\\Local\\Temp\\WeChat
Files\\ab1258d004023e58642558f83560faa.png](/home/jimzeus/outputs/AANN/images/media/image337.png){width="4.3694444444444445in"
height="2.579861111111111in"}

**论文**

MobileFaceNets: Efficient CNNs for Accurate RealTime Face Verification
on Mobile Devices

[[https://arxiv.org/pdf/1804.07573]{.underline}](https://arxiv.org/pdf/1804.07573)

参考:

[[https://blog.csdn.net/Fire\_Light\_/article/details/80279342]{.underline}](https://blog.csdn.net/Fire_Light_/article/details/80279342)

[[https://www.jianshu.com/p/10fd3e32fc91]{.underline}](https://www.jianshu.com/p/10fd3e32fc91)

人体姿态估计（Pose Estimation）
-------------------------------

又叫人体关键点检测，人体姿态估计是计算机视觉中一个很基础的问题。从名字的角度来看，可以理解为对人体的姿态（关键点，比如头，左手，右脚等）的位置估计。一般我们可以这个问题再具体细分成4个任务：

-   单人姿态估计 (Single-Person Skeleton Estimation)

-   多人姿态估计 (Multi-person Pose Estimation)

-   人体姿态跟踪 （Video Pose Tracking)

-   3D人体姿态估计 （3D Skeleton Estimation)

**单人姿态估计**的输入是已经crop过的人体图像（相当于行人检测的结果），这个方向单独应用并不是很实用，通常在论文中都是**多人姿态估计**的一个子任务。

应用最广泛的是**多人姿态估计，多人姿态估计**算法大致可以分为两种：

-   Top-down：先通过目标检测，从图片中找到人体，抠出来，再送到网络中进行姿态估计。换言之，top-down是将多人姿态估计的问题转化为多个单人姿态估计的问题。

-   Bottom-up：先找出图片中所有关键点，然后对关键点进行分组，得到一个个人。

具体算法分类如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image338.jpeg){width="6.498611111111111in"
height="3.0in"}

**参考**

人体关键点检测（姿态估计）简介+分类汇总

[[https://zhuanlan.zhihu.com/p/102457223]{.underline}](https://zhuanlan.zhihu.com/p/102457223)

人体姿态估计(Human Pose Estimation)经典方法整理

[[https://zhuanlan.zhihu.com/p/104917833]{.underline}](https://zhuanlan.zhihu.com/p/104917833)

人体姿态估计的过去，现在，未来

[[https://zhuanlan.zhihu.com/p/85506259]{.underline}](https://zhuanlan.zhihu.com/p/85506259)

重新思考人体姿态估计 Rethinking Human Pose Estimation

[[https://zhuanlan.zhihu.com/p/72561165]{.underline}](https://zhuanlan.zhihu.com/p/72561165)

### 数据集

MPII

COCO

### CPM (201602)

**论文**

[[https://arxiv.org/pdf/1602.00134.pdf]{.underline}](https://arxiv.org/pdf/1602.00134.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/102468356]{.underline}](https://zhuanlan.zhihu.com/p/102468356)

代码：

[[https://github.com/timctho/convolutional-pose-machines-tensorflow]{.underline}](https://github.com/timctho/convolutional-pose-machines-tensorflow)

### HourGlass (201603)

**论文**

[[https://arxiv.org/pdf/1603.06937.pdf]{.underline}](https://arxiv.org/pdf/1603.06937.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/102470330]{.underline}](https://zhuanlan.zhihu.com/p/102470330)

代码：

[[https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation]{.underline}](https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation)

### OpenPose (201611)

**论文**

[[https://arxiv.org/pdf/1611.08050.pdf]{.underline}](https://arxiv.org/pdf/1611.08050.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/102472863]{.underline}](https://zhuanlan.zhihu.com/p/102472863)

代码：

旧：[[https://github.com/ZheC/Realtime\_Multi-Person\_Pose\_Estimation]{.underline}](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

新：[[https://github.com/CMU-Perceptual-Computing-Lab/openpose]{.underline}](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

### CPN (201711)

**论文**

[[https://arxiv.org/pdf/1711.07319.pdf]{.underline}](https://arxiv.org/pdf/1711.07319.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/102475395]{.underline}](https://zhuanlan.zhihu.com/p/102475395)

代码：

[[https://github.com/chenyilun95/tf-cpn]{.underline}](https://github.com/chenyilun95/tf-cpn)

### MSPN (201901)

**论文**

[[https://arxiv.org/pdf/1901.00148.pdf]{.underline}](https://arxiv.org/pdf/1901.00148.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/102494631]{.underline}](https://zhuanlan.zhihu.com/p/102494631)

代码：

[[https://github.com/megvii-detection/MSPN]{.underline}](https://github.com/megvii-detection/MSPN)

### HRNet (201902)

**论文**

[[https://arxiv.org/pdf/1902.09212.pdf]{.underline}](https://arxiv.org/pdf/1902.09212.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/102494979]{.underline}](https://zhuanlan.zhihu.com/p/102494979)

代码：

[[https://github.com/leoxiaobin/deep-high-resolution-net.pytorch]{.underline}](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

研究方向：视频
==============

视频理解的主要关注点是人的行为，当然也可以应用到非人类出现的场景。大致可以分为以下几个方向：

-   **行为识别**：Action Recognition，又叫Video Action
    Recognition或者Video
    Classification，就是给一段视频分类，差不多对应图像处理中的**图像分类**。解决了What的问题，但是跟**图像分类**一样，在实际应用中不是很多。

-   **时序行为检测**：Temporal Action
    Detection，给定一段视频，要检测其中的行为片段，也就是这段视频里包括几个行为，每个行为的开始和结束的时间点是什么。这个方向比行为识别高级，解决了What和When。

-   **时空行为检测**：Spatio-temporal Action
    Detection，这个相较于时序行为检测又高级了一些，对给定视频，要检测出其中每个行为开始和结束的时间点，以及每一帧中该行为的位置（bounding
    box），解决了What、When和Where的问题。

-   **视频描述**：Video Caption，根据视频生成描述

-   **视频问答**：Video
    QA，这是视频理解的终极王者，给定一段视频和一个问题，NN可以给出回答，比如"视频中的女孩一共跳了几下"

行为识别（Video Action Classification）
---------------------------------------

**视频动作分类**，**Video Action
Classification**，或者**行为识别，Action
Recognition**，或者**视频识别**，**Video
Recognition**，是NN在CV应用中，关于视频的一个最简单的方向，这个应用的目的是在给定一小段视频的情况下，预测其所属的行为类别（类似图像分类）。

在2014年之前，深度学习就应用在视频理解上，但是效果并不好（没有传统方法好，比如iDT）。

直到2014年**双流（two-stream）网络**出来，时空双网络也成为视频理解的一大流派。之后又有TSN等网络走这条路。

视频理解的另外一个方向是**3D网络**，起始于C3D，即加上了时间方向（帧）的3D卷积。

目前的视频理解论文大都基于这两个方向，也有两者的融合。

除了3D网络和双流网络之外，还有两个小点的方向，**LSTM+CNN**和**skeleton-based**（即**人体姿态估计Pose
Estimation**），LSTM的方向很久都没什么paper出来，而skeleton方向缺乏数据集。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image339.jpeg){width="5.808333333333334in"
height="2.1506944444444445in"}

具体的包括：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image340.jpeg){width="6.248611111111111in"
height="3.995138888888889in"}

**参考**

视频理解近期研究进展

[[https://zhuanlan.zhihu.com/p/36330561]{.underline}](https://zhuanlan.zhihu.com/p/36330561)

[[https://zhuanlan.zhihu.com/p/33040925]{.underline}](https://zhuanlan.zhihu.com/p/33040925)

简评 \| Video Action Recognition 的近期进展

[[https://zhuanlan.zhihu.com/p/59915784]{.underline}](https://zhuanlan.zhihu.com/p/59915784)

一文了解通用行为识别ActionRecognition：了解及分类

[[https://zhuanlan.zhihu.com/p/103566134]{.underline}](https://zhuanlan.zhihu.com/p/103566134)

基于3D骨架的深度学习行为识别综述

[[https://zhuanlan.zhihu.com/p/107983551]{.underline}](https://zhuanlan.zhihu.com/p/107983551)

Deep learning for video action recognition review

[[http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review]{.underline}](http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review)

PWC：

[[https://paperswithcode.com/task/action-recognition-in-videos]{.underline}](https://paperswithcode.com/task/action-recognition-in-videos)

### 数据集

**参考**

[[https://zhuanlan.zhihu.com/p/36330561]{.underline}](https://zhuanlan.zhihu.com/p/36330561)

#### UCF101

-   101类、13320个视频剪辑、每个视频不超过10秒、共27小时、分辨率320\*240、共6.5
    GB。

-   视频源于YouTube，内容包含化妆刷牙、爬行、理发、弹奏乐器、体育运动五大类。

-   每类动作由25个人做动作，每人做4-7组。

-   在摄像机运动、物体外观和姿态、物体尺度、视点、杂乱背景、光照条件等方面存在较大的差异

官网：

[[https://www.crcv.ucf.edu/data/UCF101.php]{.underline}](https://www.crcv.ucf.edu/data/UCF101.php)

**论文**

[[https://arxiv.org/pdf/1212.0402.pdf]{.underline}](https://arxiv.org/pdf/1212.0402.pdf)

#### HMDB51

HMDB51包括51类、6766剪辑视频、每个视频不超过10秒、分辨率320\*240、共2
GB。视频源于YouTube和谷歌视频，内容包括人面部、肢体、和物体交互的动作这几大类。

**论文**

[[https://serre-lab.clps.brown.edu/wp-content/uploads/2012/08/Kuehne\_etal\_iccv11.pdf]{.underline}](https://serre-lab.clps.brown.edu/wp-content/uploads/2012/08/Kuehne_etal_iccv11.pdf)

官网：

[[https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/]{.underline}](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)

#### Sports1M

487类、1,100,000视频（70%训练、20%验证、10%测试）。内容包含各种体育运动。

**论文**Large-scale video classification with convolutional neural
networks

#### Kinectics

Kinetics是**谷歌DeepMind**推出的视频数据集，先后推出了Kinetics-400、Kinetics-600和Kinetics-700三个版本，数字表示视频类别的数量。视频源于**YouTube**，Kinetics官网上的csv文件描述了每个视频的**类别、在Youtube的ID、事件开始时间、事件结束时间**等。

可以用下面的官方代码通过csv下载视频：

[[https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics]{.underline}](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)

也有非官方的下载器：

[[https://github.com/Showmax/kinetics-downloader]{.underline}](https://github.com/Showmax/kinetics-downloader)

Kinetics-400目前包括了400类、22万多训练视频、1.8万验证视频、3.5万测试视频。

Kinetics-600目前包括了600类、37万多训练视频、2.8万验证视频、5.7万测试视频。

Kinetics-700目前包括了700类、53万多训练视频、3.4万验证视频、6.8万测试视频。

Kinetics是一个大规模数据集，其在视频理解中的作用有些类似于ImageNet在图像识别中的作用，有些工作用Kinetics预训练模型迁移到其他视频数据集。

官网：

[[https://deepmind.com/research/open-source/kinetics]{.underline}](https://deepmind.com/research/open-source/kinetics)

**论文**

Kinetics-400：[[https://arxiv.org/pdf/1705.06950.pdf]{.underline}](https://arxiv.org/pdf/1705.06950.pdf)

Kinetics-600：[[https://arxiv.org/pdf/1808.01340.pdf]{.underline}](https://arxiv.org/pdf/1808.01340.pdf)

Kinetics-700：[[https://arxiv.org/pdf/1907.06987.pdf]{.underline}](https://arxiv.org/pdf/1907.06987.pdf)

#### Somthing-something V2

Something-something是20BN推出的视频数据集

官网：

[[https://20bn.com/datasets/something-something/v2]{.underline}](https://20bn.com/datasets/something-something/v2)

### 衡量标准

由于跟Image Classification的相似性，衡量标准同图像分类。

### 传统方法

传统方法中最出色的是iDT，迄今为止iDT和神经网络方法的差距都不是很大。

#### iDT

### Two-Stream

#### Two-stream (201406)

two-stream这篇论文的主要贡献有3个：

1、首先，提出了一种融合时空网络的双流convnet体系结构。

2、第二，证明了在多帧密集光流上训练的convnet的方法，即使在训练数据有限的情况下仍能获得很好的性能。

3、最后，将多任务学习应用于两种不同的动作分类数据集，可以增加训练数据量，提高训练数据的性能。

two-stream的概念是这篇文章最大的贡献，在之后的网络中屡次被用到，这里的两个stream指的是两个独立的CNN，分别用于处理空间数据（Spatial
Stream ConvNet）和时间数据（Temporal Stream ConvNet）。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image341.jpeg){width="5.7555555555555555in"
height="2.5555555555555554in"}

上层的Spatio Stream
Convet处理静态的帧画面，输入是单个帧画面，得到画面中物体的特征，还可以使用ImageNet
challenge数据集种有标注的图像数据进行预训练。temporal stream
convet则是通过多帧画面的光流位移来获取画面中物体的运动信息，其输入由多个连续帧之间的光流位移场叠加而成，这种输入显式地描述了视频帧之间的运动，这使得识别更加容易，因为网络不需要隐式地估计运动。两个不同的stream都通过CNN实现，最后进行信息融合。

optical flow是由一些位移矢量场（displacement vector
fields）（每个矢量用dt表示)组成的，其中dt是一个向量，表示第t帧的displacement
vector，是通过第t和第t+1帧图像得到的。dt包含水平部分dtx和竖直部分dty，可以看下图中的（d）和（e）。因此如果一个video有L帧，那么一共可以得到2L个channel的optical
flow，然后才能作为Figure1中temporal stream convnet网络的输入。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image342.jpeg){width="5.727083333333334in"
height="1.9743055555555555in"}

上图中的（a）和（b）表示连续的两帧图像，（c）表示一个optical
flow，（d）和（e）分别表示一个displacement vector
field的水平和竖直两部分。

所以如果假设一个video的宽和高分别是w和h，L表示连续的帧数，那么Figure1中temporal
stream convnet的输入维度应该是下面这样的。其中τ表示任意的一帧。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image343.png){width="1.75in"
height="0.4166666666666667in"}

光流数据作为Temporal ConvNet的输入，有以下几种形式：

1.  Optical flow stacking:
    输入的大小为w×h×2L，L表示堆叠的帧数。可以描述输入map某个位置的光流变化。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image344.jpeg){width="3.4229166666666666in"
height="2.182638888888889in"}

2.  轨迹叠加(Trajectory
    stacking)：在基于轨迹的描述符的启发下，另一种运动表示法将在多个帧的相同位置采样的光流替换为沿运动轨迹采样的流。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image345.jpeg){width="4.209722222222222in"
height="2.292361111111111in"}

3.  Bi-directional optical flow:
    输入大小为w×h×2L，由于是双向的，L/2表示堆叠的帧数。

4.  Mean flow subtracion:
    减掉平均光流，用来去除部分相机运动对光流的影响。

论文的第三个贡献，就是用"multi-task
learning"来克服数据量不足的问题。其实就是CNN的最后一层连到多个softmax的层上，对应不同的数据集，这样就可以在多个数据集上进行multi-task
learning。spatial stream
convnet因为输入是静态的图像，因此其预训练模型容易得到（一般采用在ImageNet数据集上的预训练模型），但是temporal
stream
convnet的预训练模型就需要在视频数据集上训练得到，但是目前能用的视频数据集规模还比较小（主要指的是UCF-101和HMDB-51这两个数据集，训练集数量分别是9.5K和3.7K个video）。因此作者采用multi-task的方式来解决。怎么做呢？首先原来的网络（temporal
stream
convnet）在全连接层后只有一个softmax层，现在要变成两个softmax层，一个用来计算HDMB-51数据集的分类输出，另一个用来计算UCF-101数据集的分类输出，这就是两个task。这两条支路有各自的loss，最后回传loss的时候采用的是两条支路loss的和。

**论文**

[[https://arxiv.org/pdf/1406.2199v2.pdf]{.underline}](https://arxiv.org/pdf/1406.2199v2.pdf)

**代码**

[[https://github.com/woodfrog/ActionRecognition]{.underline}](https://github.com/woodfrog/ActionRecognition)

**参考**

[[https://zhuanlan.zhihu.com/p/61605147]{.underline}](https://zhuanlan.zhihu.com/p/61605147)

[[https://zhuanlan.zhihu.com/p/34929782]{.underline}](https://zhuanlan.zhihu.com/p/34929782)

#### Two-stream Fusion (201604)

更好的双流融合

**论文**

Convolutional Two-Stream Network Fusion for Video Action Recognition

[[https://arxiv.org/pdf/1604.06573.pdf]{.underline}](https://arxiv.org/pdf/1604.06573.pdf)

#### TSN (201608)

由于信息的来源是单帧图像和光流图，two-stream的缺点是对长视频的学习能力不足，已经有的解决方法是dense
temporal
sampling，但缺点是计算量太大。此外CNN训练需要大量的数据，而当时的数据量不够大，容易过拟合。

TSN（Temporal Segment
Network）也是建立在two-stream的基础上的，在TSN中，作者首先将数据等时间的划分成K个segments（原文K=3），然后在每一个segment中随机的采样出一个snippet，这样就生成了K个snippets。每一个snippet都作为网络的input就能得到K个关于视频所属类别的scores，然后这K个scores通过一个融合函数G来生成最终评分。G主要有
max, average和weighted average这三种，作者通过对比实验最终选择average。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image346.jpeg){width="4.951388888888889in"
height="3.009027777777778in"}

也就是说，给定一段视频 V，把它按相等间隔分为 K 段 {S~1~,S~2~,⋯,S~K~}。接着，TSN按如下方式对一系列片段进行建模：

TSN(T~1~,T~2~,⋯,T~K~)=H( G( F(T~1~;W), F(T~2~;W), ⋯, F(T~K~;W) ) )

-   (T~1~,T~2~,⋯,T~K~) 代表片段序列，每个片段 T~k~ 从它对应的段 S~k~
    中随机采样得到。

-   F(T~k~;W)函数代表采用 W作为参数的卷积网络作用于短片段 T~k~，函数返回
    T~k~ 相对于所有类别的得分。

<!-- -->

-   段共识函数 G（The segmental consensus
    function）结合多个短片段的类别得分输出以获得他们之间关于类别假设的共识。（本文G为取平均值）

-   基于这个共识，预测函数 H预测整段视频属于每个行为类别的概率（本文 H
    选择了Softmax函数）。

-   结合标准分类交叉熵损失（cross-entropy
    loss），关于部分共识的最终**损失函数**的形式为：![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image347.png){width="1.49375in"
    height="0.4166666666666667in"}，其中C表示类别数，y~i~是标签。

网络内部采用BN-Inception。

此外，作者还提出了**RGB diff**与**Warped Optical
Flow是**这两种modality，来作为RGB和optical flow的补充。最后作者发现RGB +
optical flow + warped optical flow的组合是最好的，引入RGB
diff反而会带来较差的结果。

-   **RGB
    diff**：**RGB差值**，连续的两个frames相减，能凸显出运动的部分，补充运动信息

-   **Warped Optical
    Flow**：**扭曲光流**，由于现实拍摄的视频中，通常存在摄像机的运动，这样光流场就不是单纯体现出人类行为。由于相机的移动，视频背景中存在大量的水平运动。受到iDT（improved
    dense
    trajectories）工作的启发，作者提出将扭曲的光流场作为额外的输入。通过估计单应性矩阵（homography
    matrix）和补偿相机运动来提取扭曲光流场。图2中，扭曲光流场抑制了背景运动，使得专注于视频中的人物运动

作者还使用了多种减轻过拟合的技术。用RGB模型初始化optical
flow模型、freeze除了第一层以外的其余BN层的均值和方差、采用多种数据增强技术。

**论文**

Temporal Segment Network: Towards Good Practices for Deep Action
Recognition

[[https://arxiv.org/pdf/1608.00859.pdf]{.underline}](https://arxiv.org/pdf/1608.00859.pdf)

**代码**

[[https://github.com/yjxiong/temporal-segment-networks]{.underline}](https://github.com/yjxiong/temporal-segment-networks)

**参考**

[[https://zhuanlan.zhihu.com/p/32777430]{.underline}](https://zhuanlan.zhihu.com/p/32777430)

[[https://zhuanlan.zhihu.com/p/62121305]{.underline}](https://zhuanlan.zhihu.com/p/62121305)

[[https://blog.csdn.net/chen1234520nnn/article/details/104901072]{.underline}](https://blog.csdn.net/chen1234520nnn/article/details/104901072)

#### TRN (201711)

**论文**

[[https://arxiv.org/pdf/1711.08496.pdf]{.underline}](https://arxiv.org/pdf/1711.08496.pdf)

**参考**

代码：

### 3D Conv

#### C3D (201412)

3D卷积是在图像的二维卷积（长、宽）上加一维时间方向的维度。

C3D中C指的是Convolution，C3D网络中的视频片段尺寸定义为c\*l\*h\*w（c为通道、l为帧数、h和w为高和宽），下图为C3D的结构：

![c3d](/home/jimzeus/outputs/AANN/images/media/image348.png){width="5.767361111111111in"
height="1.0083333333333333in"}

C3D网络有8个卷积层，5个最大池化层和2个全连接层，最后是softmax输出层。所有的3D卷积核都是3×3×3，在空间和时间上都有步长1。卷积核的数量表示在每个框中。3D池化层由pool1到pool5表示。所有池化核为2×2×2，除了pool1为1×2×2。每个全连接层有4096个输出单元。

C3D的一个贡献就是发现3\*3\*3的卷积核效果最好。

**论文**

[[https://arxiv.org/pdf/1412.0767.pdf]{.underline}](https://arxiv.org/pdf/1412.0767.pdf)

**参考**

[[https://www.jianshu.com/p/0b4964261673]{.underline}](https://www.jianshu.com/p/0b4964261673)

[[https://www.jianshu.com/p/09d1d8ffe8a4]{.underline}](https://www.jianshu.com/p/09d1d8ffe8a4)

#### I3D (201705)

I3D中的I表示Inflated，是一个双流网络和3D卷积的融合。

论文提到，当前3D网络的主要问题是：

-   相对于2D网络，参数量有较大增加，更难训练

-   因为是全新的网络，无法利用已经成熟的预训练2D网络，只能用层数较少的CNN在小数据集上从头训练

I3D把two-stream结构中的2D卷积扩展为3D卷积。并且利用了预训练的模型Inception，其方法是将N\*N的Inception
module复制N遍（即Inflate，膨胀），变为N\*N\*N的module。

由于时间维度不能缩减过快，前两个汇合层的卷积核大小是1×2×2，最后的汇合层的卷积核大小是2\*7\*7。和之前文章不同的是，two-tream的两个分支是单独训练的，测试时融合它们的预测结果。

**论文**

[[https://arxiv.org/pdf/1705.07750v3.pdf]{.underline}](https://arxiv.org/pdf/1705.07750v3.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/34919655]{.underline}](https://zhuanlan.zhihu.com/p/34919655)

**代码**

[[https://github.com/deepmind/kinetics-i3d]{.underline}](https://github.com/deepmind/kinetics-i3d)

#### T3D (201711)

T3D中的T表示Temporal，来自于论文中的**TTL层（Temporal Transition
Layer）**。

T3D的三维卷积主要是在densenet的基础上改进得到，具体来说就是将原始网络中二维卷积修改为三维卷积，二维pooling修改为三维pooling。

T3D最主要的创新来源于**TTL层（Temporal Transition
Layer）**，TTL层包含几个不同大小和时间域深度的卷积kernel和三维pooling层构成，所希望达到的效果是能够对短、中、长三个不同的时间长度的序列信息进行建模。

![2020-07-17
17-07-50屏幕截图](/home/jimzeus/outputs/AANN/images/media/image349.png){width="5.7659722222222225in"
height="2.935416666666667in"}

**论文**

[[https://arxiv.org/pdf/1711.08200.pdf]{.underline}](https://arxiv.org/pdf/1711.08200.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/90374193]{.underline}](https://zhuanlan.zhihu.com/p/90374193)

**代码**

[[https://github.com/MohsenFayyaz89/T3D]{.underline}](https://github.com/MohsenFayyaz89/T3D)

#### P3D (201711)

P3D中的P表示Pseudo，P3D用一个1×3×3的空间方向卷积和一个3×1×1的时间方向卷积来近似原3×3×3卷积。这种技巧用于减少运算量，类似于2D卷积中的**深度可分离卷积（Depthwise
Separable Conv）**和**SS卷积（Spatial Separable Conv）**。

通过组合三种不同的模块结构，进而得到P3D ResNet。P3D
ResNet在参数数量、运行速度等方面对C3D作出了优化。

![2020-07-17
16-09-12屏幕截图](/home/jimzeus/outputs/AANN/images/media/image350.png){width="5.757638888888889in"
height="2.3652777777777776in"}

**论文**

[[https://arxiv.org/pdf/1711.10305.pdf]{.underline}](https://arxiv.org/pdf/1711.10305.pdf)

#### R(2+1)D (201711)

R(2+1)D中的R表示Resnet。

R(2+1)D的创新包括2个：

-   新的混合型结构：在浅层使用3维卷积，在深层接上2维卷积。

-   新的微结构：**2+1维卷积块**，就是把3维卷积操作分解成两个接连进行的子卷积块，2维空间卷积和1维时间卷积（类似P3D-A）。

**论文**

[[https://arxiv.org/pdf/1711.11248.pdf]{.underline}](https://arxiv.org/pdf/1711.10305.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/48267318]{.underline}](https://zhuanlan.zhihu.com/p/48267318)

#### ECO (201804)

**论文**

[[https://arxiv.org/pdf/1804.09066.pdf]{.underline}](https://arxiv.org/pdf/1804.09066.pdf)

#### SlowFast (201812)

SlowFast创造了一种新的时空交互的方法：

1.  两路3d卷积，分别侧重时间（fast）和空间（slow）

2.  将侧重时间的支路信息融合入空间支路

两个支路slow和fast指的是T维的卷积核大小核跨步不同。

-   Fast路的时间维度卷积核大小为aT,时间维度stride为s/a,通道数为c/a，比较轻量;

-   Slow支路时间维度卷积核大小为T，时间维度stride为s,通道数为c。a=8.配置如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image351.jpeg){width="5.372916666666667in"
height="6.532638888888889in"}

网络结构就是以上两种3d卷积的支路，中间将slow支路融合到fast，有三种方法：

1.  time2channel：reshape fast维度的\[bC, aT, S, S\] to \[abC, T, S, S\]

2.  对fast路的特征图进行temporal stride sample 统一时间维度

3.  fast接跨步3d卷积统一时间维度

**论文**

[[https://arxiv.org/pdf/1812.03982v3.pdf]{.underline}](https://arxiv.org/pdf/1812.03982v3.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/103577209]{.underline}](https://zhuanlan.zhihu.com/p/103577209)

代码：

[[https://github.com/facebookresearch/SlowFast]{.underline}](https://github.com/facebookresearch/SlowFast)

### Skeleton-based

Skeleton-based类型的神经网络需要先对视频做**人体关键点检测**（Keypoint
Detection，或称**人体姿态评估**Pose
Estimation），然后将得到的人体关键点作为输入给到神经网络，得出视频分类。这类神经网络的论文不多，或许与数据集太少有关。

#### ST-GCN (201801)

**论文**

[[https://arxiv.org/pdf/1801.07455.pdf]{.underline}](https://arxiv.org/pdf/1801.07455.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/108255643]{.underline}](https://zhuanlan.zhihu.com/p/108255643)

代码：

[[https://github.com/yysijie/st-gcn]{.underline}](https://github.com/yysijie/st-gcn)

#### TSM (201811)

**论文**

[[https://arxiv.org/pdf/1811.08383.pdf]{.underline}](https://arxiv.org/pdf/1811.08383.pdf)

**代码**

[[https://github.com/MIT-HAN-LAB/temporal-shift-module]{.underline}](https://github.com/MIT-HAN-LAB/temporal-shift-module)

#### Multigrid (201912)

**论文**

[[https://arxiv.org/pdf/1912.00998v2.pdf]{.underline}](https://arxiv.org/pdf/1912.00998v2.pdf)

**代码**（其中使用了multigrid方法）

[[https://github.com/facebookresearch/SlowFast]{.underline}](https://github.com/facebookresearch/SlowFast)

### LSTM-based

#### LRCN (201411)

**论文**

[[https://arxiv.org/pdf/1411.4389.pdf]{.underline}](https://arxiv.org/pdf/1411.4389.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/90378536]{.underline}](https://zhuanlan.zhihu.com/p/90378536)

时序行为识别（Temporal Action Recognition）
-------------------------------------------

**时序行为识别**，Temporal Action
Recognition或者**时域动作检测**，Temporal Action Detection，Temporal
Action Localization。

**时序行为识别**的目的是给定一段视频，算法需要检测视频中的行为片段，包括其开始时间、结束时间以及类别，一段视频中可能包括一个或者若干个片段。

**Video Action Recognition**和**Temporal Action
Recognition**之间的关系就像**Image Classification** 和 **Object
Detection**。Temporal Action
Detection不仅要识别动作的类别，还需要知道动作的起始帧和结束帧。

**参考**

[[https://zhuanlan.zhihu.com/p/26603387]{.underline}](https://zhuanlan.zhihu.com/p/26603387)

[[https://zhuanlan.zhihu.com/p/52524590]{.underline}](https://zhuanlan.zhihu.com/p/52524590)

[[https://www.jianshu.com/p/92bb5cad2319]{.underline}](https://www.jianshu.com/p/92bb5cad2319)

### 数据集

#### THUMOS14

该数据集即为THUMOS Challenge
2014。该数据集包括**行为识别**和**时序行为检测**两个任务。它的训练集为UCF101数据集，包括101类动作，共计13320段分割好的视频片段。THUMOS2014的验证集和测试集则分别包括1010和1574个未分割过的视频。在时序行为检测任务中，只有20类动作的未分割视频是有时序行为片段标注的，包括200个验证集视频（包含3007个行为片段）和213个测试集视频（包含3358个行为片段）。这些经过标注的未分割视频可以被用于训练和测试时序行为检测模型。实际上之后还有THUMOS
Challenge
2015,包括更多的动作类别和视频数，但由于上面可以比较的方法不是很多，所以目前看到的文章基本上还是在THUMOS14上进行实验

官网：

[http://crcv.ucf.edu/THUMOS14/]{.underline}

#### ActivityNet

目前最大的数据库，同样包含分类和检测两个任务 ，这个数据集仅提供视频的youtube链接，而不能直接下载视频，所以还需要用python中的youtube下载工具来自动下载。该数据集包含200个动作类别，20000（训练+验证+测试集）左右的视频，视频时长共计约700小时。

### 衡量标准

由于跟Object Detection的相似性，时序行为识别的衡量标准也是IoU和mAP。

2017的大多数工作在IOU=0.5的情况下达到了20%-30%的MAP，虽然较2016年提升了10%左右，但是在IOU=0.7时直接降低到了10%以下，2018年IOU=0.5有34%的mAP。

### SSN

时空行为识别
------------

**时空行为识别（Spatio-Temporal
Detection）**比**时序行为识别**更进一步，不仅需要在视频中检测出每个动作的**起始帧和结束帧**，还需要检测出这个动作在每一帧中的**位置**。

### 数据集

#### AVA

AVA（Atomic Vision
Action）是**谷歌**推出的数据集，类似Kinetics，其具体视频也是在Youtube上，官网上下载的只是Annotation。

官网：

[[https://research.google.com/ava/]{.underline}](https://research.google.com/ava/)

视频字幕（Video Captioning）
----------------------------

### 数据集

YouCook2

视频问答（Video QA）
--------------------

如果**行为识别**是视频理解的入门，那么**视频问答**就是视频理解的终极难度。任务是输入一段视频和一个文本问题，得到一个答案。

举个例子，输入一段几个小孩拍球的视频，问：蓝色衣服的小朋友拍了几次球？

### Zhou (201804)

**论文**

[[https://arxiv.org/pdf/1804.00819v1.pdf]{.underline}](https://arxiv.org/pdf/1804.00819v1.pdf)

**代码**

[[https://github.com/salesforce/densecap]{.underline}](https://github.com/salesforce/densecap)

### Video-BERT (201904)

**论文**

[[https://arxiv.org/pdf/1904.01766.pdf]{.underline}](https://arxiv.org/pdf/1904.01766.pdf)

视频目标跟踪（Video Object Tracking）
-------------------------------------

广义的**VOT（Video Object Tracking）**任务分为**单目标跟踪（SOT，Single
Object Tracking）**和**多目标跟踪**（**MOT，Multiple Object
Tracking**），而狭义的VOT指的是单目标跟踪（SOT）。

目标跟踪的意思，就是在一段视频上对移动物体做跨时间的跟踪。那么为什么不直接针对每帧图像做Object
Detection呢？因为用目标检测的方法会有一些问题：

-   如果图像中有若干个物体，那如何确定上下两帧之间物体的对应关系？

-   如果跟踪的物体离开画面若干帧（比如被柱子挡住），之后另一个物体出现在画面中，如何确定这两个物体是否是同一个？

或者说，如果仅仅针对每帧做图像处理，我们无法得知物体的移动情况。

此外还会有一些目标检测中多余的计算，也许在VOT中并不需要：

-   直接对视频中每帧图片做目标检测，其计算量代价是比较大的，可以通过一些算法节省算力，比如根据物体在t时刻和t+1时刻的位置，预测其在t+2时刻的位置

-   基本的VOT算法是不需要做物品分类的，分类只是个可选项而已。

再者就是考虑到视频前后帧之间的联系，可以

-   有效的去除错误的目标检测结果

-   补上遗漏的目标检测结果

目前整个目标跟踪领域的发展都比较停滞，依然有很多都是非深度学习的方法，远落后于目标检测（表现为有些目标检测的算法基础上修改为目标跟踪算法，效果都好于传统目标跟踪算法）。深度学习在目标跟踪领域尚未和相关滤波类的算法拉开什么差距。

**论文**

Deep Learning for Visual Tracking: A Comprehensive Survey

[[https://arxiv.org/pdf/1912.00535.pdf]{.underline}](https://arxiv.org/pdf/1912.00535.pdf)

A Review of Visual Trackers and Analysis of its Application to Mobile
Robot

[[https://arxiv.org/pdf/1910.09761.pdf]{.underline}](https://arxiv.org/pdf/1910.09761.pdf)

参考

[[https://cv-tricks.com/object-tracking/quick-guide-mdnet-goturn-rolo/]{.underline}](https://cv-tricks.com/object-tracking/quick-guide-mdnet-goturn-rolo/)

[[https://www.zhihu.com/question/26493945/answer/156025576]{.underline}](https://www.zhihu.com/question/26493945/answer/156025576)

[[https://zhuanlan.zhihu.com/p/44265232]{.underline}](https://zhuanlan.zhihu.com/p/44265232)

### 衡量标准

**参考**

[[https://www.jianshu.com/p/8cd0bcc9792c]{.underline}](https://www.jianshu.com/p/8cd0bcc9792c)

#### Precision Slot / Success Plot

-   **Precision
    Plot**：预测位置中心点与标注的中心位置间的欧式距离，以像素为单位。\
    结果用average precision plot来表示，即为该视频序列所有帧的平均误差。

<!-- -->

-   **Success
    plot**：主要指的是预测目标所在benchmark的重合程度，即IOU（交并比）

#### Accuracy / Robustness / EAO

-   **Accuracy**：**准确率**，是指跟踪器在单个测试序列下的平均重叠率（两矩形框的相交部分面积除以两矩形框的相并部分的面积。即average
    success plot。

-   **Robustness**：**鲁棒性**，是指单个测试序列下的跟踪器失败次数，当重叠率为0时即可判定为失败。

-   **EAO（Expected Average
    Overlap**）：**平均重叠期望**，对每个跟踪器在一个短时图像序列上的非重置重叠的期望值，即在一个N帧视频上每帧的IOU的均值。

> ![](/home/jimzeus/outputs/AANN/images/media/image352.png){width="1.8090277777777777in"
> height="0.8548611111111111in"}

这三者也是VOT2017所使用的评价标准。

### 数据集

#### VOT

**Visual-Object-Tracking Challenge (VOT)**
是当前国际上在线目标跟踪领域最权威的测评平台，由伯明翰大学、卢布尔雅那大学、布拉格捷克技术大学、奥地利科技学院联合创办，旨在评测在复杂场景下单目标跟踪的算法性能。

官网：

[[http://www.votchallenge.net/]{.underline}](http://www.votchallenge.net/)

VOT Chanllenges

[[https://votchallenge.net/challenges.html]{.underline}](https://votchallenge.net/challenges.html)

#### OTB

分为OTB50（OTB-2013）和OTB100（OTB-2015），

多目标跟踪（Multiple Object Tracking）
--------------------------------------

广义的VOT任务分为**单目标跟踪（SOT，Single Object
Tracking）**和**多目标跟踪**（**MOT，Multiple Object
Tracking**），而通常VOT指的是单目标跟踪。

MOT通常的工作流程如下：

1.  给定视频的原始帧

2.  运行对象检测器以获得对象的边界框（目标检测）

3.  对于每个检测到的物体，计算出不同的特征，通常是视觉和运动特征；

4.  之后，相似度计算步骤计算两个对象属于同一目标的概率；

5.  最后，关联步骤为每个对象分配数字ID。

MOT和SOT之间的区别还是比较大的。

按照初始化方法，MOT可以分为Detection-Based
Tracking（DBT）和Detection-Free Tracking（DFT）两种：

-   DBT：首先检测目标，然后链接到轨迹中，给定一个序列，在每帧中对特定类型的目标检测，然后进行跟踪。可自动发现新目标，自动终止消失的目标。现行的MOT通常指的是这种方式

-   DFT：需要在第一帧手动初始化一定数量的目标，然后在后续帧定位这些物体。

按照处理模式分，分为Online追踪和Offline追踪：

-   Online用的是当前帧及之前的信息

-   Offline可以用到未来帧的信息

**参考**

[[https://zhuanlan.zhihu.com/p/97449724]{.underline}](https://zhuanlan.zhihu.com/p/97449724)

[[https://blog.csdn.net/yuhq3/article/details/78742658]{.underline}](https://blog.csdn.net/yuhq3/article/details/78742658)

[[https://zhuanlan.zhihu.com/c\_1102212337087401984]{.underline}](https://zhuanlan.zhihu.com/c_1102212337087401984)

**论文**

Multiple Object Tracking: A Literature Review

[[https://arxiv.org/pdf/1409.7618.pdf]{.underline}](https://arxiv.org/pdf/1409.7618.pdf)

Deep Learning in Video Multi-Object Tracking: A Survey

[[https://arxiv.org/pdf/1907.12740.pdf]{.underline}](https://arxiv.org/pdf/1907.12740.pdf)

### 衡量标准

**参考**

[[https://zhuanlan.zhihu.com/p/75776828]{.underline}](https://zhuanlan.zhihu.com/p/75776828)

#### MOTA/MOTP

MOTA（**MOT Accuracy**）和MOTP（**MOT Precision**）都是MOT
Challenge数据集提出的MOT衡量标准Clear MOT的一部分。

**MOTA**：多目标跟踪的准确度，体现在确定目标的个数，以及有关目标的相关属性方面的准确度，用于统计在跟踪中的误差积累情况，包括FP、FN、ID
Sw。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image353.png){width="2.1333333333333333in"
height="0.33541666666666664in"}

**m~t~**：是FP,缺失数（漏检数），即在第t帧中该目标 ![IMG\_257](/home/jimzeus/outputs/AANN/images/media/image354.png){width="2.0833333333333332e-2in"
height="2.0833333333333332e-2in"} 没有假设位置与其匹配。

**fp~t~**：是FN，误判数，即在第t帧中给出的假设位置 ![IMG\_259](/home/jimzeus/outputs/AANN/images/media/image355.png){width="2.0833333333333332e-2in"
height="2.0833333333333332e-2in"} 没有跟踪目标与其匹配。

**mme~t~**：是ID
Sw，误配数，即在第t帧中跟踪目标发生ID切换的次数，多发生在这档情况下。

**MOTP**：多目标跟踪的精确度，体现在确定目标位置上的精确度，用于衡量目标位置确定的精确程度。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image356.png){width="1.1756944444444444in"
height="0.41875in"}

c~t~：表示第t帧目标o~i~和假设h~j~的匹配个数

d^i^~t~：表示第t帧目标o~i~与其配对假设位置之间的距离，即匹配误差。

**参考**

[[https://zhuanlan.zhihu.com/p/75776828]{.underline}](https://zhuanlan.zhihu.com/p/75776828)

### 数据集

#### MOT Challenge

MOT Challenge是当前用的最多的多目标追踪数据集

官网

[[https://motchallenge.net/]{.underline}](https://motchallenge.net/)

**参考**

[[https://zhuanlan.zhihu.com/p/133670271]{.underline}](https://zhuanlan.zhihu.com/p/133670271)

#### KITTI

用的较少

### 传统算法

当前的MOT（及VOT）尚未有端到端的神经网络的实现（其原因应该是计算量过大、数据集不够），通常都要加上一些其他的算法。

#### 卡尔曼滤波（Kalman Filter）

**卡尔曼滤波（Kalman
filter）**是一种高效率的递归滤波器（自回归滤波器），它能够从一系列的不完全及包含噪声的测量中，估计动态系统的状态。卡尔曼滤波会根据各测量量在不同时间下的值，考虑各时间下的联合分布，再产生对未知变数的估计，因此会比只以单一测量量为基础的估计方式要准。

简单来说，卡尔曼滤波解决的是如何**从多个不确定数据中提取相对精确的数据**。前提是：

-   实践前提是这些数据满足高斯分布。

-   理论前提是一个高斯斑乘以另一个高斯斑可以得到第三个高斯斑，第三个高斯斑即为提取到相对精确的数据范围。

**参考**

[[https://www.zhihu.com/question/23971601/answer/375355599]{.underline}](https://www.zhihu.com/question/23971601/answer/375355599)

[[https://www.zhihu.com/question/23971601/answer/194464093]{.underline}](https://www.zhihu.com/question/23971601/answer/194464093)

[[https://zh.wikipedia.org/wiki/%E5%8D%A1%E5%B0%94%E6%9B%BC%E6%BB%A4%E6%B3%A2]{.underline}](https://zh.wikipedia.org/wiki/%E5%8D%A1%E5%B0%94%E6%9B%BC%E6%BB%A4%E6%B3%A2)

#### 匈牙利算法 & KM算法

**匈牙利算法（Hungarian
Algorithm）和KM算法（Kuhn--Munkres算法，Munkres分配算法）**是都是求解二分图的最大匹配问题的组合优化算法。

二分图就是能分成两组，U,V。其中，U上的点不能相互连通，只能连去V中的点，同理，V中的点不能相互连通，只能连去U中的点。这样，就叫做二分图。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image357.jpeg){width="2.51875in"
height="2.51875in"}

可以把二分图理解为视频中连续两帧中的所有检测框，第一帧所有检测框的集合称为U，第二帧所有检测框的集合称为V。同一帧的不同检测框不会为同一个目标，所以不需要互相关联，相邻两帧的检测框需要相互联通，最终将相邻两帧的检测框尽量完美地两两匹配起来。而求解这个问题的最优解就要用到**匈牙利算法**或者**KM算法**（KM算法解决的是带权二分图的最优匹配问题）。

**参考**

[[https://zhuanlan.zhihu.com/p/62981901]{.underline}](https://zhuanlan.zhihu.com/p/62981901)

[[https://zh.wikipedia.org/wiki/%E5%8C%88%E7%89%99%E5%88%A9%E7%AE%97%E6%B3%95]{.underline}](https://zh.wikipedia.org/wiki/%E5%8C%88%E7%89%99%E5%88%A9%E7%AE%97%E6%B3%95)

### SORT (201602)

现在多目标跟踪算法的效果，与目标检测的结果息息相关，因为主流的多目标跟踪算法都是TBD（Tracking-by-Detecton）策略。

SORT采用的是在线跟踪的方式，不使用未来帧的信息。在保持100fps以上的帧率的同时，也获得了较高的MOTA（在当时16年的结果中）。

SORT的贡献主要有三：

-   利用强大的CNN检测器的检测结果来进行多目标检测

-   使用基于**卡尔曼滤波**与**匈牙利算法**的方法来进行跟踪

-   开源了代码，为MOT领域提供一个新的baseline

**论文**

Simple Online and Real-time Tracking

[[https://arxiv.org/pdf/1602.00763.pdf]{.underline}](https://arxiv.org/pdf/1602.00763.pdf)

**代码**

[[https://github.com/abewley/sort]{.underline}](https://github.com/abewley/sort)

**参考**

[[https://zhuanlan.zhihu.com/p/62858357]{.underline}](https://zhuanlan.zhihu.com/p/62858357)

[[https://zhuanlan.zhihu.com/p/59148865]{.underline}](https://zhuanlan.zhihu.com/p/59148865)

### DeepSORT (201703)

DeepSORT是在SORT基础上做的修改，增加了Deep Association Metric。

**论文**

Simple Online and Real-time Tracking with a Deep Association Metric

[[https://arxiv.org/pdf/1703.07402.pdf]{.underline}](https://arxiv.org/pdf/1703.07402.pdf)

**代码**

[[https://github.com/nwojke/deep\_sort]{.underline}](https://github.com/nwojke/deep_sort)

### MOTDT (201809)

**论文**

[[https://arxiv.org/pdf/1809.04427.pdf]{.underline}](https://arxiv.org/pdf/1809.04427.pdf)

### JDE(201909)

**论文**

[[https://arxiv.org/pdf/1909.12605.pdf]{.underline}](https://arxiv.org/pdf/1909.12605.pdf)

**代码**

[[https://github.com/Zhongdao/Towards-Realtime-MOT]{.underline}](https://github.com/Zhongdao/Towards-Realtime-MOT)

### FairMOT (202004)

**论文**

[[https://arxiv.org/pdf/2004.01888v3.pdf]{.underline}](https://arxiv.org/pdf/2004.01888v3.pdf)

**代码**

[[https://github.com/ifzhang/FairMOT]{.underline}](https://github.com/ifzhang/FairMOT)

行人识别（Person Recognition）
------------------------------

类似Face Recognition，Person
Recognition泛指跟行人（Person）相关的一些研究方向（这些研究方向会有交叉），比如：

-   **行人检索：Person
    Retrieval**，总结图像/视频中的行人的特征，比如头发长短、性别、衣服颜色，是否带眼镜等等，以便在图像/视频中检索出具有特定特征的行人。

-   **行人检测：Pedestrian Detection (Person
    Detection)**，从图像找出行人，是Object Detection的特例。

-   **行人识别：Person
    Identification**，类似人脸识别，根据行人图像，从行人特征值数据库找出该图像属于谁。

-   **行人再识别：Person Re-Identification，Person
    Re-ID**，指的是在一组监控摄像头中交叉跟踪一个人（从某个摄像头的范围中走出，之后出现在另一个摄像头范围中），由于不同摄像头的角度、光照、色差、分辨率等等的区别，使得神经网络需要去消除这些区别，**Re-ID**作为一个较为通用的技术，在其他领域，比如视频目标跟踪（VOT）也有用到。

-   **行人搜索：Person
    Search**，与行人再识别的区别在于，需要从完整的图像/视频（即监控画面）中找出需要找的人，而re-ID的任务是根据目标人，从（裁剪出来的）人形图像库中搜索。也就是说多了个**Person
    Detection**的任务。

### 数据集

SoftBioSearch

Market1501

CUHK03

DPM

### MLCNN（2015ICB）

**Multi-Label
CNN**将人体分为15个部分，从上到下5行，左中右3列。每个部分之间有重叠（overlapping）。然后将这15个部分接入一个多标签的卷积神经网络（Multi-Label
CNN），每个部分单独卷积。每个标签会和相对应的部分连接（比如头发和头部，衣服和身体部分，短裤和腿部），如下：

![](/home/jimzeus/outputs/AANN/images/media/image358.png){width="3.8847222222222224in"
height="3.435416666666667in"}

每个标签所对应的部分都是预定义好的：

![](/home/jimzeus/outputs/AANN/images/media/image359.png){width="4.166666666666667in"
height="1.7458333333333333in"}

两张行人图像之间的距离（Fusion Distance）由属性距离（Attribute-based
Distance）和底层距离（Low-level based Distance）共同构成：

![](/home/jimzeus/outputs/AANN/images/media/image360.png){width="4.684722222222222in"
height="1.7326388888888888in"}

论文:

Multi-label CNN Based Pedestrian Attribute Learning for Soft Biometrics

### OIM损失函数（201604）

《Joint Detection of and Identification Feature Learning for Person
Search》的论文作者认为自己的贡献在3个方面：

1.  提出了一个新的神经网络

2.  提出了一个新的损失函数OIM（Online Instance Matching）

3.  创建标注了一个新的数据集

以下为该论文提出的神经网络：

![2019-12-02
14-29-08屏幕截图](/home/jimzeus/outputs/AANN/images/media/image362.png){width="5.747916666666667in"
height="2.2270833333333333in"}

-   使用ResNet-50作为基础网络，首先通过Stem
    CNN（ResNet-50的前半部分）提取出特征图。

-   基于Anchor Box，Pedestrain Proposal
    Net负责从特征值中找到并定位行人。

-   接着用RoI Pooling池化特征图

-   RoI Pooling出来的结果通过Identification
    Net（ResNet-50的后半部分）提取出一个2048维的特征值

-   这个特征值一方面被送去过滤可能的假阳性结果（图中右下），另一方面被L2-Normalize为一个256维的id
    feat，OIM使用这个id feat来计算和目标行人的余弦距离。

论文提出了一个LUT（Lookup Table）用于保存Labeled Identities
和一个CQ（Circular Queue）用于保存Unlabeled Identities：

![2019-12-02
17-03-11屏幕截图](/home/jimzeus/outputs/AANN/images/media/image363.png){width="5.747916666666667in"
height="2.890972222222222in"}

**论文**

Joint Detection and Identification Feature Learning for Person Search

[[https://arxiv.org/pdf/1604.01850]{.underline}](https://arxiv.org/pdf/1604.01850)

代码：

[[https://github.com/ShuangLI59/person\_search]{.underline}](https://github.com/ShuangLI59/person_search)

### PCB和RPP（201711）

《Beyond Part Models：Person Retrieval with Refined Part Pooling (and A
Strong Convolutional Baseline)》论文有两个贡献：

-   提出了一个新的神经网络PCB（Part-based Convolutional Baseline）

-   提出了一个新的分块（Part）方法RPP（Refined Part Pooling）

通常现行的Part方法分成两种：

-   利用外部方法进行分块，比如使用Human Pose Estimation网络

-   不利用外部方法，比如PAR和本文

PCB的基础网络可以是任何主流分类网络，比如Inception或ResNet，本文使用了ResNet50作为基础网络。

**论文**

Beyond Part Models: Person Retrieval with Refined Part Pooling

(and A Strong Convolutional Baseline)

[[https://arxiv.org/pdf/1711.09349]{.underline}](https://arxiv.org/pdf/1711.09349)

代码：

[[https://github.com/layumi/Person\_reID\_baseline\_pytorch]{.underline}](https://github.com/layumi/Person_reID_baseline_pytorch)

### Height, Color, Gender（201810）

《**Person Retrieval in Surveillance Video using Height, Color and
Gender**》这篇论文提出了一个若干步骤的方法，通过身高、颜色和性别进行行人检索：

1.  首先通过**Mask R-CNN**得到视频/图像中的行人的语义分割（像素级别）

2.  将所有得到的行人（语义分割）输入Height
    Filter，根据相机的参数（安装位置、角度、焦距等），通过矩阵计算（Tsai
    Camera
    Calibration）得到行人的身高（取视频中各帧的平均值），对比输入的身高条件，过滤掉一部分行人。

3.  将剩下的行人输入Color Filter，根据Height
    Filter计算出的身高，切取从上到下20%到50%的部分作为上身，50%到100%作为腿，将这两部分人体切片送入AlexNet为基础的颜色分类器得到颜色，与输入的颜色条件，得到匹配的结果

4.  如果需要，将上一步得到的行人结果（全身的语义分割）送入AlexNet为基础的性别分类器得到性别，与输入的性别条件比对，得到结果。

![](/home/jimzeus/outputs/AANN/images/media/image364.png){width="4.733333333333333in"
height="2.5840277777777776in"}

**论文**

Person Retrieval in Surveillance Video using Height, Color and Gender

[[https://arxiv.org/pdf/1810.05080]{.underline}](https://arxiv.org/pdf/1810.05080)

### st-ReID（201812）

**论文**

Spatial-Temporal Person Re-identification

[[https://arxiv.org/pdf/1812.03282]{.underline}](https://arxiv.org/pdf/1812.03282)

代码：

[[https://github.com/Wanggcong/Spatial-Temporal-Re-identification]{.underline}](https://github.com/Wanggcong/Spatial-Temporal-Re-identification)

### DG-net（201904）

**论文**

Joint Discriminative and Generative Learning for Person
Re-identification

[[https://arxiv.org/pdf/1904.07223]{.underline}](https://arxiv.org/pdf/1904.07223)

代码：

[[https://github.com/NVlabs/DG-Net]{.underline}](https://github.com/NVlabs/DG-Net)

研究方向：自然语言处理（NLP）
=============================

**Natural Language
Processing，自然语言处理**，这个范畴本身也是超越深度学习/神经网络的。神经网络兴起的早期，RNN是NLP的一个非常重要的研究方向，Transformer则后来居上。

早期的NLP使用One-hot或者词袋模型来表示词，之后Word2Vec的出现使词嵌入成为表示词汇的主流方式，再之后预训练模型（GPT系列、BERT系列）的出现使得Transformer成为主流。

**论文**

中文NLP模型综述

[[https://arxiv.org/pdf/2004.13922.pdf]{.underline}](https://arxiv.org/pdf/2004.13922.pdf)

**参考**

NLP的巨人肩膀

[[https://zhuanlan.zhihu.com/p/50443871]{.underline}](https://zhuanlan.zhihu.com/p/50443871)

自然语言处理中注意力机制综述

[[https://zhuanlan.zhihu.com/p/54491016?utm\_source=qq&utm\_medium=social&utm\_oi=616755169208307712]{.underline}](https://zhuanlan.zhihu.com/p/54491016?utm_source=qq&utm_medium=social&utm_oi=616755169208307712)

[[https://blog.csdn.net/jiaowoshouzi/article/details/89073944]{.underline}](https://blog.csdn.net/jiaowoshouzi/article/details/89073944)

概念
----

### 语言学相关

这里是语言学相关的一些概念，对理解NLP任务有帮助。

-   **Phonology（音位）**：语言学分支，研究语音的功能

-   **Orthography（正写）**：书写一门语言的惯例，比如英语中的拼写、大小写、标点等

-   **Morphology（词法）**：语言学的分支，研究单词的内部结构和形成方式

-   **Morpheme（语素）**：指最小的语法单位，是最小的语音语义结合体，英语中比如unbreakable这个词可以分为un-、break、-able三个语素。

-   **Syntax（句法）**：即句子的格式，主谓宾这样的结构是否正确，句法正确的句子不一定有意义。

-   **Semantics（语义）**：简言之即语言（或者逻辑等）的意义

-   **Pragmatics（语用）**：语用研究的是不同语境对语言含义的影响，但其和语义间的区别变得越来越模糊

-   **Grammar（语法）**：也叫句法，是个范围比较大的概念，包括Phonology、Morpholog、Syntax，也包括Phonetics、Semantics、Pragmatics。

-   **Part of Speech（POS）**：词性，比如动词、名词、形容词。

**论文**

[[http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf]{.underline}](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

### Tokenize（分词）

Tokenize即分词，这是NLP的第一步，将文本分割为模型可识别的单位token（通常是词，也有的时候是子词），因此tokenizer（分词器）和模型是对应的。其中还有一个隐含的步骤是将切割后的词转换为对应的id，最终转换为模型可识别的encodings

### TF-IDF

-   **TF：Term
    Frequency（词频）**，词的量化表示的一种方式。某词在文档中出现的频率，通常用词频除以文档总的词数来归一化。

-   **IDF：Inverse Document
    Frequency（逆文档频率）**，词的量化表示的一种方式。语料库中的文档总数/包含该词的文档数。

    TF-IDF是一种统计方法，用以评估**某个词对于文件集（语料库）的其中一份文件的重要程度**。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

    上述引用总结就是, 一个词语在一篇文章中出现次数越多,
    同时在所有文档中出现次数越少,
    越能够代表该文章。这也就是TF-IDF的含义。

    **参考**

    通俗易懂理解------TF-IDF与TextRank

    [[https://zhuanlan.zhihu.com/p/41091116]{.underline}](https://zhuanlan.zhihu.com/p/41091116)

### 语言模型（LM）

**语言模型（Language
Model）**可以简单理解为一个句子s在所有句子中出现的概率分布**P(s)**。而考虑到s是一个由词构成的序列w~1~,w~2~,\...w~n~，P(s)的概率可表示为：

**P(s) = P(w~1~,w~2~,\...,w~n~)**

再展开得到：

**P(w~1~,w~2~,\...,w~n~) =
P(w~1~)·P(w~2~\|w~1~)·P(w~3~\|w~1,~w~2~)···P(w~n~\|w~1~,w~2~,\...,w~n-1~)**

**参考**

[[https://zhuanlan.zhihu.com/p/43453548]{.underline}](https://zhuanlan.zhihu.com/p/43453548)

### n-grams（n元语法）

语言模型的计算和存储复杂度会随着词汇表和文本序列的增长而变得过大而无法计算。n-grams（n元语法）通过**马尔可夫假设**（虽然并不一定成立）简化了语言模型的计算。这里的马尔可夫假设是指一个词的出现只与前面n-1个词相关，即**n-1阶马尔可夫链（Markov
chain of
order n）**。例如n=1，那么有P(w~3~∣w~1~,w~2~)=P(w~3~∣w~2~)。如果基于n−1阶马尔可夫链，我们可以将语言模型改写为：

**P(w~1~,w~2~,...,w~T~) ≈**
$\prod_{t = 1}^{T}\mathbf{P}$**(w~t~∣w~t−(n−1)~,...,w~t−1~)**

以上也叫**n元语法（n-grams）**。它是基于n-1阶马尔可夫链的概率语言模型。当n分别为1、2和3时，我们将其分别称作**一元语法（unigram）**、**二元语法（bigram）**和**三元语法（trigram）**。例如，长度为4的序列w~1~,w~2~,w~3~,w~4~在一元语法、二元语法和三元语法中的概率分别为：

**一元语法：P(w~1~,w~2~,w~3~,w~4~)=P(w~1~)·P(w~2~)·P(w~3~)·P(w~4~)**

**二元语法：P(w~1~,w~2~,w~3~,w~4~)=P(w~1~)·P(w~2~∣w~1~)·P(w~3~∣w~2~)·P(w~4~∣w~3~)**

**三元语法：P(w~1~,w~2~,w~3~,w~4~)=P(w~1~)·P(w~2~∣w~1~)·P(w~3~∣w~1~,w~2~)·P(w~4~∣w~2~,w~3~)**

当n较小时，n元语法往往并不准确。例如，在一元语法中，由三个词组成的句子"你走先"和"你先走"的概率是一样的。然而，当n较大时，n元语法需要计算并存储大量的词频和多词相邻频率。

任务
----

NLP的任务比较复杂，并且还在不断发展，并没有一个较为官方的或者统一的分类。

传统的NLP任务定义泛指和自然语言相关的任务，因此从传统意义上说：

-   语音识别（Speech recognition，语音转文字）

-   语音合成（Text-to-Speech，文字转语音）

-   语音分段（Speech segmentation）

-   OCR（Optical Character Recognition，光学字符识别）

这些也应该算是NLP。但现在在深度学习研究领域里，所说的NLP任务只包括**文字相关**的任务，这可能是因为大部分难点和主要的研究方向都集中在这部分。

NLP的任务种类比较复杂：

-   **种类繁多**，且任务之间的关系交织密切（有交叉，有包含）

-   根据关注点和能力（比如知识图谱）的发展，也在不断的**发展新任务**

-   **分类方式**和维度也各有**不同**，不同分类方式中的任务名称和定义可又能会有**区别**

-   由于语言的相关特性，以及该语言NLP的发展，**不同语言**的任务也会有所区别

### 语言学NLP任务分类

以下是维基百科的**Natural Language
Processing**词条里，对NLP任务的分类，这里有些是可以应用的真实世界的任务，有些则是其他任务的子任务。

维基百科的NLP任务分类的特点是：

-   基于英文

-   比较详细

-   更学术化

-   更偏向**语言学**

可以将此作为理解神经网络NLP方向的任务分类基础。

-   **Text and Speech Processing**：**文本和语音处理**

    -   **OCR**：**光学字符识别**，识别图像中的文字

    -   **Speech Recognition**：**语音识别**，语音片段转化为文字

    -   **Speech
        > Segmentation**：**语音分段**，将语音片段分词，语音识别的子任务

    -   **TTS（Text to Speech）**：**语音合成**，文字转化为语音

    -   **Tokenization**：也叫**Word
        > Segmentation**，**分词**，将文字片段拆分成词

-   **Morphological Analysis**：**词法分析**

    -   **Lemmatization**：**词元化**，将词还原为其基本形式，例如去掉各种时态语态的变化，通常是通过字典来完成

    -   **Stemming**：**词干提取**，类似Lemmatization，区别在于Stemming的工作方式相对简单粗暴，一般是通过切除前缀或者后缀来实现。

    -   **Morphological
        > Segmentation**：**词分割**，将词拆分成语素，例如"unbreakable"
        > 输出"un-break-able"

    -   **POST（Part-of-speech
        > Tagging）**：**词性标注**，标定词汇的词性（动词还是名词）

-   **Syntactic Analysis**：**句法分析**

    -   **Grammar
        > Induction**：**语法归纳**，根据输入，产生出该语言的语法规则

    -   **Sentence Breaking**：**句子分割**，给定一段文本，切分成句子

    -   **Parsing**：**解析**，输入句子，生成解析树（Parsing tree）

-   **Lexical Semantics**：**词汇语义**

    -   **Lexical
        > Semantics**：**词汇语义**，单个词汇在上下文中的计算意义（比如词嵌入）

    -   **Distributional
        > Semantics**：**分布式语义**，分布式语义基于一个分布式假设：有类似分布的语义元素有类似的意义（即在同样的上下文环境中出现的词有类似的意思）

    -   **NER（Named Entity
        > Recognition）**：**命名实体识别**，识别文本中具有特定意义的实体，包括人名、地名、机构名、专有名词等，以及时间、数量、货币、比例等文字。

    -   **Sentiment
        > Analysis**：**情感分析**，分析文本的情感倾向（正面/负面/中性）

    -   **Terminology
        > Extraction**：**关键词提取**，从文本中提取相关的关键词

    -   **Word Sense
        > Disambiguation**：**词义消歧**，很多词都有若干意义，该任务是为了选择词汇在上下文环境中到底是哪个意义。

-   **Relational Semantics**：**关系语义**

    -   **RE（Relationship Extraction）**：**关系抽取**，类似Information
        > Extraction，从输入中检测并分类语义关系

    -   **Semantic
        > Parsing**：**语义解析**，将自然语言表达转换成逻辑形式，通常由几个更基本的子任务构成。可以应用于更高级的任务比如机器翻译、智能问答中。

    -   **SRL（Semantic Role Labeling）**：也叫**Shallow Semantic
        > Parsing**，**语义角色标注**，给句子中的词添加语义角色标签，简单的比如主谓宾等。

-   **Discourse**：**言谈**

    -   **Coreference
        > resolution**：**共指消解**，从一段文本或者对话中找出指代同一个人或事物的所有词汇

    -   **DA（Discourse
        > Analysis）**：**言谈分析**，这是个更偏语言学的任务

    -   **Implicit Semantic Role
        > Labelling**：**隐性语义角色标注**，在给句子做**语义角色标注**的基础上，给词标注出句子间的语义角色，有点类似**共指消解**。

    -   **TE（Recognizing Texture
        > entailment）**：**文字蕴涵**，判断两段文本（文本T和假设H）的关系是正向蕴涵、矛盾蕴涵还是独立蕴涵。

    -   **Topic Segmentation and
        > Recognition**：**主题分割和识别**，将文本分割为段落，并识别出段落的主题

-   **High-Level NLP
    Application**：高级NLP应用，每个都包含了若干较下层的子任务

    -   **Text Summarization**：**文本摘要**，也叫**Automatic
        > Summarization**，提取文章的主要内容

    -   **Book Generation**：**书籍生成**

    -   **Dialogue Management**：**对话管理**，也叫**CA（Conversational
        > Agent）**，类似**智能问答**，区别在于并不一定是以问答的形式

    -   **Document AI**：**文档AI**，从文档中抽取所需的特定数据

    -   **Grammatical Error
        > Correction**：**语法错误纠正**，包括语法错误的检测和纠正

    -   **Machine Translation**：**机器翻译**

    -   **NLG（Natural Language
        > Generation）**：**自然语言生成**，这里指的是从数据库等结构化数据中转换为（人类易于读懂的）文本。

    -   **NLU（Natural Language
        > Understanding）**：**自然语言理解**，跟NLG相反，将文本段落转换为计算机更易于理解的结构

    -   **Question
        > Answering**：**智能问答**，通常是简单的有确定答案的问题（比如"今天星期几"），但有时候也会需要考虑开放式的问题。

**参考**

[[https://en.wikipedia.org/wiki/Natural\_language\_processing]{.underline}](https://en.wikipedia.org/wiki/Natural_language_processing)

### 神经网络NLP任务分类

神经网络中的NLP任务和维基百科中的语言学NLP任务分类多有重合，下面为在语言学NLP的基础上所做的补充：

-   **Language Modelling**：语言模型，参考上节

-   **Constituency Parsing（成分句法分析）**：属于Semantic Parsing

-   **Dependency Parsing（依赖句法分析）**：属于Semantic Parsing

-   **Stance
    Detection（立场检测）**：分类任务，输入是两个序列，输出是一个类别，表示后面的序列是否与前面的序列站在同一立场。常用的立场检测包括
    SDQC 四种标签，支持 (Support)，否定 (Denying)，怀疑
    (Querying)，Commenting (注释)。

-   **Veracity Prediction（事实验证）**：，也叫Fake News Detection。

-   **Search
    Engine（搜索引擎）**：模型的输入是一个关键词或一个问句和一堆文章，输出是每篇文章与该问句的相关性。谷歌有把
    BERT 用在搜素引擎上，在语义理解上得到了提升。

-   **New Word Detection（新词发现）**：

-   **Text Classification（文本分类）**：

-   文本匹配

-   **IE（Information Extraction，信息抽取）**：

-   **Reading Comprehension（阅读理解）**：

-   **Knowledge Graph（知识图谱）**：

-   **AMR（Abstract Meaning Representation，抽象意义表示）**：

**参考**

[[https://paperswithcode.com/area/natural-language-processing]{.underline}](https://paperswithcode.com/area/natural-language-processing)

[[https://zhuanlan.zhihu.com/p/163281686]{.underline}](https://zhuanlan.zhihu.com/p/163281686)

[[https://zhuanlan.zhihu.com/p/50755570\
]{.underline}](https://zhuanlan.zhihu.com/p/50755570)

透过现象看本质，仅从任务的输出、输出的角度看，可以将NLP大致分为NLU和NLG：

-   **NLU：Natural Languange
    Understand，自然语言理解**，通常输入是文本序列，输出是分类（或者标签），又可以分为：

    -   输入是句子，输出是分类：情感分析、文本分类

    -   输入是句子对，输出是分类

    -   输入是句子，输出是标签

    -   \...

-   **NLG：Natural Language
    Generate，自然语言生成**，通常输入是文本序列，输出也是文本序列

    -   机器翻译

    -   智能问答

    -   文本摘要

    -   \...

衡量标准
--------

### Accuracy

参考《[[概念定义 \> NN相关 \> 二分类任务衡量标准]{.underline}](\l)》

### F1-Score

参考《[[概念定义 \> NN相关 \> 二分类任务衡量标准]{.underline}](\l)》

### BLEU

**BiLingual Evaluation
Understudy**，一个用于衡量机器翻译的指标。其背后的原则是：机器翻译的结果越接近人工翻译的结果，得分越高。

数据集
------

NLP的数据集也叫语料库（corpus）

### 中文情感分析数据集汇总

[[https://github.com/DinghaoXi/chinese-sentiment-datasets]{.underline}](https://github.com/DinghaoXi/chinese-sentiment-datasets)

[[https://zhuanlan.zhihu.com/p/80029681]{.underline}](https://zhuanlan.zhihu.com/p/80029681)

### 中文摘要数据集汇总

[[https://zhuanlan.zhihu.com/p/341398288]{.underline}](https://zhuanlan.zhihu.com/p/341398288)

### ChnSentiCorp

中文的情感分析语料库，来自携程评论，文本分类任务，将每个样本（评论）标示为正面或者负面。

下载地址：

[[http://file.hankcs.com/corpus/ChnSentiCorp.zip]{.underline}](http://file.hankcs.com/corpus/ChnSentiCorp.zip)

### SQuAD

[[https://rajpurkar.github.io/SQuAD-explorer/]{.underline}](https://rajpurkar.github.io/SQuAD-explorer/)

### GLUE

**GLUE（General Language Understanding
Evaluation）**是目前业界通用的，用于测试NLP模型能力的Benchmark，包含了若干任务（的数据集及衡量标准），主流的NLP模型都在上面测试过。GLUE全部是分类任务，包括以下数据集/任务：

-   **CoLA**：The Corpus of Linguistic
    Acceptability，语言可接受性语料库，即判断输入**可接受程度**（是否合乎语法），语料来自语言理论的书籍和期刊，每个句子被标注为是否合乎语法的单词序列。**二分类任务**，结果为1（合乎语法）或者0（不合乎语法）。

<!-- -->

-   **SST-2**：The Stanford Sentiment
    Treebank，斯坦福感情树库。包含电影评论中的句子和它们情感的人类注释。是判断句子的**情感分析任务**，是个**二分类任务**，结果为正面或者负面。

-   **MRPC**：The Microsoft Research Paraphrase
    Corpus，微软研究院释义语料库，**相似性和释义任务**，是从在线新闻源中自动抽取句子对成为语料库。**二分类任务**，判断输入的两者是否互为释义。

-   **STSB**：The Semantic Textual Similarity
    Benchmark，语义文本相似性基准测试，**相似性和释义任务**，判断输入句子对的相似性。是从新闻标题、视频标题、图像标题以及自然语言推断数据中提取的句子对的集合，每对都是由人类注释的，任务就是预测这些相似性得分（从0到5），本质上是一个**回归问题**，但是依然可以用分类的方法，可以归类为句子对的文本**五分类任务**。

-   **QQP**：The Quora Question Pairs,
    Quora问题对数集，**相似性和释义任务**，是社区问答网站Quora中问题对的集合。任务是确定一对问题在语义上是否等效。**二分类任务**，等效或者不等效。

-   **MNLI**：The Multi-Genre Natural Language Inference Corpus,
    多类型自然语言推理数据库，**自然语言推断任务**，是通过众包方式对句子对进行文本蕴含标注的集合。给定前提（premise）语句和假设（hypothesis）语句，任务是预测前提语句是否包含假设（蕴含,
    entailment），与假设矛盾（矛盾，contradiction）或者两者都不（中立，neutral）,是个**三分类任务**。

-   **QNLI**：Qusetion-answering
    NLI，问答自然语言推断，**自然语言推断任务**。QNLI是从另一个数据集The
    Stanford Question Answering Dataset(斯坦福问答数据集, SQuAD
    1.0)转换而来的。SQuAD
    1.0是有一个问题-段落对组成的问答数据集，其中段落来自维基百科，段落中的一个句子包含问题的答案。这里可以看到有个要素，来自维基百科的段落，问题，段落中的一个句子包含问题的答案。通过将问题和上下文（即维基百科段落）中的每一句话进行组合，并过滤掉词汇重叠比较低的句子对就得到了QNLI中的句子对。相比原始SQuAD任务，消除了模型选择准确答案的要求；也消除了简化的假设，即答案适中在输入中并且词汇重叠是可靠的提示。是个**二分类任务**，蕴含和不蕴含。

-   **RTE**：The Recognizing Textual Entailment
    datasets，识别文本蕴含数据集，**自然语言推断任务**，它是将一系列的年度文本蕴含挑战赛的数据集进行整合合并而来的，包含RTE1，RTE2，RTE3，RTE5等，这些数据样本都从新闻和维基百科构建而来。将这些所有数据转换为**二分类任务**（蕴含和不蕴含），对于三分类的数据，为了保持一致性，将中立（neutral）和矛盾（contradiction）转换为不蕴含（not
    entailment）。

-   **WNLI**：Winograd
    NLI，Winograd自然语言推断，**自然语言推断任务**，数据集来自于竞赛数据的转换。Winograd
    Schema
    Challenge，该竞赛是一项阅读理解任务，其中系统必须读一个带有代词的句子，并从列表中找到代词的指代对象。这些样本都是都是手动创建的，以挫败简单的统计方法：每个样本都取决于句子中单个单词或短语提供的上下文信息。为了将问题转换成句子对分类，方法是通过用每个可能的列表中的每个可能的指代去替换原始句子中的代词。任务是预测两个句子对是否有关（蕴含、不蕴含），因此是个**二分类任务**。训练集两个类别是均衡的，测试集是不均衡的，65%是不蕴含。

可以看出来，GLUE包含的任务全部是**分类任务**，输入是单个句子或者句子对：

-   输入单句，输出二分类：CoLA、SST-2

-   输入句对，输出二分类：MRPC、QQP、QNLI、RTE、WNLI

-   输入句对，输出多分类：STSB、MNLI

    以下为GLUE官方网站上的排行榜LeaderBoard：

    ![](/home/jimzeus/outputs/AANN/images/media/image365.png){width="6.538194444444445in"
    height="3.7215277777777778in"}

**论文**

[[https://arxiv.org/pdf/1804.07461.pdf]{.underline}](https://arxiv.org/pdf/1804.07461.pdf)

官方**代码**

[[https://github.com/nyu-mll/GLUE-baselines]{.underline}](https://github.com/nyu-mll/GLUE-baselines)

官方网站：

[[https://gluebenchmark.com/]{.underline}](https://gluebenchmark.com/)

**参考**

[[https://zhuanlan.zhihu.com/p/135283598]{.underline}](https://zhuanlan.zhihu.com/p/135283598)

### SuperGLUE

**论文**

[[https://arxiv.org/pdf/1905.00537.pdf]{.underline}](https://arxiv.org/pdf/1905.00537.pdf)

官方网站：

[[https://super.gluebenchmark.com/]{.underline}](https://super.gluebenchmark.com/)

代码：

[[https://jiant.info/]{.underline}](https://jiant.info/)

### CLUE

CLUE（Chinese
GLUE），即中文GLUE。中文语言理解测评基准，包括代表性的数据集、基准(预训练)模型、语料库、排行榜，是用于衡量中文NLP模型能力的一个通用的数据集/任务集合。

具体的任务如下：

-   **AFQMC**：Ant Financial Question Matching
    Corpus，**蚂蚁金融语义相似度**。输入为句子对，输出二分类。每一条数据有三个属性，从前往后分别是
    句子1，句子2，句子相似度标签。其中label标签，1
    表示sentence1和sentence2的含义类似，0表示两个句子的含义不同。

-   **TNEWS'**：Short Text Classification for
    News。**文本分类任务**。输入为句子，输出为多分类。每一条数据有三个属性，从前往后分别是
    分类ID，分类名称，新闻字符串（仅含标题）。该数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等。

-   **IFLYTEK'**：Long Text
    Classification，**长文本分类任务**。输入为长文本，输出为多分类。该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别：\"打车\":0,\"地图导航\":1,\"免费WIFI\":2,\"租车\":3,....,\"女性\":115,\"经营\":116,\"收款\":117,\"其他\":118(分别用0-118表示)。每一条数据有三个属性，从前往后分别是类别ID，类别名称，文本内容。

-   **OCNLI**：Original Chinese Natural Language
    Inference，**中文原版自然语言推理**。输入为句子对，输出为多分类。是第一个非翻译的、使用原生汉语的大型中文自然语言推理数据集。
    OCNLI包含5万余训练数据，3千验证数据及3千测试数据。除测试数据外，我们将提供数据及标签。测试数据仅提供数据。OCNLI为中文语言理解基准测评（CLUE）的一部分。中文原版数据集OCNLI替代了CMNLI，使用bert\_base作为初始化分数。

-   **CMNLI**：Chinese Multi-Genre
    NLI，中文语言推理任务（已被OCNLI替代）。CMNLI数据由两部分组成：XNLI和MNLI。数据来自于fiction，telephone，travel，government，slate等，对原始MNLI数据和XNLI数据进行了中英文转化，保留原始训练集，合并XNLI中的dev和MNLI中的matched作为CMNLI的dev，合并XNLI中的test和MNLI中的mismatched作为CMNLI的test，并打乱顺序。该数据集可用于判断给定的两个句子之间属于蕴涵、中立、矛盾关系。每一条数据有三个属性，从前往后分别是
    句子1，句子2，蕴含关系标签。其中label标签有三种：neutral，entailment，contradiction。

-   **WSC2020**：**WSC
    Winograd模式挑战中文版**，Winograd模式是图灵测试的一个变种，旨在判定AI系统的常识推理能力。参与挑战的计算机程序需要回答一种特殊但简易的常识问题：代词消歧问题，即对给定的名词和代词判断是否指代一致。其中label标签，true表示指代一致，false表示指代不一致。

-   **CSL**：Keyword
    Recognition，**论文关键词识别任务**。中文科技文献数据集（CSL）包含中文核心论文摘要及其关键词。
    用TF-IDF生成伪造关键词与论文真实关键词混合，生成摘要-关键词对，关键词中包含伪造的则标签为0。每一条数据有四个属性，从前往后分别是
    数据ID，论文摘要，关键词，真假标签。

-   **CMRC2018**：Reading Comprehension for Simplified
    Chinese，**简体中文阅读理解任务**。CMRC2018是讯飞的中文阅读理解评测。

-   **DRCD**：Reading Comprehension for Traditional
    Chinese，**繁体中文阅读理解任务**。台達閱讀理解資料集 Delta Reading
    Comprehension Dataset
    (DRCD)，屬於通用領域繁體中文機器閱讀理解資料集。
    本資料集期望成為適用於遷移學習之標準中文閱讀理解資料集。

-   **ChID**：**成语阅读理解填空 Chinese IDiom Dataset for Cloze
    Test**。成语完形填空，文中多处成语被mask，候选项中包含了近义的成语。

-   **C3**：Multiple-Choice Chinese Machine Reading
    Comprehension，中文多选阅读理解数据集，包含对话和长文等混合类型数据集。

CLUE包含的部分任务是**分类任务**，输入是单个句子或者句子对：

-   输入单句，输出分类：TNEWS'、IFLYTEK'

-   输入句对，输出分类：AFQMC、OCNLI、CMNLI

-   输入句子+词，输出分类：WSC2020、CSL

官方网站：

[[https://www.cluebenchmarks.com/]{.underline}](https://www.cluebenchmarks.com/)

**论文**

[[https://arxiv.org/pdf/2004.05986.pdf]{.underline}](https://arxiv.org/pdf/2004.05986.pdf)

**代码**

旧版：[[https://github.com/chineseGLUE/chineseGLUE]{.underline}](https://github.com/chineseGLUE/chineseGLUE)

新版：[[https://github.com/CLUEbenchmark/CLUE]{.underline}](https://github.com/CLUEbenchmark/CLUE)

框架
----

严格来说，这里列出的也包括各种NLP模型的库/工具包，而不仅仅是框架。

### SpaCy

### NLTK

### Stanford CoreNLP（2014）

Stanford CoreNLP是个已经训练好的NLP模型，支持多种语言。

在各个语言上的能力如下表：

![](/home/jimzeus/outputs/AANN/images/media/image366.png){width="5.554166666666666in"
height="3.4881944444444444in"}

官方：

[[https://stanfordnlp.github.io/CoreNLP/]{.underline}](https://stanfordnlp.github.io/CoreNLP/)

**论文**

[[https://nlp.stanford.edu/pubs/StanfordCoreNlp2014.pdf]{.underline}](https://nlp.stanford.edu/pubs/StanfordCoreNlp2014.pdf)

**代码**

[[https://github.com/stanfordnlp/CoreNLP]{.underline}](https://github.com/stanfordnlp/CoreNLP)

**参考**

[[CSDN：StanfordNLP的安装及使用]{.underline}](https://blog.csdn.net/lizzy05/article/details/87483539)

[[知乎：Stanford
CoreNLP入门指南]{.underline}](https://zhuanlan.zhihu.com/p/137226095)

[[StanfordCoreNLP的简单使用]{.underline}](https://www.cnblogs.com/maoerbao/p/13019276.html)

### Gensim

Gensim
是用于特定**文本主题建模**的高端行业级软件。它的功能非常强大，独立于平台，并且具有可扩展性。不仅可以用来判断两个报纸文章之间的**语义相似性**，而且可以利用简单的函数调用来执行此操作并返回其相似度分数。

任务：主题建模，文本摘要，语义相似度

官网：

[[https://radimrehurek.com/gensim/]{.underline}](https://radimrehurek.com/gensim/)

**代码**

[[https://github.com/RaRe-Technologies/gensim]{.underline}](https://github.com/RaRe-Technologies/gensim)

### OpenNMT

OpenNMT
是用于机器翻译和序列学习任务的便捷而强大的工具。其包含的高度可配置的模型和培训过程，让它成为了一个非常简单的框架。

官网：

[[https://opennmt.net/]{.underline}](https://opennmt.net/)

**代码**

[[https://github.com/OpenNMT/OpenNMT-py]{.underline}](https://github.com/OpenNMT/OpenNMT-py)

### ParlAI

ParlAI 是 Facebook 的＃1
框架，用于共享、训练和测试用于各种对话任务的对话模型。其提供了一个支持多种参考模型、预训练模型、数据集等的多合一环境。

官网：

[[https://parl.ai/]{.underline}](https://parl.ai/)

**代码**

[[https://github.com/facebookresearch/ParlAI]{.underline}](https://github.com/facebookresearch/ParlAI)

### DeepPavlov

官网：

[[https://deeppavlov.ai/]{.underline}](https://deeppavlov.ai/)

**代码**

[[https://github.com/deepmipt/DeepPavlov]{.underline}](https://github.com/deepmipt/DeepPavlov)

### SnowNLP

SnowNLP是个**非神经网络**的NLP库，比较老，最后更新已经是2017年，优点是**非常易用**，且由于并未使用神经网络，**速度较快**。

SnowNLP是一个python写的类库，可以方便的处理中文文本内容，是受到了[TextBlob](https://github.com/sloria/TextBlob)的启发而写的，由于现在大部分的自然语言处理库基本都是针对英文的，于是写了一个方便处理中文的类库，并且和TextBlob不同的是，这里没有用NLTK，所有的算法都是自己实现的，并且自带了一些训练好的字典。

**代码**

[[https://github.com/isnowfy/snownlp]{.underline}](https://github.com/isnowfy/snownlp)

使用示例（情感分析）：

\$ pip install snownlp

from snownlp import SnowNLP

s = SnowNLP(content)

result = s.sentiments

训练示例（情感分析）：

from snownlp import sentiment

sentiment.train(\'neg.txt\', \'pos.txt\')

sentiment.save(\'sentiment.marshal\')

之后修改snownlp/sentiment/\_\_init\_\_.py里的data\_path指向刚训练好的文件即可

### Senta

百度推出的情感分析的NLP库

**代码**

[[https://github.com/baidu/senta]{.underline}](https://github.com/baidu/senta)

**论文**

[[https://arxiv.org/abs/2005.05635]{.underline}](https://arxiv.org/abs/2005.05635)

使用示例（情感分析）：

from senta import Senta

my\_senta = Senta()

my\_senta.init\_model(task=\"sentiment\_classify\")

result1 = my\_senta.predict(text)

### HuggingFace Transformers

Transformers是[[HuggingFace]{.underline}](\l)推出的包含各种基于**transformer结构**的NLP模型的工具包。包含了Pytorch和Tensorflow两种格式的模型。

2017年，transformer结构被提出，之后NLP的方向开始从RNN转向Transformer，2018年，Google发布了基于Transformer的NLP预训练模型BERT，之后基于BERT的各种模型（XLNet、ALBERT、RoBERTa\...）层出不穷，不断刷新纪录。

BERTs模型虽然很香，但是用起来还是有一些障碍，比如：

-   预训练需要大量的资源，一般研究者无法承担。以RoBERTa为例，它在160GB文本上利用1024块32GB显存的V100卡训练得到，如果换算成AWS上的云计算资源的话，但这一模型就需要10万美元的开销。

-   很多大机构的预训练模型被分享出来，但没有得到很好的组织和管理。

-   BERT系列的各种模型虽然师出同源，但在模型细节和调用接口上还是有不少变种，用起来容易踩坑

为了让这些预训练语言模型使用起来更加方便，Huggingface在github上开源了Transformers。这个项目开源之后备受推崇，截止2020年5月，已经累积了26k的star和超过6.4k的fork。

Transformers最早的名字叫做**pytorch-pretrained-bert**，推出于google
BERT之后。顾名思义，它是基于pytorch对BERT的一种实现。pytorch框架上手简单，BERT模型性能卓越，集合了两者优点的pytorch-pretrained-bert自然吸引了大批的追随者和贡献者。

其后，在社区的努力下，GPT、GPT-2、Transformer-XL、XLNET、XLM等一批模型也被相继引入，整个家族愈发壮大，这个库适时地更名为**pytorch-transformers**。

之后又加入了pytorch和TF2.0+的互操作性，使得pytorch和tensorflow两大阵营的模型之间可以相互转换，项目的名字，也改成了现在的Transformers。时至今日，Transformers已经在100+种人类语言上提供了32+种预训练语言模型。

**官方文档：**

[[https://huggingface.co/transformers/]{.underline}](https://huggingface.co/transformers/)

**哈工大和讯飞联合实验室训练的中文模型：**

[[https://huggingface.co/hfl]{.underline}](https://huggingface.co/hfl)

**参考**

[[https://zhuanlan.zhihu.com/p/141527015]{.underline}](https://zhuanlan.zhihu.com/p/141527015)

**代码**

Transformers：[[https://github.com/huggingface/transformers]{.underline}](https://github.com/huggingface/transformers)

数据集操作库（原名nlp）：[[https://github.com/huggingface/datasets]{.underline}](https://github.com/huggingface/datasets)

**论文**

[[https://arxiv.org/pdf/1910.03771.pdf]{.underline}](https://arxiv.org/pdf/1910.03771.pdf)

#### 介绍

##### 主要组件

Transformers提供了三个主要的组件：

-   **Configuration**配置。存储模型和分词器的参数，诸如词表大小，隐层维数，dropout
    rate等。配置类对深度学习框架是**透明的**。

-   **Tokenizer**分词器。每个模型都有对应的分词器，存储token到index的映射，负责每个模型特定的序列编码解码流程，比如BPE(Byte
    Pair
    Encoding)，SentencePiece等等。也可以方便地添加特殊token或者调整词表大小，如CLS、SEP等等。

-   **Model**模型。实现模型的计算图和编码过程，实现前向传播过程，通过一系列self-attention层直到最后一个隐藏状态层。

这三个组件的基类分别为：

-   PreTrainedConfig：所有配置类的基类

-   PreTrainedTokenizerBase：分词器类的基类，有两个子类分别是所有慢速分词器的基类（PreTrainedTokenizer）和快速分词器的基类（PreTrainedTokenizerFast）

-   PreTrainedModel/TFPreTrainedModel：分别为pytorch和tensorflow的模型的基类

##### 具体模型

对于每个具体的模型（Bert、Albert、Electra等），有三个类与基本组件相关，分别用于描述该模型的配置、分词器、模型，例如对于BERT模型，有BertModel、BertConfig、BertTokenizer。这三个类都可以通过成员函数from\_pretrained()加载（参数为模型ID）具体权重、分词器和超参数，并通过成员函数save\_pretrained()保存这些值到本地。

对于每个具体模型Model，通常有以下相关的类：

-   XXXPreTrainedModel：继承自PreTrainedModel，是该系列模型的基类，比如BertPreTrainedModel

-   XXXModel：继承自XXXPreTrainedModel，是该类模型的"裸"模型（bare
    model），即输出隐变量，不接任何任务的Head

-   XXXForYYYY：继承自XXXPreTrainedModel，是该类模型为了各种不同任务的实现，通常结构为一个裸模型+不同任务的head（或者若干层），比如BertForMaskedLM、BertForSequenceClassification。

##### 自动类

Transformers的作者们还为以上组件提供了一系列**自动类**，这些自动类主要提供了from\_pretrained()函数，能够从一个模型ID（如"bert-base-cased"）里自动推测出来应该实例化哪种配置类、分词器类和模型类。

Transformers提供两大类的模型架构，一类用于语言生成NLG任务，比如GPT、GPT-2、Transformer-XL、XLNet和XLM，另一类主要用于语言理解任务，如Bert、DistilBert、RoBERTa、XLM。

-   AutoConfig：根据模型ID自动返回对应的config，对用户透明

-   AutoTokenizer：根据模型ID自动找到的对应的tokenizer

-   AutoModel：根据模型ID自动返回对应的模型

-   AutoModelXXX：任务相关，根据模型ID自动返回对应的带任务head的模型

这几个类都是通过AutoXXX.from\_pretrained()函数（参数为模型ID）快速的载入（配置中的超参数、分词器中的词汇表、模型中的权重），具体实现是通过模型ID找到对应的模型/分词器/配置，调用其from\_pretrained()函数。

##### from\_pretrained函数

**from\_pretrained()**是自动类（AutoModel、AutoTokenizer、AutoConfig）和具体模型、分词器（比如BertModel、BertTokenizer、BertConfig）都有的成员函数。

其主要参数为model
id（**模型ID**），或者称为**预训练模型名**，类型为字符串，可以从https://huggingface.co/models看到所有模型ID的列表，也可以是本地的路径。

具体模型/分词器/配置根据模型ID，调用from\_pretrained()函数获得某个预训练过的模型的各种参数，比如BertModel.from\_pretrained("bert-base-uncased")得到一个包括权重的BERT模型。而自动类则首先通过模型ID确定模型/分词器/配置（根据"bert-base-uncased"得到模型为BertModel），再调用具体模型/配置/分词器（的基类）的from\_pretrained()函数。

##### 混淆注意

跟模型相关的类和值有点多，这里来理一理，相关的参数维度包括：

-   **类型**（例如Bert、Electra）

-   **任务**（例如Sequence Classification、Question Ansering等）

-   **权重**：可以通过不同的模型ID来加载不同的预训练**权重**，预训练权重和模型的**类型**及**任务**都有关。

与之相关的包括

-   所有模型的基类：PreTrainedModel

-   模型自动类：比如AutoModel、AutoModelForSequenceClassification等,通过from\_pretrained函数返回具体的、针对**任务**的、带权重的预训练模型，除了AutoModel之外其他类（AutoModelXXX）和具体任务有关。

-   裸模型类：每种**类型**（Bert、Electra）的裸模型，不带任务head，比如BertModel

-   任务模型类：确定**类型**和**任务**的模型，带具体任务相关的head，例如BertForSequenceClassification

-   模型ID：字符串，对应带模型的预训练**权重**，例如"bert-base-uncased"或"hfl/chinese-bert-wwm-ext"，具体列表在https://huggingface.co/models

##### 流水线

**流水线（pipeline）**实现了端到端的操作过程，流水线主要包括了三个部分：分词器、模型、任务相关的后期处理。

-   Pipeline：流水线的基类

-   XXXPipeline：具体任务的流水线类，例如TextClassificationPipeline

而pipeline()函数返回一个带了参数的具体流水线类（Pipeline的子类）的实例，pipeline()函数的功能类似AutoXXX.from\_pretrained()。

-   Trainer/TFTrainer：训练器类

[[https://huggingface.co/transformers/main\_classes/trainer.html]{.underline}](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer)

#### 主要基类

各种基类均在transformers的主目录下：

-   modeling\_utils.py：pytorch模型相关的类

-   modeling\_tf\_utils.py：tensorflow模型相关的类

-   configuration\_utils.py：配置相关的类

-   tokenization\_xxx.py：分词器相关的类

##### 模型

-   modeling\_utils.PreTrainedModel：所有Pytorch模型的基类

    -   config\_class：

    -   base\_model\_prefix：模型的前缀，字符串

    -   \_keys\_to\_ignore\_on\_load\_missing：一系列正则表达式，当加载模型权重出错（缺失）时，忽略匹配这些正则表达式的张量名

    -   \_keys\_to\_ignore\_on\_load\_unexpected：一系列正则表达式，当加载模型权重出错（多余）时，忽略匹配这些正则表达式的张量名

    -   \_keys\_to\_ignore\_on\_save：一系列正则表达式，保存模型时要忽略

    -   from\_pretrained()：类方法，从配置中加载一个预训练的模型，并返回

-   modeling\_tf\_utils.TFPreTrainedModel：所有tensorflow模型的基类

##### 分词器的基类

PreTrainedTokenizerBase是所有分词器的基类，有两个类继承了它，分别是表示（慢速）分词器的PreTrainedTokenizer，和表示快速分词器的PreTrainedTokenizerFast。

-   tokenization\_utils\_base.PreTrainedTokenizerBase：所有分词器的基类

    -   \_\_call\_\_()：分词器的主函数，将输入文本转换为id

    -   decode()：将id解码为文本

-   tokenization\_utils.PreTrainedTokenizer：所有慢分词器（slow
    tokenizer）的基类，PreTrainedTokenizerBase的子类

-   tokenization\_utils\_fast.PreTrainedTokenizerFast：所有快分词器（fast
    tokenizer）的基类，PreTrainedTokenizerBase的子类

##### 配置

-   configuration\_utils.PreTrainedConfig：所有配置的基类

    -   

-   modeling\_outputs：包含描述各种任务的输出结果类

    -   MultipleChoiceModelOutput：多选择任务的输出

#### 自动类

自动类这个名称是有歧义的，其本身**并非**配置/分词器/模型，也不能实例化，主要作用是通过成员函数from\_pretrained()返回与参数匹配的配置/分词器/模型。

##### 自动配置

-   models.auto.configuration\_auto.CONFIG\_MAPPING：从描述**模型类型**的字符串（比如"bert"）到对应配置（比如BertConfig）的映射

-   models.auto.configuration\_auto.AutoConfig：该类不能实例化，只包含类函数

    -   for\_model()：根据CONFIG\_MAPPING，从**模型类型字符串**得到对应的配置

    -   from\_pretrained()：根据**模型ID**，返回具体的配置（比如BertConfig）

##### 自动分词器

-   models.auto.tokenization\_auto.TOKENIZER\_MAPPING：

-   models.auto.tokenization\_auto.tokenizer\_class\_from\_name()：

-   models.auto.tokenization\_auto.AutoTokenizer：该类不能实例化，存在的唯一作用就是提供from\_pretrained()函数

    -   from\_pretrained()：类方法，流程如下：

        -   首先根据模型ID，通过AutoConfig.from\_pretrained()找到对应的配置类

        -   再调用该类的from\_pretrained()函数返回一个预训练过的（即初始化过的）tokenizer

##### 自动模型

自动模型类AutoModel和AutoModelForXXX，通过from\_pretrained()函数，提供了根据模型ID得到带预训练权重的具体模型实例。

-   models.auto.modeling\_auto.MODEL\_MAPPING：从配置（比如BertConfig）到模型（比如BertModel）的映射

-   models.auto.modeling\_auto.MODEL\_FOR\_PRETRAINING\_MAPPING：从配置（比如BertConfig）到对应的用于预训练的模型（比如BertForPreTraining）的映射

-   models.auto.modeling\_auto.MODEL\_WITH\_LM\_HEAD\_MAPPING：从配置（比如BertConfig）到带LM头的模型（比如BertForMaskedLM）的映射

-   models.auto.modeling\_auto.MODEL\_FOR\_CASUAL\_LM\_MAPPING：从配置（比如BertConfig）到带模型（比如BertLMHeadModel）的映射

-   models.auto.modeling\_auto.MODEL\_FOR\_MASKED\_LM\_MAPPING：从配置（比如BertConfig）到masked
    LM模型（比如BertForMaskedLM）的映射

-   models.auto.modeling\_auto.MODEL\_FOR\_SEQ\_TO\_SEQ\_CASUAL\_LM\_MAPPING：从配置（比如T5Config）到模型（比如T5ForConditionalGeneration）的映射

-   models.auto.modeling\_auto.MODEL\_FOR\_SEQUENCE\_CLASSIFICATION\_MAPPING：从配置（比如BertConfig）到序列分类模型（比如BertForSequenceClassification）的映射

-   models.auto.modeling\_auto.MODEL\_FOR\_QUESTION\_ANSWERING\_MAPPING：从配置（比如BertConfig）到问答模型（比如BertForQuestionAnswering）的映射

-   models.auto.modeling\_auto.MODEL\_FOR\_TABLE\_QUESTION\_ANSWERING\_MAPPING：从配置（比如TapasConfig）到模型（比如TapasForQuestionAnswering）的映射

-   models.auto.modeling\_auto.MODEL\_FOR\_TOKEN\_CLASSIFICATION\_MAPPING：从配置（比如BertConfig）到token分类模型（比如BertForTokenClassification）的映射

-   models.auto.modeling\_auto.MODEL\_FOR\_MUTIPLE\_CHOICE\_MAPPING：从配置（比如BertConfig）到multiple
    choice模型（比如BertForMultipleChoice）的映射

-   models.auto.modeling\_auto.MODEL\_FOR\_NEXT\_SENTENCE\_PREDICTION\_MAPPING：从配置（比如BertConfig）到NSP任务模型（比如BertForNextSentencePrediction）的映射

-   models.auto.modeling\_auto.AutoModel：该类不能实例化，提供from\_pretrained()函数和from\_config()函数

    -   from\_config()：类方法，通过MODEL\_MAPPING映射，根据传入的配置（比如BertConfig）得到对应的模型（比如BertModel），此时并不加载权重

    -   from\_pretrained()：类方法，参数为模型ID，返回预训练模型，流程如下：

        -   通过AutoConfig.from\_pretrained()函数，根据模型ID得到对应的配置（比如BertConfig）

        -   通过MODEL\_MAPPING映射，根据配置得到对应的模型（比如BertModel）

        -   调用模型的from\_pretrained()函数（来自基类PreTrainedModel，参数为模型ID）加载权重，得到包含权重的预训练模型

-   models.auto.modeling\_auto.AutoModelForXXX：所有的这些类都类似AutoModel，不能实例化，提供from\_pretrained()函数和from\_config()函数

    -   from\_config()：类方法，通过MODEL\_FOX\_XXX\_MAPPING映射，根据传入的配置（比如BertConfig）得到对应的模型（比如BertForXXX），此时并不加载权重

    -   from\_pretrained()：类方法，参数为模型ID，返回预训练模型，流程如下：

        -   通过AutoConfig.from\_pretrained()函数，根据**模型ID**得到对应的配置（比如BertConfig）

        -   通过MODEL\_FOR\_XXX\_MAPPING映射，根据配置得到对应的模型（比如BertForXXX）

        -   调用模型的from\_pretrained()函数（来自基类PreTrainedModel，参数为模型ID）加载权重，得到包含权重的预训练模型

#### pipelines（流水线）

流水线（pipeline）是Transformer最易用的使用方法，提供了一个从端到端的工具，包括：

-   Sentiment analysis：情感分析，输入文本是正面还是负面情绪

-   Text Generation：文本生成，根据文本生成后续

-   Name Entity
    Recognition（NER）：命名实体识别，标示输入文本中每个entity（人名、地名、专有名词等等）

-   Question Answering：问题回答

-   Filling masked text：填空，替换\[MASK\]标记

-   Summarization：摘要，给一段长文本摘要

-   Translation：翻译

-   Feature Extraction：特征提取，返回用于表示输入文本的张量

##### 使用示例

\>\>\> from transformers import pipeline

\>\>\> classifier = pipeline(\'sentiment-analysis\')

\>\>\> classifier(\'We are very happy to show you the Transformers
library.\')

\[{\'label\': \'POSITIVE\', \'score\': 0.9997795224189758}\]

或者可以指定模型：

\>\>\> classifier = pipeline(\'sentiment-analysis\',
model=\"nlptown/bert-base-multilingual-uncased-sentiment\")

##### Python包结构

-   pipelines.\_\_init\_\_.SUPPORTED\_TASK：pipeline支持的任务，以及对应的流水线类、自动模型类、对应的缺省模型ID，包括：

    -   \"feature-extraction\"：特征提取任务

        -   \"impl\"：FeatureExtractionPipeline，对应的流水线类

        -   \"tf\"：TFAutoModel，对应的tensorflow自动类

        -   \"pt\"：AutoModel，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   \"model\"：缺省的预训练权重的模型ID

                -   \"pt\": \"distilbert-base-cased\"

                -   \"tf\": \"distilbert-base-cased\"

    -   \"sentiment-analysis\"：情感分析任务

        -   \"impl\"：TextClassificationPipeline，对应的流水线类

        -   \"tf\"：TFAutoModelForSequenceClassification，对应tensorflow自动类

        -   \"pt\"：AutoModelForSequenceClassification，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   \"model\"：缺省的预训练模型权重的模型ID

                -   \"pt\":
                    > \"distilbert-base-cased-finetuned-sst-2-english\"

                -   \"tf\":
                    > \"distilbert-base-cased-finetuned-sst-2-english\"

    -   \"ner\"：NER任务

        -   \"impl\"：TokenClassificationPipeline，对应的流水线类

        -   \"tf\"：TFAutoModelForTokenClassification，对应的tensorflow自动类

        -   \"pt\"：AutoModelForTokenClassification，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   \"model\"：缺省的预训练权重的模型ID

                -   \"pt\":\"dbmdz/bert-large-cased-finetuned-conll03-english\"

                -   \"tf\":\"dbmdz/bert-large-cased-finetuned-conll03-english\"

    -   \"question-answering\"：智能问答任务

        -   \"impl\"：QuestionAnsweringPipeline，对应的流水线类

        -   \"tf\"：TFAutoModelForQuestionAnswering，对应的tensorflow自动类

        -   \"pt\"：AutoModelForQuestionAnswering，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   \"model\"：缺省的预训练权重的模型ID

                -   \"pt\": \"distilbert-base-cased-distilled-squad\"

                -   \"tf\": \"distilbert-base-cased-distilled-squad\"

    -   \"table-question-answering\"：表格问答任务

        -   \"impl\"：TableQuestionAnsweringPipeline，对应的流水线类

        -   \"tf\"：TFAutoModelForTableQuestionAnswering，对应的tensorflow自动类

        -   \"pt\"：AutoModelForTableQuestionAnswering，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   \"model\"：缺省的预训练权重的模型ID

                -   \"pt\": \"google/tapas-base-fintuned-wtq\"

                -   \"tf\": \"google/tapas-base-fintuned-wtq\"

                -   \"tokenizer\": \"google/tapas-base-fintuned-wtq\"

    -   \"fill-mask\"：填空任务（即语言模型）

        -   \"impl\"：FillMaskPipeline，对应的流水线类

        -   \"tf\"：TFAutoModelForMaskedLM，对应的tensorflow自动类

        -   \"pt\"：AutoModelForMaskedLM，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   \"model\"：缺省的预训练权重的模型ID

                -   \"pt\": \"distilroberta-base\"

                -   \"tf\": \"distilroberta-base\"

    -   \"summarization\"：文本摘要任务

        -   \"impl\"：SummarizationPipeline，对应的流水线类

        -   \"tf\"：TFAutoModelForSeq2SeqLM，对应的tensorflow自动类

        -   \"pt\"：AutoModelForSeq2SeqLM，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   \"model\"：缺省的预训练权重的模型ID

                -   \"pt\": \"sshleifer/distilbart-cnn-12-6\"

                -   \"tf\": \"t5-small\"

    -   \"translation\"：翻译任务

        -   \"impl\"：TranslationPipeline，对应的流水线类

        -   \"tf\"：TFAutoModelForSeq2SeqLM，对应的tensorflow自动类

        -   \"pt\"：AutoModelForSeq2SeqLM，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   (\"en\",\"fr\")：\"model\"：英文-\>法文缺省的预训练权重的模型ID

                -   \"pt\": \"t5-base\"

                -   \"tf\": \"t5-base\"

            -   (\"en\",\"de\")：\"model\"：英文-\>德文缺省的预训练权重的模型ID

                -   \"pt\": \"t5-base\"

                -   \"tf\": \"t5-base\"

            -   (\"en\",\"ro\")：\"model\"：英文-\>罗马尼亚文缺省的预训练权重的模型ID

                -   \"pt\": \"t5-base\"

                -   \"tf\": \"t5-base\"

    -   \"text2text-generation\"：文本到文本任务

        -   \"impl\"：Text2TextGenerationPipeline，对应的流水线类

        -   \"tf\"：TFAutoModelForSeq2SeqLM，对应的tensorflow自动类

        -   \"pt\"：AutoModelForSeq2SeqLM，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   \"model\"：缺省的预训练权重的模型ID

                -   \"pt\": \"t5-base\"

                -   \"tf\": \"t5-base\"

    -   \"text-generation\"：文本生成任务

        -   \"impl\"：TextGenerationPipeline，对应的流水线类

        -   \"tf\"：TFAutoModelForCasualLM，对应的tensorflow自动类

        -   \"pt\"：AutoModelForCasualLM，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   \"model\"：缺省的预训练权重的模型ID

                -   \"pt\": \"gpt2\"

                -   \"tf\": \"gpt2\"

    -   \"zero-shot-classification：返回ZeroShotClassificationPipeline

        -   \"impl\"：ZeroShotClassificationPipeline，对应的流水线类

        -   \"tf\"：TFAutoModelForSequenceClassification，对应tensorflow自动类

        -   \"pt\"：AutoModelForSequenceClassification，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   \"model\"：缺省的模型预训练权重的模型ID

                -   \"pt\": \"facebook/bart-large-mnli\"

                -   \"tf\": \"roberta-large-mnli\"

            -   \"config\"：缺省的配置预训练权重的模型ID

                -   \"pt\": \"facebook/bart-large-mnli\"

                -   \"tf\": \"roberta-large-mnli\"

            -   \"tokenizer\"：缺省的分词器预训练权重的模型ID

                -   \"pt\": \"facebook/bart-large-mnli\"

                -   \"tf\": \"roberta-large-mnli\"

    -   \"conversation\"：对话任务

        -   \"impl\"：ConversationPipeline，对应的流水线类

        -   \"tf\"：TFAutoModelForCasualLM，对应的tensorflow自动类

        -   \"pt\"：AutoModelForCasualLM，对应的Pytorch自动类

        -   \"default\"：缺省参数

            -   \"model\"：缺省的预训练权重的模型ID

                -   \"pt\": \"microsoft/DialoGPT-medium\"

                -   \"tf\": \"microsoft/DialoGPT-medium\"

-   pipelines.\_\_init\_\_.check\_task()：辅助函数，通过SUPPORTED\_TASK映射，根据任务名称，返回具体任务相关的结构

-   pipelines.base.get\_default\_model()：辅助函数，根据check\_task返回的任务相关结构，以及框架，返回模型Model

    -   targeted\_task：任务相关的结构，来自check\_task()

    -   framework：框架，tf或是pt

    -   返回：字符串

-   pipelines.\_\_init\_\_.pipeline()：根据任务返回Pipeline

    -   参数：

        -   task：任务，str类型

        -   model：流水线里的模型

        -   config：配置，类型为PreTrainedConfig或者字符串(模型id）

        -   tokenizer：分词器，类型为PreTrainedTokenizer或者字符串(模型id）

        -   framework：框架，"tf"表示tensorflow，"pt"表示Pytorch

        -   revision：版本

        -   use\_fast：是否使用快速分词器（fast tokenizer）

        -   返回：Pipeline

    -   流程：

        -   调用check\_task()，根据任务名得到SUPPORTED\_TASK中对应的任务结构，包括任务流水线task\_class、自动模型model\_class、缺省的预训练权重model

        -   通过AutoTokenizer和AutoConfig获得tokenizer和config

        -   根据model\_class和model获得加载了权重的模型model

        -   将tokenizer、config、model作为参数，传给任务流水线task\_class，得到完整的流水线

##### 流水线的类

-   pipelines.base.Pipeline：Pipeline类是所有流水线类的基类

    -   task：任务，str类型

    -   model：模型

    -   tokenizer：分词器，PretrainedTokenizer

    -   modelcard：

    -   framework：框架，"tf"（Tensorflow）或是"pt"（Pytorch）

    -   device：设备

    -   binary\_output：

    -   \_\_init\_\_()：构造函数，

    -   \_parse\_and\_tokenize()：调用成员变量tokenizer，返回分词结果

    -   \_forward()：调用成员变量model，返回结果

    -   \_\_call\_\_()：先调用\_parse\_and\_token()分词，再调用\_forward()进行模型运算

    -   save\_pretrained()：保存Model和Tokenizer到指定位置

        -   save\_directory()：保存的位置

    -   check\_model\_type()：工具函数，检查Pipeline是否支持当前的Model

        -   supported\_models：传入的（当前Pipeline）支持的Model列表

-   pipelines.conversational.Conversation：工具类，包含了一个对话及其历史

-   pipelines.conversational.ConversationalPipeline：用于对话的流水线，Pipeline的子类

    -   \_\_init\_\_()：构造函数，先调用super.\_\_init\_\_()，再调用Pipeline的成员函数check\_model\_type()检查模型是否合法

    -   \_\_call\_\_()：调用父类的\_\_call\_\_()，再对输出结果进行格式上的加工并返回

    -   \_get\_history(conversation)：

    -   \_parse\_and\_tokenize()：

    -   \_clean\_padding\_history()：

    -   \_concat\_inputs\_history()：

-   pipelines.feature\_extraction.FeatureExtractionPipeline：用于特征提取的流水线，Pipeline的子类

    -   \_\_init\_\_()：调用super.\_\_init\_\_()

    -   \_\_call\_\_()：将super.\_\_call\_\_()的输出转换为list返回

-   pipelines.fill\_mask.FillMaskPipeline：用于填空的流水线，Pipeline的子类

    -   \_\_init\_\_()：调用super.\_\_init\_\_()，初始化成员变量top\_k

    -   \_\_call\_\_()：将super.\_\_call\_\_()的输出转换为list返回

    -   ensure\_exactly\_one\_mask\_token()：工具函数，确认输入只包含一个\[MASK\]标记

-   pipelines.question\_answering.QuestionAnsweringArgumentHandler：

-   pipelines.question\_answering.QuestionAnsweringPipeline：

-   pipelines.table\_question\_answering.TableQuestionAnsweringArgumentHandler：

-   pipelines.table\_question\_answering.TableQuestionAnsweringPipeline：

-   pipelines.text\_classification.TextClassificationPipeline：用于文本分类的流水线，Pipeline的子类

    -   \_\_init\_\_()：构造函数，先调用super.\_\_init\_\_()，再调用Pipeline的成员函数check\_model\_type()检查模型是否合法

    -   \_\_call\_\_()：调用父类的\_\_call\_\_()，再对输出结果进行格式上的加工并返回

<!-- -->

-   pipelines.text\_generation.TextGenerationPipeline：

-   pipelines.text2text\_generation.Text2TextGenerationPipeline：

-   pipelines.text2text\_generation.SummariazationPipeline：

-   pipelines.text2text\_generation.TranslationPipeline：

<!-- -->

-   pipelines.token\_classification.TokenClassificationArgumentHandler：

-   pipelines.token\_classification.TokenClassificationPipeline：

-   pipelines.zero\_shot\_classification.ZeroShotClassificationArgumentHandler：

-   pipelines.zero\_shot\_classification.ZeroShotClassificationPipeline：

#### models（模型）

##### Bert系列

-   models.bert.configuration\_bert.BertConfig：PreTrainedConfig的子类，描述Bert类型网络模型的超参数等配置

    -   vocab\_size：词汇表长度，缺省为30522

    -   hidden\_size：隐向量的大小（即H值，transformer的宽），缺省为768

    -   num\_hidden\_layers：Transformer结构的层数，缺省为12

    -   num\_attention\_heads：attention头的数量，缺省为12

    -   hidden\_act：激活函数，缺省为gelu

    -   intermediate\_size：缺省为3072

    -   hidden\_dropout\_prob：（嵌入、encoder和pooler）中FC层的dropout比例，缺省为0.1

    -   attention\_probs\_dropout\_prob：

    -   max\_position\_embeddings：最大文本序列长度（词的数量），缺省为512

    -   type\_vocab\_size：token\_type\_ids（即表示前句/后句的分割嵌入种类）的数量，缺省为2

    -   initializer\_range：权重初始化时的标准差

    -   layer\_norm\_eps：层Normalization时的epsilon值（ε）

    -   gradient\_checkpointing：

    -   position\_embedding\_type：位置编码的类型（参考论文《Self-Attention
        > with Relative Position Representations》和《Improve
        > Transformers Models with better relative position
        > embeddings》）

    -   use\_cache：decoder中是否使用encoder传来的key/value

-   models.bert.modeling\_bert.BertEmbeddings：

    -   word\_embeddings：词嵌入

    -   position\_embeddings：位置嵌入

    -   token\_type\_embeddings：即bert中的分割嵌入

    -   LayerNorm：

    -   dropout：

    -   forward()：将三个嵌入相加，再经过LayerNorm和dropout层，返回

-   models.bert.modeling\_bert.BertSelfAttention：Attention层

    -   \_\_init\_\_()：构造函数，参数为config

    -   num\_attetion\_heads：注意力头的个数，来自config中同名变量

    -   attention\_head\_size：单个注意力头的大小，来自config.hidden\_size除以注意力头个数

    -   all\_head\_size：等于config.hidden\_size

    -   query：描述Query矩阵的FC层，输入输出大小都等于config.hidden\_size

    -   key：描述Key矩阵的FC层，大小同上

    -   value：描述Value矩阵的FC层，大小同上

    -   dropout：dropout层，比例为config.attention\_probs\_dropout\_prob

    -   position\_embedding\_type：来自config中同名变量

    -   is\_decoder：是否为decoder（否则为encoder），来自config.is\_decoder

    -   transpose\_for\_scores()：改变序列，

    -   forward()：前向推理函数

        -   hiddent\_state：输入的隐向量，唯一的必填参数

        -   attention\_mask：

        -   head\_mask：

        -   encoder\_hidden\_state：从encoder传来的隐向量，缺省为None

        -   encoder\_attention\_mask：

        -   past\_key\_value：从encoder传来的key和value，缺省为None

##### Albert系列

modeling\_albert.py包括：

-   albert网络的各种构成

-   albert模型基类：AlbertPreTrainedModel

-   albert裸模型：AlbertModel

-   针对各种任务的albert模型（包含一个裸模型）

    Albert网络的构成部分：

<!-- -->

-   models.albert.modeling\_albert.AlbertMLMHead：MLM任务的head，依次包括：

    -   dense：FC层，维度从hidden\_size（隐向量）到embedding\_size（词嵌入）

    -   activation：激活层

    -   LayerNorm：层标准化（layer normalization）

    -   decoder：FC层，维度从embedding\_size（词嵌入）到vocab\_size（词汇表）

    -   bias：decoder的偏置

-   models.albert.modeling\_albert.AlbertSOPHead：SOP任务的head，依次包括：

    -   dropout：dropout层

    -   classifier：FC层，维度从hidden\_size（隐向量）到标签个数

<!-- -->

-   models.albert.modeling\_albert.AlbertEmbeddings、AlbertAttention、AlbertLayer、AlbertLayerGroup、AlbertTransformer：Albert模型的各种组件

    Albert网络模型的基类：

<!-- -->

-   models.albert.modeling\_albert.AlbertPreTrainedModel：ALBERT模型的基类，PreTrainedModel的子类，包含一个初始化权重的函数，并初始化了几个成员变量

    -   config\_class：继承自PreTrainedModel，为AlbertConfig

    -   base\_model\_prefix：类的前缀，继承自PreTrainedModel，为"ablert\_"

    -   \_init\_weights()：初始化权重

        Albert网络的裸模型：

-   models.albert.modeling\_albert.AlbertModel：AlbertPreTrainedModel的子类，"裸"的Albert模型，输出原始的隐藏向量，没有根据任务加任何head。

    针对各种任务的Albert模型：

-   models.albert.modeling\_albert.AlbertForPreTraining：用于预训练的Albert模型，AlbertPreTrainedModel的子类，包括了2个head，即Albert模型的两个任务：一个是MLM（Masked
    Language Modeling）任务，另一个是SOP（Sentence Order
    Prediction）任务

    -   albert：裸Albert网络模型，AlbertModel实例

    -   predictions：用于MLM任务的head，AlbertMLMHead实例

    -   sop\_classifier：用于SOP任务的head，AlbertSOPHead实例

-   models.albert.modeling\_albert.AlbertForMaskedLM：用于MLM任务的Albert模型，AlbertPreTrainedModel的子类，包括一个裸Albert和一个MLM头

    -   albert：裸Albert网络模型，AlbertModel的实例

    -   predictions：语言模型的head，AlbertMLMHead实例

-   models.albert.modeling\_albert.AlbertForSequenceClassification：用于序列分类/回归的Albert模型，AlbertPreTrainedModel的子类，包括一个裸Albert、一个dropout和一个FC层

    -   ablert：裸Albert网络模型，AlbertModel的实例

    -   dropout：dropout层

    -   classifier：FC层，从hidden\_size（隐向量）到标签个数

-   models.albert.modeling\_albert.AlbertForTokenClassification：用于分类token的Albert网络，是AlbertPreTrainedModel的子类，包括一个裸Albert、一个dropout和一个FC层

    -   ablert：裸Albert网络模型，AlbertModel的实例

    -   dropout：dropout层

    -   classifier：FC层，从hidden\_size（隐向量）到标签个数

-   models.albert.modeling\_albert.AlbertForQuestionAnswering：用于问答的Albert模型，也是AlbertPreTrainedModel的子类，包括一个裸Albert和一个FC层。

    -   num\_labels：标签个数

    -   ablert：裸Albert网络模型，AlbertModel的实例

    -   qa\_outputs：FC层，从hidden\_size（隐向量）到标签个数

-   models.albert.modeling\_albert.AlbertForMultipleChoice：用于的Albert模型，也是AlbertPreTrainedModel的子类，

    -   ablert：裸Albert网络模型，AlbertModel的实例

    -   dropout：dropout层

    -   classifier：FC层，从hidden\_size（隐向量）到1

##### Electra系列

modeling\_electra.py包括：

-   electra网络的各种构成

-   electra模型基类

-   electra裸模型

-   针对各种任务的electra模型（包含一个裸模型）

    Electra网络的构成部分：

<!-- -->

-   models.electra.modeling\_electra.ElectraSelfAttention：Electra模型的self-attention结构，复制自modeling\_bert.BertSelfAttention。

-   models.electra.modeling\_electra.ElectraEmbedding、ElectraSelfOutput、ElectraAttention、ElectraIntermediate、ElectraOutput、ElectraEncoder、ElectraDiscriminatorPredictions、ElectraGeneratorPredictions：各种Electra模型的组件

    Electra模型的基类：

-   models.electra.modeling\_electra.ElectraPreTrainedModel：Electra模型的基类，PreTrainedModel的子类，包含一个初始化权重的函数，并初始化了几个成员变量

    -   config\_class：继承自PreTrainedModel，为ElectraConfig

    -   base\_model\_prefix：类的前缀，继承自PreTrainedModel，为"electra"

    -   \_init\_weights()：初始化权重

        Electra的裸模型：

<!-- -->

-   models.albert.modeling\_albert.ElectraModel：ElectraPreTrainedModel的子类，"裸"的Electra模型，输出原始的隐藏向量，没有根据任务加任何head。

    针对各种任务的Electra模型：

-   models.electra.modeling\_electra.ElectraForSequenceClassification：用于序列分类/回归的Electra模型，ElectraPreTrainedModel的子类，包括一个裸Electra裸模型

#### train（训练器）

训练相关的类都在transformers根目录下，包括：

-   trainer\*.py：训练器

-   training\_args\*.py：训练参数相关

-   trainer.Trainer：训练器，训练和评估的循环，适用Pytorch框架

    -   model：被训练的模型，PreTrainedModel或者torch.nn.Module类型

    -   args：训练参数，TrainingArguments类

    -   data\_collator：

    -   train\_dataset：训练集

    -   eval\_dataset：验证集

    -   tokenizer：

    -   model\_init：

    -   compute\_metrics：

    -   callbacks：

    -   optimizers：

    -   \_\_init\_\_()：

-   trainer\_tf.TFTrainer：训练器，训练和评估的循环，适用tensorflow框架

-   training\_args.TrainingArguments：训练参数，pytorch框架

    -   output\_dir：模型预测和checkpoint的输出位置

-   training\_args\_tf.TFTrainingArguments：训练参数，tensorflow框架

### HanLP

HanLP目前由青岛自然语义公司维护，提供了RESTful和Python本地接口。HanLP分成1.x和2.x两个版本，其中1.x主要是由机器学习算法实现，2.x则使用了深度学习。

官方网站：

[[https://hanlp.hankcs.com/]{.underline}](https://hanlp.hankcs.com/)

官方文档：

[[https://hanlp.hankcs.com/docs/index.html]{.underline}](https://hanlp.hankcs.com/docs/index.html)

**代码**

[[https://github.com/hankcs/HanLP]{.underline}](https://github.com/hankcs/HanLP)

### AllenNLP

基于Pytorch的开源项目

官方网站：

[[https://allennlp.org/]{.underline}](https://allennlp.org/)

**代码**

[[https://github.com/allenai/allennlp]{.underline}](https://github.com/allenai/allennlp)

NN：词的表征
------------

NLP的基础是词，所有NLP任务的第一步，是将输入的文本拆分成Token（词），这个步骤被称为Tokenize（分词），分词这步比较简单，可以是一些很粗糙的逻辑（比如英文中用空格和标点分词，当然实际情况要更复杂）。

分词之后的问题就是**词的表征（word
representation）**，就是如何表示词好输入给下一步的模型。NLP有不少经典论文都是关于词向量的表示。

词通常是用向量来表示，最简单就是One-hot形式，此外还有词袋模型、n-gram、TF-IDF、Word2Vec、动态表示（BERT）等等。

**论文**

[[http://www.iro.umontreal.ca/\~vincentp/Publications/lm\_jmlr.pdf]{.underline}](http://www.iro.umontreal.ca/~vincentp/Publications/lm_jmlr.pdf)

**参考**

语言模型：从n元模型到NNLM

[[https://zhuanlan.zhihu.com/p/43453548]{.underline}](https://zhuanlan.zhihu.com/p/43453548)

史上最全词向量讲解（LSA/word2vec/Glove/FastText/ELMo/BERT）

[[https://zhuanlan.zhihu.com/p/75391062]{.underline}](https://zhuanlan.zhihu.com/p/75391062)

### NNLM（2003）

**A Neural Probabilistic Language Model**，深度学习三巨头之一的Yoshua
Bengio于2003年发表的论文，也是第一篇神经网络的语言模型的论文，在得到语言模型的同时，也产出了副产品------词向量。

参考《[[研究方向:NLP \> 概念 \> 语言模型]{.underline}](\l)》

NNLM基于n-grams，输入为（之前的）n-1个词

文中的语言模型如下，包括三层（或者可以将后两层理解为一层）：

-   第一层是输入层，将输入的n-1个词映射为向量，并首尾拼接得到新的向量

-   第二层是隐藏层，将第一层的输出向量进行矩阵运算，激活函数是tanh

-   第三层是输出层，是个softmax多分类器

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image367.png){width="5.886805555555555in"
height="4.011111111111111in"}

而第一层中的向量即为副产品------词向量。

**论文**

[[http://www.iro.umontreal.ca/\~vincentp/Publications/lm\_jmlr.pdf]{.underline}](http://www.iro.umontreal.ca/~vincentp/Publications/lm_jmlr.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/21240807]{.underline}](https://zhuanlan.zhihu.com/p/21240807)

【语言模型】NNLM(神经网络语言模型)

[[https://zhuanlan.zhihu.com/p/206878986]{.underline}](https://zhuanlan.zhihu.com/p/206878986)

### Word2Vec（201301）

词嵌入（Word Embeddings，或者叫词向量，Word
Vectors）是用于在低维空间（通常是几十到几百）表示原本在高维空间（词的one-hot表示，词的词向量稀疏表示法）。

最早用于表示一个词用的是one-hot标签的向量，即每个向量维度等同于词库中词的数量，向量中该词对应的元素为1，其余为0。后来逐渐从这种原始的词向量稀疏表示法过渡到现在的低维空间中的密集表示。用稀疏表示法在解决实际问题时经常会遇到**维数灾难**，并且语义信息无法表示，无法揭示word之间的潜在联系。而采用低维空间表示法，不但解决了维数灾难问题，并且挖掘了word之间的关联属性，从而提高了向量语义上的准确度。

在多年对各种NLP任务的研究中发现，词向量的生成作为很多NLP任务的第一步，可以被独立出来，在一个模型中被单独学习生成，并且被应用到不同的NLP任务中。这篇论文就将注意力集中在如何通过简单的模型来学习词向量。

在 NLP 中，把 x 看做一个句子里的一个词语，y
是这个词语的上下文词语，那么这里的 f，便是 NLP
中经常出现的『语言模型』（language model），这个模型的目的，就是判断
(x,y)
这个样本，是否符合自然语言的法则，更通俗点说就是：词语x和词语y放在一起，是不是人话。

Word2vec 正是来源于这个思想，但它的最终目的，不是要把 f
训练得多么完美，而是只关心模型训练完后的副产物------模型参数（这里特指神经网络的权重），并将这些参数，作为输入
x 的某种向量化的表示，这个向量便叫做------词向量

模型训练的复杂度为 O=
E\*T\*Q，其中O为复杂度，E为epoch数量，T为训练集中的单词数量，Q则是每个模型所定义的。

文中提出了两种模型，第一种称为**CBOW（Continuous Bag-of-Word
Model）**，连续词袋模型。第二种称为**Continuous Skip-gram**模型。

-   如果是用一个词语作为输入，来预测它周围的上下文，那这个模型叫做『Skip-gram
    模型』

-   而如果是拿一个词语的上下文作为输入，来预测这个词语本身，则是 『CBOW
    模型』

**CBOW模型**

基于语料库，CBOW的任务是通过中心词周围的词来预测中心词（及其概率），比如对于"今天下午我去钓鱼"这句话，如果设"我"是中心词，则通过今天、下午、去、钓鱼来推测出中心词。之所以叫做Bag，表明不考虑输入的这些中心词周围词的顺序。

而Skip-gram的任务正好相反，输入是某个中心词，而输出是各个周围词（及其概率）。

两种模型的结构图如下，左侧为CBOW，右侧为Skip-gram，输入和输出的w均为词向量（在实际的代码中每个词有两个词向量，分别作为中心词和周围词时使用）：

![2019-12-13
14-35-59屏幕截图](/home/jimzeus/outputs/AANN/images/media/image368.png){width="5.754166666666666in"
height="3.9180555555555556in"}

以CBOW为例解释一下模型是如何生成词嵌入的。图左为CBOW，可以看到输出为中心词w(t)，而输入为w(t)之前的词w(t-1)和w(t-2)，以及之后的词w(t+1)和w(t+2)。比如"今天早上我吃的玉米"，w(t)是"我"，则w(t-2)是"今天"，w(t-1)是"早上"，w(t+1)是"吃的"，w(t+2)是"玉米"。

1.  输入层：w的格式就是词的one-hot向量，其维度为V（1\*V矩阵，即vocabulary，词库中词的数量，通常V很大，能到10万级别）。CBOW的输入为N（图中N=4）个one-hot向量。

2.  projection层：中间的projection层有权重矩阵W（大小为V\*D），projection层生成D维向量，D即为词嵌入的维度（通常为几十到几千）。N个one-hot向量分别和W做点积，取生成的N个D维向量的平均值作为输出。由于one-hot向量的稀疏性，每个one-hot向量仅和W中的一行做运算，其计算复杂度为D，projection层总的计算复杂度为N\*D。

3.  输出层：输出层有权重矩阵W'（大小为D\*V），生成V维向量（即对输出词的预测），激活函数为softmax，因此这V维向量的和为1，与ground
    truth的one-hot做对比，误差越小越好。其计算复杂度为D\*log2V，（为什么不是D\*V下文解释）。

因此CBOW模型总的计算复杂度Q为：Q = N × D + D × log2V。

下图描述了整个模型：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image369.png){width="4.006944444444445in"
height="3.9409722222222223in"}

训练结束之后，我们需要的并不是这个模型，而是其中的权重矩阵W，每个词的one-hot向量和W点积之后得到的D维向量（即W中对应该词的那一行）即是该词的D维词嵌入。

但在不同的解读文章中，对CBOW模型（以及Skip-gram模型）有不同的解读，比如在Amazon的《动手学习深度学习》中：

[[https://zh.d2l.ai/chapter\_natural-language-processing/word2vec.html]{.underline}](https://zh.d2l.ai/chapter_natural-language-processing/word2vec.html)

在这些论述中，每个词都有两个词嵌入，分别对应其作为中心词和背景词时，如果对比上文中的模型，projection层的权重矩阵（V\*D大小）的每一行（D维）可以被视为背景词嵌入。

而通过其中的计算方式可知，中心词嵌入无法对应输出层的权重矩阵，即：

-   上文中V维输出的计算方式是**背景词嵌入的平均值**（D维）与**D\*V矩阵**（输出层的权重）点积生成的。

-   而d2l文中的V维输出是**背景词嵌入的平均值**（D维）分别与所有（V个）中心词嵌入（D维）点积得出的V个标量。

在对Word2Vec论文质疑的文章中：

[[https://zhuanlan.zhihu.com/p/68401048]{.underline}](https://zhuanlan.zhihu.com/p/68401048)

我们可以看到，也许这就是word2vec论文和代码的区别

**Skip-gram模型**

对于Skip-gram模型来说，整个过程与CBOW类似，区别仅在于，输入的中心词one-hot词向量只有一个，而输出的周围词有C个，计算复杂度为：Q
= C × (D + D × log2V)

**Hierarchical Softmax**

从输入层到projection层是个V维向量（one-hot向量）到D维向量（N个词嵌入的均值）的矩阵乘法，但考虑到V维向量是one-hot向量，其计算复杂度只有N\*D，而从projection层到输出层是个D维向量到V维向量的矩阵乘法，正常来讲其复杂性为D\*V，因为V很大，因此这部分消耗了绝大部分算力。

因此，利用**霍夫曼树**的Hierarchical
Softmax作为激活函数（从而实际上形成了若干层隐藏层），使得通过若干次（log~2~V次）二元判断可以得到最后的V维向量，从而把计算复杂度降低为D\*log~2~V。

[[https://www.cnblogs.com/pinard/p/7243513.html]{.underline}](https://www.cnblogs.com/pinard/p/7243513.html)

**Negtive Sampling**

负采样（Negative
Sampling）是论文中用于降低计算量的另外一个手段。对于训练语言模型来说，softmax层非常难算，由于要预测的是当前位置是哪个词，那么这个类别数就等同于词典规模，因此动辄几万几十万的类别数对算力有很大的消耗。

负采样的思想是，不直接让模型从整个词表找最可能的词了，而是直接给定这个词（即正例）和几个随机采样的噪声词（即采样出来的负例），只要模型能从这里面找出正确的词就认为完成目标。

**参考**

[[https://cloud.tencent.com/developer/news/84841]{.underline}](https://cloud.tencent.com/developer/news/84841)

[[https://blog.csdn.net/bitcarmanlee/article/details/82291968]{.underline}](https://blog.csdn.net/bitcarmanlee/article/details/82291968)

秒懂词向量Word2vec的本质

[https://zhuanlan.zhihu.com/p/26306795]{.underline}

**论文**

Efficient Estimation of Word Representations in Vector Space

[[https://arxiv.org/pdf/1301.3781v3.pdf]{.underline}](https://arxiv.org/pdf/1301.3781v3.pdf)

**代码：**

官方代码：[<https://code.google.com/archive/p/word2vec/> ]{.underline}

网友搬运到**代码**[[https://github.com/dav/word2vec]{.underline}](https://github.com/dav/word2vec)

### GloVe(2014)

GloVe这个名字来自Global Vector，类似Word2Vec，也是一个计算词向量的算法。

**共现矩阵（Co-occurrence Probabilities
Matrix）**的元素X~ij~的意义为，在整个语料库中，单词i和单词j共同出现在一个上下文窗口的次数（早期的一种词向量表征工具LSA，就是基于共现矩阵的）。GloVe是通过共现矩阵来计算词向量。

官方连接：

[[https://nlp.stanford.edu/projects/glove/]{.underline}](https://nlp.stanford.edu/projects/glove/)

**论文**

[[https://nlp.stanford.edu/pubs/glove.pdf]{.underline}](https://nlp.stanford.edu/pubs/glove.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/60208480]{.underline}](https://zhuanlan.zhihu.com/p/60208480)

通俗易懂理解------Glove算法原理

[[https://zhuanlan.zhihu.com/p/42073620]{.underline}](https://zhuanlan.zhihu.com/p/42073620)

四步理解Glove

[[https://zhuanlan.zhihu.com/p/79573970]{.underline}](https://zhuanlan.zhihu.com/p/79573970)

GloVe详解

[[https://www.fanyeong.com/2018/02/19/glove-in-detail/]{.underline}](https://www.fanyeong.com/2018/02/19/glove-in-detail/)

### fastText(201607)

在word2vec中，我们并没有直接利用构词学中的信息。无论是在skip-gram模型还是连续词袋模型中，我们都将形态不同的单词用不同的向量来表示。例如，"dog"和"dogs"分别用两个不同的向量表示，而模型中并未直接表达这两个向量之间的关系。鉴于此，fastText提出了**子词嵌入（subword
embedding）**的方法，从而试图将构词信息引入word2vec中的Skip-gram模型。

在fastText中，每个中心词被表示成子词的集合。下面我们用单词"where"作为例子来了解子词是如何产生的。首先，我们在单词的首尾分别添加特殊字符"\<"和"\>"以区分作为前后缀的子词。然后，将单词当成一个由字符构成的序列来提取n元语法。例如，当n=3时，我们得到所有长度为3的子词："\<wh"、"whe"、"her"、"ere"、"re\>"以及特殊子词"\<where\>"。

fastText也用到了基于**霍夫曼树**的**Hierarchical Softmax**。

**论文**

[[https://arxiv.org/pdf/1607.01759.pdf]{.underline}](https://arxiv.org/pdf/1607.01759.pdf)

代码：

[[https://github.com/facebookresearch/fastText]{.underline}](https://github.com/facebookresearch/fastText)

**参考**

[[https://zhuanlan.zhihu.com/p/158043574]{.underline}](https://zhuanlan.zhihu.com/p/158043574)

[[http://yzstr.com/2018/11/30/fasttext/]{.underline}](http://yzstr.com/2018/11/30/fasttext/)

fastText原理及实践

[[https://zhuanlan.zhihu.com/p/32965521]{.underline}](https://zhuanlan.zhihu.com/p/32965521)

### ELMo(201802)

之前的**词向量表示**（Word2Vec，GloVe，fastText等）都是固定的，对多义词的表示无能为力，ELMo的工作是对于多义词提出了一个较好的解决方案。

ELMo中的词-向量转换不再是一个对应关系，而是一个预训练好的模型，输入词，输出词向量。这个模型是个**两个单向双层LSTM**，如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image370.jpeg){width="4.645833333333333in"
height="2.1104166666666666in"}

**参考**

ELMo原理解析及简单上手使用

[[https://zhuanlan.zhihu.com/p/51679783]{.underline}](https://zhuanlan.zhihu.com/p/51679783)

词嵌入：ELMo原理

[[https://zhuanlan.zhihu.com/p/88993965]{.underline}](https://zhuanlan.zhihu.com/p/88993965)

**论文**

Deep Contextualized word Representation

[[https://arxiv.org/pdf/1802.05365.pdf]{.underline}](https://arxiv.org/pdf/1802.05365.pdf)

NN：基于RNN
-----------

### Encoder-Decoder（201406）

《Learning Phrase Representations using RNN Encoder-Decoder for
Statistical Machine Translation》的贡献有2个：

-   提出了RNN Encoder-Decoder，一种NvsM类型RNN的实现。

-   提出了GRU微结构

Encoder-Decoder由两个RNN组成，即通过一个Nvs1的RNN（Encoder）先将输入序列转换为一个向量，而另一个1vsN的RNN（Decoder）则将这个向量转换为一个输出序列。这种模型可以处理输入和输出序列不等长的情况，**机器翻译**等NLP任务基本上使用的都是此类模型。

RNN Encoder-Decoder的结构如下：

![2019-12-20
15-16-18屏幕截图](/home/jimzeus/outputs/AANN/images/media/image371.png){width="3.7618055555555556in"
height="3.910416666666667in"}

其中下面的框是encoder，X是输入句子中的各个词，而上面的框是decoder，Y是输出句子的各个词。

对于encoder来说，每个步骤的隐藏状态表示为：

![2019-12-20
15-30-10屏幕截图](/home/jimzeus/outputs/AANN/images/media/image372.png){width="2.3444444444444446in"
height="0.6354166666666666in"}

对于decoder来说，每个步骤的隐藏状态为：

![2019-12-20 15-30-10屏幕截图
(复件)](/home/jimzeus/outputs/AANN/images/media/image373.png){width="2.452777777777778in"
height="0.49583333333333335in"}

每个输出symbol（词）的概率分布为：

![2019-12-20
16-54-27屏幕截图](/home/jimzeus/outputs/AANN/images/media/image374.png){width="3.5743055555555556in"
height="0.5in"}

Encoder和Decoder这两个RNN会被联合训练，以便使得以下表达式尽量大：

![2019-12-20 16-54-27屏幕截图
(复件)](/home/jimzeus/outputs/AANN/images/media/image375.png){width="2.4479166666666665in"
height="0.7756944444444445in"}

其中 θ为模型参数，(xn, yn) 为训练集的输入和标签。

模型训练完成之后，可以通过2种方式使用：

-   给定一个输入序列，生成一个输出序列

-   给定一对输入序列-输出序列，得到该序列对的可能性得分

GRU这里不做介绍，参考《[[网络构成 \> RNN微结构 \>
GRU单元]{.underline}](\l)》

**论文**

Learning Phrase Representations using RNN Encoder-Decoder for
Statistical Machine Translation

[[https://arxiv.org/pdf/1406.1078]{.underline}](https://arxiv.org/pdf/1406.1078)

### Seq2Seq（201409）

Seq2Seq有时候指代这篇Google的论文《Sequence to Sequence Learning with
Neural Network》，但通常更广义的指代所有这种类型的模型（参看RNN中的N vs
M类型的模型），与Encoder-Decoder同义。

Seq2Seq通常会和Encoder-Decoder被视为同一概念，这两篇论文出现的时间也差不多，和Encoder-Decoder的区别在于：

-   使用LSTM而非GRU作为Encoder和Decoder的单元

-   使用多层（4层）网络而非1层网络

-   倒序输入序列（句子），能大幅提高网络性能

**论文**

Sequence to Sequence Learning with Neural Network

[[https://arxiv.org/pdf/1409.3215]{.underline}](https://arxiv.org/pdf/1409.3215)

### Attention机制（201409）

参考《[[网络构成 \> Attetion机制 \> RNN中的Attention]{.underline}](\l)》

**参考**

[[https://zhuanlan.zhihu.com/p/42724582]{.underline}](https://zhuanlan.zhihu.com/p/42724582)

[[https://blog.csdn.net/qq\_42189083/article/details/89326085]{.underline}](https://blog.csdn.net/qq_42189083/article/details/89326085)

NN：基于Transformer
-------------------

《Attention is all you
need》是Google在2017年发表的论文，文中提出了Transformer单元（Self-Attention架构）。

Transformer的出现，使得NLP模型的通用性大为增加，同时模型的大小及需要的算力也大增，在Transformer的基础上，出现了**GPT系列**和**BERT系列**两个系列的通用预训练NLP网络（这些网络的特点都是使用了Transformer的一部分）。

**BERT**源自Google，之后不同的团队也开发了各种变种（比如ALBERT，RoBERTa等）。这些NLP的预训练网络类似CV中不同任务中的模型的骨干网络（比如ResNet、VGG、MobileNet等），针对各种任务（主要是分类任务）所需要的附加结构简单，通常只是一些分类的Dense
+ Softmax层，在后期Fine-tuning的工作量也更小。

**GPT系列**来自OpenAI团队，GPT系列的思想和BERT不一样，GPT是完全舍弃Fine-Tuning，转而使用一个更大、更通用的预训练网络来完成各种不同的任务。因此GPT的网络也更大，比如GPT-2的参数量是15亿，GPT-3的参数数量更是达到了惊人的1750亿。

**参考**

Transformer结构及其应用详解\--GPT、BERT、MT-DNN、GPT-2

[[https://zhuanlan.zhihu.com/p/69290203]{.underline}](https://zhuanlan.zhihu.com/p/69290203)

推荐一个宝藏博主，让你搞懂Transformer、BERT、GPT！

[[https://zhuanlan.zhihu.com/p/137858220]{.underline}](https://zhuanlan.zhihu.com/p/137858220)

### Transformer(201706)

2017年，Google团队发布了关于NLP的著名论文**《Attention is All You
Need》**，提出了**Self-Attention（自注意力）**的概念，并据此提出了**Transformer**架构，Transformer架构也成为之后NLP模型的基础。之后随着发展，Transformer在其他领域也有大量的应用。

在Self-Attention之前，NLP的任务大都是RNN Seq2Seq +
Attention的方式，传统的基于RNN的Seq2Seq模型难以处理长序列的句子，无法实现并行，并且面临对齐的问题。这类模型的发展基本有以下几个方向：

-   输入方向性：单向 -\> 双向

-   深度：单层 -\> 多层

-   单元类型：基础RNN -\> LSTM/GRU

再然后CNN由计算机视觉也被引入到deep
NLP中，CNN不能直接用于处理变长的序列样本但可以实现并行计算。完全基于CNN的Seq2Seq模型虽然可以并行实现，但非常占内存，很多的trick，大数据量上参数调整并不容易。

而Self-Attention的出现替代了RNN或者CNN，只用Attention处理NLP。这里的"Self"意思是Query、Key、Value皆从同一个Source中得出。

#### 网络结构

-   **整个网络**分成了Encoder部分和Decoder部分

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image376.png){width="4.503472222222222in"
height="2.823611111111111in"}

-   **Encoder部分**由6层Encoder Transformer单元组成，6层Encoder
    Transformer在结构上完全一样，但不共享权重。

-   **Decoder**则由6层Decoder
    Transformer单元组成，他们的结构也一样，不共享权重

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image377.png){width="4.684722222222222in"
height="3.05in"}

其中每个Encoder层被称为一个Encoder Transformer单元，由两个子层组成：

-   Self-Attention层

-   Dense层

    每个Decoder层被称为一个Decoder Transformer单元，由三个子层组成：

-   Self-Attention层

-   Encoder-Decoder Attention层

-   Dense层

#### Self-Attention

**Self
Attention结构**是Transformer的核心，就是句子中的某个词对于本身的所有词做一次Attention。算出每个词对于这个词的权重，然后将这个词表示为所有词的加权和。每一次的Self
Attention操作，就像是为每个词做了一次Convolution操作或Aggregation操作。

输入序列长度为N（比如512，长度不够的通过空白单词**\<pad\>**补足），输入文本的开始和结束用特殊单词**\<s\>**和**\<e\>**表示。

**流程解释一（词向量）**

论文中的词向量维度为512，query、key、value三个向量维度为64，流程如下：

-   首先每个词的词向量X~i~（1×512）都要分别点积三个矩阵W^q^（512×64）,
    W^k^（512×64）, W^v^（512×64），生成每个词自己的query（q~i~，1×64）,
    key（k~i~，1×64）, value（v~i~，1×64）三个向量

-   以序列中某个位置的词为中心进行Self
    Attention时，都是用该词的key向量（k~i~，1×64）与每个词的query向量的转置（q~i~^T^，64×1）做点积（q~i~^T^·k~i~），得到N个score。

-   将该词的所有score（N个）除以8（key向量的维度64的平方根），再通过Softmax归一化出权重（N个），这些权重表示了序列中各个词对当前位置的影响（毫无疑问影响最大的是该词本身）。

-   然后通过这些权重（N个）算出所有词的value（1×64）的加权和，作为这个位置的输出向量Z~i~（1×64）。

**流程解释二（词向量）**

假设输入序列为2个词"Think Machine"，流程如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image378.png){width="4.01875in"
height="3.8194444444444446in"}

**流程解释三（矩阵）**

而整个序列X的计算过程组合起来，则可以用矩阵运算来表示：

-   首先，输入序列X的词嵌入矩阵（N×512，下图显示为2×4）通过W^Q^、W^K^、W^V^（皆为512×64，图示为4×3）分别转换为三个矩阵Q、K、V（皆为N×64，图示为2×3）

![](/home/jimzeus/outputs/AANN/images/media/image379.png){width="1.75in"
height="2.0631944444444446in"}

-   然后Q（N×64，图示为2×3）与转置K（64×N，图示为3×2）做点积，生成表示序列中每个位置对每个位置的影响的score矩阵（N×N，图中未显示，若有应为2×2），再做softmax运算得出每个输入的权重（N×N），乘以V（N×64，图示为2×3），得出表示各个位置的输出的矩阵Z（N×64，图示为2×3）

![](/home/jimzeus/outputs/AANN/images/media/image380.png){width="2.251388888888889in"
height="1.15625in"}

公式为：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image381.png){width="3.209722222222222in"
height="0.9993055555555556in"}

**流程解释四（矩阵）**

下图是台大李宏毅视频中的截图，同样表示了从输入矩阵I（序列×嵌入）到输出矩阵O（序列×嵌入）的整个过程：

![](/home/jimzeus/outputs/AANN/images/media/image382.png){width="4.176388888888889in"
height="2.859027777777778in"}

#### Multi-head结构

Transformer论文中还提出了Multi-head（多头结构）。

Multi-head将数据分别输入到若干个（8个）不同的self-attention结构中，得到8个加权的特征矩阵Z~i~（N×64），之后按列将它们拼成一个大的特征矩阵（N×(64\*8)），经过一层全连接层，即图中的W^o^权重矩阵（512×512）之后得到输出Z（N×512）。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image383.png){width="5.836805555555555in"
height="3.2291666666666665in"}

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image384.png){width="6.368055555555555in"
height="3.5652777777777778in"}

除了第一个encoder的输入是词向量（维度512），其后各个encoder的输入都是前一个encoder的输出（维度为64\*8=512）

#### Position-encoding位置编码

因为模型中不包括Recurrent和CNN，因此输入是位置无关的，也就是说，无论输入数据的顺序如何（输入句子中词的顺序，或者输入图像中各个像素的位置等），结果是类似的。因此论文还提出了**Position
Encoding（位置编码）**概念。

文中Position
Encoding就是直接在输入的词向量（1×512）上加上一个hard-coded的Position
Encoding向量（1×512），从而输出一个带位置信息的嵌入positional
embedding（1×512）。

位置编码公式为：

![](/home/jimzeus/outputs/AANN/images/media/image385.png){width="2.3756944444444446in"
height="0.5145833333333333in"}

结构如下图：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image386.png){width="5.441666666666666in"
height="3.0006944444444446in"}

#### Encoder Transformer单元

整个Encoder包括了6层这样的Encoder Transformer：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image387.png){width="4.111805555555556in"
height="2.982638888888889in"}

每个Encoder Transformer单元都有两个子层组成：

-   Self-Attention层

-   全连接层（Feed Forward）

    这两个子层都有残差（Residual）结构，即图中的Add&Norm，具体是将该子层的输入和输出相加（Add），并做归一化（LayerNorm）：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image388.png){width="3.892361111111111in"
height="3.640972222222222in"}

#### Decoder Transformer

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image389.png){width="4.554166666666666in"
height="2.5881944444444445in"}

Decoder Transformer类似Encoder Transformer，区别在于：

-   **Self-Attention层**和**Dense层**之间多了一层**Encoder-Decoder
    Attention层**

Encoder最上面一层的输出转化为了K和V，被作为Decoder中的Encoder-Decoder
Attention层的输入的一部分，来帮助Decoder聚焦在输入序列的正确位置。也就是说该层的Q用的是下面的Self-Attention层输出的Q，而K和V则来自于Encoder

-   Decoder的Self-Attention层也被成为**Masked
    Attention层**，其输入只允许先于当前位置的词，其后的位置都被mask

Masked Attention层相较于Self-Attention层（Encoder
Transformer中）的区别在于，在进入Softmax之前，与（&）一个掩码矩阵

![](/home/jimzeus/outputs/AANN/images/media/image390.png){width="2.1347222222222224in"
height="1.9083333333333334in"}

公式如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image391.png){width="3.8368055555555554in"
height="0.5104166666666666in"}

#### 最后的Linear层和Softmax层

Decoder的输出是一个向量（N×512），这个向量会经过一个Linear层（全连接层），该层的输出维度很大（N×Vocab\_size），等同于词库中词汇的数量，再经过一个Softmax层，得出对每个词的预测的结果概率（N×Vocab\_size）。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image392.png){width="4.339583333333334in"
height="2.801388888888889in"}

#### 整体结构

整体结构如下：

-   Encoder

    -   Encoder Transformer × 6

        -   Self-Attention层（+Add&Norm）

        -   全连接层（+Add&Norm）

-   Decoder

    -   Decoder Transformer × 6

        -   Masked Self-Attention层（+Add&Norm）

        -   Encoder-Decoder Attention层

        -   全连接层（+Add&Norm）

    -   Linear层 + Softmax

![](/home/jimzeus/outputs/AANN/images/media/image393.png){width="2.5631944444444446in"
height="3.8541666666666665in"}

**论文**

Attention is All You Need

[[https://arxiv.org/pdf/1706.03762.pdf]{.underline}](https://arxiv.org/pdf/1706.03762.pdf)

**参考**

The Illustrated Transformer：

[[http://jalammar.github.io/illustrated-transformer/]{.underline}](http://jalammar.github.io/illustrated-transformer/)

Transformer结构及其应用详解\--GPT、BERT、MT-DNN、GPT-2

[[https://zhuanlan.zhihu.com/p/69290203]{.underline}](https://zhuanlan.zhihu.com/p/69290203)

详解Transformer（Attention is all you need）

[[https://zhuanlan.zhihu.com/p/48508221]{.underline}](https://zhuanlan.zhihu.com/p/48508221)

李宏毅视频：[[https://www.bilibili.com/video/BV15b411g7Wd?p=92]{.underline}](https://www.bilibili.com/video/BV15b411g7Wd?p=92)

[[https://mp.weixin.qq.com/s/RLxWevVWHXgX-UcoxDS70w]{.underline}](https://mp.weixin.qq.com/s/RLxWevVWHXgX-UcoxDS70w)

Weighted Transformer Network for Machine Translation

[[https://arxiv.org/pdf/1711.02132.pdf]{.underline}](https://arxiv.org/pdf/1711.02132.pdf)

代码：

[[https://github.com/Kyubyong/transformer]{.underline}](https://github.com/Kyubyong/transformer)

[[https://github.com/JayParks/transformer]{.underline}](https://github.com/JayParks/transformer)

### 自回归模型（GPT系列）

**GPT**系列是OpenAI团队（一个非盈利的AI组织）开发的，基于Transformer的NLP预训练模型。除了模型尺度和训练数据增加之外，GPT三代之间的结构基本没有什么变化。

GPT和BERT是当前两个最主流的NLP模型类型，其共同点是都基于Transformer，而

GPT使用Decoder Transformer 单元（少一层），BERT使用Encoder
Transformer单元

基于上面的原因，GPT是个生成模型，而BERT则不是。

**参考**

[[https://zhuanlan.zhihu.com/p/228857593]{.underline}](https://zhuanlan.zhihu.com/p/228857593)

#### GPT（201806）

GPT采用**单向Transformer单元**，就是去掉**Encoder-Decoder
Attention层**的**Decoder Transformer**（或者也可以被认为是将**Encoder
Transformer**里的**Self-Attention层**替换为**Decoder
Transfomer**中的**Masked
Self-Attention层**），使得模型只能看得见上文的词。

而训练的过程其实非常的简单，就是将句子n个词的词向量（第一个为特殊单词\<s\>，表示start）加上**Positional
Encoding**后输入到前面提到的Transfromer中，n个输出分别预测该位置的下一个词（\<s\>预测句子中的第一个词，最后一个词的预测结果不用于语言模型的训练）。

由于使用了Masked
Self-Attention，所以每个位置的词都不会"看见"后面的词，也就是预测的时候是看不见"答案"的，保证了模型的合理性，而因此GPT模型每次都是吐出一个单词，而这个单词会被加入下一轮的输入。

![](/home/jimzeus/outputs/AANN/images/media/image394.png){width="5.7659722222222225in"
height="2.926388888888889in"}

**论文**

[[https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language\_understanding\_paper.pdf]{.underline}](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

#### GPT-2（201902）

GPT-2继续沿用了原来GPT中使用的**单向Transformer单元**来建立模型，而这篇文章的目的就是尽可能利用**单向Transformer**的优势，做一些BERT使用的双向Transformer所做不到的事。那就是通过上文生成下文文本。

GPT-2和GPT的结构基本没有区别，主要就是更大的结构和更多的训练数据：

GPT-2的想法就是完全舍弃Fine-Tuning过程，转而使用一个容量更大、无监督训练、更加通用的语言模型来完成各种各样的任务。我们完全不需要去定义这个模型应该做什么任务，因为很多标签所蕴含的信息，就存在于语料当中。

GPT-2中使用了不同大小的网络（维度和深度），例如其中GPT-2
small堆叠了12层单向Transformer单元，词向量的维度为768：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image395.png){width="5.414583333333334in"
height="2.564583333333333in"}

而最大的GPT-2 Extra Large的深度是48层，使用的词向量宽度为1600。

训练后的GPT-2包含两个权值矩阵：词向量矩阵和位置编码矩阵（参考Transformer）

**论文**

[[https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf]{.underline}](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

**参考**

The Illustrated GPT-2

[[http://jalammar.github.io/illustrated-gpt2/]{.underline}](http://jalammar.github.io/illustrated-gpt2/)

完全图解GPT-2：看完这篇就够了（一）

[[https://zhuanlan.zhihu.com/p/79714797]{.underline}](https://zhuanlan.zhihu.com/p/79714797)

**代码**

[[https://github.com/openai/gpt-2]{.underline}](https://github.com/openai/gpt-2)

#### GPT-3（202005）

2020年，OpenAI推出了参数量高达1750亿的GPT-3。其结构跟之前的GPT-2一样，依然是单向Transformer单元的堆叠，主要的区别就还是大小。GPT-3的层数达到了96层。

**参考**

最新最全GPT-3模型网络结构详细解析

[[https://zhuanlan.zhihu.com/p/174782647]{.underline}](https://zhuanlan.zhihu.com/p/174782647)

How GPT3 works - Visualizations and Animations

[[http://jalammar.github.io/how-gpt3-works-visualizations-animations/]{.underline}](http://jalammar.github.io/how-gpt3-works-visualizations-animations/)

**论文**

[[https://arxiv.org/pdf/2005.14165.pdf]{.underline}](https://arxiv.org/pdf/2005.14165.pdf)

**github**

[[https://github.com/openai/gpt-3]{.underline}](https://github.com/openai/gpt-3)

### 自编码模型（BERT系列）

Transformer出来之后，成为了在NLP领域替代LSTM的一大杀器，而原始Transformer（即《Attention
is all you
need》中提到的Encoder-Decoder结构）作为机器翻译这类NLG任务的模型很合适，那么该如何处理NLU这种分类任务呢，Google提出了**BERT**作为解决方法。

**BERT**（**Bidirectional Encoder Representations from
Transformers**），是Google团队推出的基于Transformer的NLP模型，在BERT的基础上，不同的团队推出了各种NLP模型。

BERT系列的模型是预训练模型，对其的使用类似图像领域中的卷积神经网络（比如VGGNet、ResNet等），使用者需要进行几步相对简单且不费算力的操作：

-   下载预训练模型（官方或者第三方已经在大语料库上进行过无监督的预训练）

-   添加全连接层（向量-\>类别）+ Softmax

-   通过对应的语料进行有监督的fine-tuning

#### BERT(201810)

2018年10月，Google发布的论文及预训练模型BERT，成功的在11项NLP任务中取得SOTA的成绩。

BERT基于Transformer，从此开始了基于Transformer的预训练模型（BERT系列和GPT系列）在NLP任务上独占骜头的局面。BERT的特点从其论文的名字上可以看出来（**BERT:
Pre-training of Deep Bidirectional Transformers for Language
Understanding**）：

-   Deep：网络深度比原始的transformer深

-   Bidirectional Transformers：基于**双向Encoder
    Transformer**，这是和使用**单向Transformer**的GPT的区别，并且没有使用到原始Transformer中的Decoder部分

-   Pre-training：BERT是大量数据无监督预训练出的模型，之后不同下游任务根据需要，用supervised数据进行对应的训练（fine-tuning）7

    BERT和GPT以及ELMo的区别如下：

    ![](/home/jimzeus/outputs/AANN/images/media/image396.png){width="5.759027777777778in"
    height="2.3201388888888888in"}

    可以看到：

-   GPT使用的是**单向Transformer Encoder**

-   而ELMo则是将**两个单向的LSTM**的输出做拼接

-   BERT则是直接使用**双向Transformer**

    BERT吸取了之前的几篇论文的经验：

-   Transformer：Tranformer单元

-   ELMo：双向、动态词嵌入

-   ULM-FiT：fine-tuning的流程

-   GPT：采用部分Transformer（BERT用的是Encoder）

    相较于原始的Transformer，BERT有如下改动：

    **网络结构**

    BERT用的是**双向Transformer单元**的堆叠，这里的双向Transformer就是Transformer中的**Encoder
    Transformer单元**，所谓"双向"是指用的是用的是Self-Attetion层，而非类似GPT所用的**单向Transformer单元**中的Masked
    Self-Attention层（只能看到左边的信息）。

    BERT分为两个大小的版本，Base版和Large版本，L（模型层数）、H（向量维度）和A（多头结构的head数量）分别为：

-   Base版本：L=12，H=768，A=12，参数总量是110M

-   Large版本：L=24，H=1024，A=16，参数数量为340M

-   原始Transformer：L=6，H=512，A=8

    **输入向量和输出向量**

    BERT的输入向量是三个向量的和，比原始Transformer多了一个分割嵌入：

-   **WordPiece嵌入**：是指将单词划分成一组有限的**公共子词单元**，能在单词的有效性和字符的灵活性之间取得一个折中的平衡。例如图中'playing'被拆分成了'play'和'ing'。

-   **位置嵌入（Position Embedding）**：对词位置的标示，参考Transformer

-   **分割嵌入（Segment
    Embedding）**：用于区分两个句子，例如B是否是A的下文，对于句子对，第一个句子的特征值为0，第二个的特征值为1

    下图中的每个块都是一个向量（base版的维度为768），红色为输入的词向量，由WordPiece嵌入（黄色）、分割嵌入（绿色）、位置嵌入（灰色）相加而来。

    ![](/home/jimzeus/outputs/AANN/images/media/image397.png){width="5.763194444444444in"
    height="1.7791666666666666in"}

    输入序列的第一个位置为**特殊单词\[CLS\]**，用于给之后的分类（classification）留出位置。第一个输出向量（base版维度也是768）用于分类，通过一个全连接层（输入为该向量，输出为分类类别数）+Softmax就可以得到很好的分类效果，比如下图中的区分垃圾邮件的二分类。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image398.png){width="5.639583333333333in"
height="3.1069444444444443in"}

BERT的预训练任务包括两个：MLM（Masked Language Model）和NLP（Next
Sentence Prediction）

**Masked Language Model**

MLM预训练任务，指的是在训练的时候随机从输入语料上mask掉一些token，然后通过上下文预测该token。注意，模型**只需要预测该token**，而无须重建（计算）整个输入。

这个随机mask的比例是15%，并且并不是每次都mask掉这些单词，而是有80%比例使用一个特殊单词\[mask\]替换该词，有10%的几率不替换，而剩下10%则随机替换为其它任意单词。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image399.png){width="5.785416666666666in"
height="3.8006944444444444in"}

**Next Sentence Prediction**

为了让BERT可以更好的把握两个句子之间的关系，比如问答判断（判断两句是否互为问答），蕴含关系（Entailment，判断两句是否互相蕴含），BERT还使用了另一个无监督预训练任务NSP（Next
Sententce Prediction)。

NSP任务是判断句子B是否是句子A的下文，如果是的话输出"IsNext"，否则输出""NotNext"。

训练数据的生成方式是从平行语料中随机抽取的连续两句话，其中50%保留抽取的两句话，它们符合IsNext关系，另外50%的第二句话是随机从预料中提取的，它们的关系是NotNext的。这个关系保存在图中的第一个输出向量中（类似MLM任务中第一个输出向量表示分类）。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image400.png){width="5.325in"
height="3.6618055555555555in"}

**BERT在具体任务的应用**

对于各种不同类型的任务，BERT需要添加不同的输出层，并加以Fine-tuning，具体的模型如下图所示，其中：

-   左上为输入为句子对，输出为分类的任务，比如MNLI、QQP、QNLI等

-   右上为输入为单个句子，输出为分类的任务，比如SST-2、CoLA

-   左下为输入为句子对，输出为句子的任务，比如SQuAD v1.1

-   右下为输入为单个句子，输出为多个标签的任务，比如CoNLL-2003 NER

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image401.png){width="5.7652777777777775in"
height="3.890972222222222in"}

Fine-tuning并不是唯一使用BERT的方式，就像ELMo，也可以使用BERT来生成上下文相关的词向量，再将这些向量输入到自己的模型。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image402.png){width="5.86875in"
height="3.2597222222222224in"}

问题是该用哪一层的输出来作为词向量最合适？BERT论文中尝试了6种不同的选择：

-   第一层

-   最后一层

-   所有12层之和

-   倒数第二层

-   最后四层之和

-   最后四层的连接（Concatenate）

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image403.png){width="5.559722222222222in"
height="3.3520833333333333in"}

**论文**

BERT:Pre-training of Deep Bidirectional Transformers for Language
Understanding

[[https://arxiv.org/pdf/1810.04805.pdf]{.underline}](https://arxiv.org/pdf/1810.04805.pdf)

**github**

Google官方：

[[https://github.com/google-research/bert]{.underline}](https://github.com/google-research/bert)

哈工大中文版本：

[[https://github.com/ymcui/Chinese-BERT-wwm]{.underline}](https://github.com/ymcui/Chinese-BERT-wwm)

**参考**

Illustrated BERT

[[http://jalammar.github.io/illustrated-bert/]{.underline}](http://jalammar.github.io/illustrated-bert/)

A Visual Gguide to Using BERT for the First Time

[[http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/]{.underline}](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

词向量之BERT

[[https://zhuanlan.zhihu.com/p/48612853]{.underline}](https://zhuanlan.zhihu.com/p/48612853)

【NLP】Google BERT模型原理详解

[[https://zhuanlan.zhihu.com/p/46652512]{.underline}](https://zhuanlan.zhihu.com/p/46652512)

NLP必读：十分钟读懂谷歌BERT模型

[[https://zhuanlan.zhihu.com/p/51413773]{.underline}](https://zhuanlan.zhihu.com/p/51413773)

#### ERNIE（201904）

**论文**

[[https://arxiv.org/pdf/1904.09223.pdf]{.underline}](https://arxiv.org/pdf/1904.09223.pdf)

#### XLNet（201906）

XLNet试图融合**自回归语言模型**（Auto
Regressive，比如GPT）和**自编码语言模型**（Auto
Encoder，比如BERT）的优点。

**论文**

[[https://arxiv.org/pdf/1906.08237.pdf]{.underline}](https://arxiv.org/pdf/1906.08237.pdf)

**参考**

XLNet:运行机制及和Bert的异同比较

[[https://zhuanlan.zhihu.com/p/70257427]{.underline}](https://zhuanlan.zhihu.com/p/70257427)

什么是XLNet，它为什么比BERT效果好？

[[https://zhuanlan.zhihu.com/p/107350079]{.underline}](https://zhuanlan.zhihu.com/p/107350079)

**代码**

官方：[[https://github.com/zihangdai/xlnet]{.underline}](https://github.com/zihangdai/xlnet)

哈工大中文版本：[[https://github.com/ymcui/Chinese-XLNet]{.underline}](https://github.com/ymcui/Chinese-XLNet)

#### ERNIE2.0（201907）

**论文**

[[https://arxiv.org/pdf/1907.12412.pdf]{.underline}](https://arxiv.org/pdf/1907.12412.pdf)

**代码**

[[https://github.com/PaddlePaddle/ERNIE]{.underline}](https://github.com/PaddlePaddle/ERNIE)

**参考**

[[https://www.ramlinbird.com/2019/08/06/ernie%E5%8F%8Aernie-2-0%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/]{.underline}](https://www.ramlinbird.com/2019/08/06/ernie%E5%8F%8Aernie-2-0%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/)

#### RoBERTa（201907）

从模型上来说，RoBERTa基本没有什么太大创新，主要是在BERT基础上做了几点调整：
1）训练时间更长，batch size更大，训练数据更多；

2.  移除了next predict loss；

3.  训练序列更长；

4.  动态调整Masking机制。

5.  Byte level BPE

    RoBERTa is trained with dynamic masking (Section 4.1), FULL -
    SENTENCES without NSP loss (Section 4.2), large mini-batches
    (Section 4.3) and a larger byte-level BPE (Section 4.4).

**参考**

[[https://zhuanlan.zhihu.com/p/103205929]{.underline}](https://zhuanlan.zhihu.com/p/103205929)

**论文**

[[https://arxiv.org/pdf/1907.11692.pdf]{.underline}](https://arxiv.org/pdf/1907.11692.pdf)

**代码**

RoBERTa中文版本：[[https://github.com/brightmart/roberta\_zh]{.underline}](https://github.com/brightmart/roberta_zh)

#### ALBERT（201909）

ALBERT也是Google团队提出的，是BERT的一个变体。

ALBERT主要有以下几点贡献：

-   Factorized embedding parameterization

-   Cross-layer parameter Sharing

-   Sentence-order prediction

    **Factorized embedding parameterization**

    **参数分解**，在BERT、XLNet、RoBERTa等模型中，由于模型结构的限制，WordePiece
    embedding的大小**E**总是与隐层大小**H**相同，即**E =
    H**。从建模的角度考虑，词嵌入学习的是单词与上下文无关的表示，而隐层则是学习与上下文相关的表示。显然后者更加复杂，需要更多的参数，也就是说模型应当增大隐层大小**H**，或者说满足**H \>\>
    E**。但实际上词汇表的大小**V**通常非常大，如果**E =
    H**的话，增加隐层大小H后将会使embedding
    matrix的维度**V×E**非常巨大。

    因此本文想要打破**E**与**H**之间的绑定关系，从而减小模型的参数量，同时提升模型表现。具体做法是将embedding
    matrix分解为两个大小分别为**V×E**和**E×H**矩阵，也就是说先将单词投影到一个低维的embedding空间**E**，再将其投影到高维的隐藏空间**H**。这使得embedding
    matrix的维度从**O(V×H）**减小到**O(V×E+E×H）**。当**H \>\>
    E**时，参数量减少非常明显。在实现时，随机初始化V×E和E×H的矩阵，计算某个单词的表示需用一个单词的one-hot向量乘以**V×E**维的矩阵（也就是lookup），再用得到的结果乘**E×H**维的矩阵即可。两个矩阵的参数通过模型学习。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image404.jpeg){width="4.774305555555555in"
height="2.3208333333333333in"}

**Cross-Layer Parameter Sharing**

**跨层参数共享**，本文提出的另一个减少参数量的方法就是层之间的参数共享，即多个层使用相同的参数。参数共享有三种方式：只共享feed-forward
network的参数、只共享attention的参数、共享全部参数。ALBERT默认是共享全部参数的。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image405.jpeg){width="5.553472222222222in"
height="1.7430555555555556in"}

**Sentence-order Prediction**

在先前的研究中，已经证明NSP是并不是一个合适的预训练任务。因此本文提出了**SOP**（**句子顺序预测**）任务来取代NSP。其正例与NSP相同，都是两句连续的句子对，但负例是通过选择一篇文档中的两个连续的句子并将它们的顺序交换构造的。这样两个句子就会有相同的话题，模型学习到的就更多是句子间的连贯性。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image406.jpeg){width="5.585416666666666in"
height="2.1333333333333333in"}

**论文**

[[https://arxiv.org/pdf/1909.11942.pdf]{.underline}](https://arxiv.org/pdf/1909.11942.pdf)

**参考**

【论文阅读】ALBERT

[[https://zhuanlan.zhihu.com/p/87562926]{.underline}](https://zhuanlan.zhihu.com/p/87562926)

BERT的youxiu变体：ALBERT论文图解介绍

[[https://zhuanlan.zhihu.com/p/142416395]{.underline}](https://zhuanlan.zhihu.com/p/142416395)

**代码**

中文版：[[https://github.com/brightmart/albert\_zh]{.underline}](https://github.com/brightmart/albert_zh)

#### ELECTRA（202003）

ELECTRA是Google和斯坦福共同提出的，本质上还是一个BERT模型。

其最主要的贡献是：

-   提出了新的预训练框架，由**Generator（生成器）**和**Discriminator（判别器）**组成，有点类似GAN

-   及其对应的新预训练任务**RTD（Replaced Token
    Detection）**，判断当前token是否被语言模型替换过

ELECTRA的预训练框架由两个部分构成，一个BERT-small作为**Generator**运行**MLM任务**，而另外一个BERT模型（即ELECTRA模型）作为**Discriminator**运行**RTD任务**，即负责判别这些生成出来的token有哪些被替换掉了。

![](/home/jimzeus/outputs/AANN/images/media/image407.png){width="5.767361111111111in"
height="2.6319444444444446in"}

上述结构有个问题，输入句子经过生成器，输出改写过的句子，因为句子的字词是离散的，所以梯度在这里就断了，判别器的梯度无法传给生成器，于是生成器的训练目标还是MLM（作者在后文也验证了这种方法更好），判别器的目标是序列标注（判断每个token是真是假），两者同时训练，但**判别器的梯度不会传给生成器**，目标函数如下：

![](/home/jimzeus/outputs/AANN/images/media/image408.png){width="2.263888888888889in"
height="0.37569444444444444in"}

因为判别器的任务相对来说容易些，RTD loss相对MLM
loss会很小，因此加上一个系数，作者训练时使用了50。

另外要注意的一点是，在**优化判别器时计算了所有token上的loss，而以往计算BERT的MLM
loss时会忽略没被mask的token**。作者在后来的实验中也验证了在所有token上进行loss计算会提升效率和效果。

事实上，ELECTRA使用的Generator-Discriminator架构与GAN还是有不少差别，作者列出了如下几点：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image409.jpeg){width="5.572222222222222in"
height="1.5625in"}

**论文**

[[https://arxiv.org/pdf/2003.10555.pdf]{.underline}](https://arxiv.org/pdf/2003.10555.pdf)

**代码**

官方：[[https://github.com/google-research/electra]{.underline}](https://github.com/google-research/electra)

哈工大的中文版ELECTRA：[[https://github.com/ymcui/Chinese-ELECTRA]{.underline}](https://github.com/ymcui/Chinese-ELECTRA)

**参考**

ELECTRA：超越BERT，19年最佳NLP预训练模型

[[https://zhuanlan.zhihu.com/p/89763176]{.underline}](https://zhuanlan.zhihu.com/p/89763176)

超越BERT模型的ELECTRA代码解读

[[https://zhuanlan.zhihu.com/p/139898040]{.underline}](https://zhuanlan.zhihu.com/p/139898040)

#### DeBERTa（202006）

**论文**

[[https://arxiv.org/pdf/2006.03654.pdf]{.underline}](https://arxiv.org/pdf/2006.03654.pdf)

**代码**

[[https://github.com/microsoft/DeBERTa]{.underline}](https://github.com/microsoft/DeBERTa)

### T5（201910）

**论文**

[[https://arxiv.org/pdf/1910.10683.pdf]{.underline}](https://arxiv.org/pdf/1910.10683.pdf)

**代码**

[[https://github.com/google-research/text-to-text-transfer-transformer]{.underline}](https://github.com/google-research/text-to-text-transfer-transformer)

研究方向：时间序列（TS）
========================

**时间序列**（**Time
Series**）是一组按照时间发生先后顺序进行排列的数据点序列。通常一组时间序列的时间间隔为一恒定值（如1秒，5分钟，12小时，7天，1年），因此时间序列可以作为离散时间数据进行分析处理。

早期的时序分析通常都是直观的数据比较或绘图观测，寻找序列中蕴涵的发展规律，这种分析方法称为**描述性时序分析**。20世纪20年代开始，学术界利用数理统计学原理来分析时间序列。研究的重心从总结表面现象（描述性时序分析）转移到分析序列值内在的相关关系上（**统计时序分析**），由此开辟来一门应用统计学学科------时间序列分析。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image410.jpeg){width="5.170833333333333in"
height="1.363888888888889in"}

在使用神经网络之前，时间序列分析使用了一系列传统模型，比如**AR**，**MA**，**ARMA**以及结合了这几个的**ARIMA模型**，而在此基础之上，**ARCH模型**则解决了传统计量经济学中**方差恒定**这个不符合实际的假设（也是2003年的诺奖的成果之一）。

总体来说，时序分析的传统方法可以分为**时域分析**和**频域分析**。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image411.jpeg){width="4.5569444444444445in"
height="3.35in"}

时域分析包括：**ACF、XCF、ARMA、ARIM、ARCH**等

频域分析包括：**傅立叶变换**、**小波变换**等

目前传统方法中效果最佳的方式是**COTE**（35个分类器的集合）及其后续**HIVE-COTE**（37个分类器的集合），但是为了实现高精度，HIVE-COTE的计算量变得非常之大。

时序预测算法按照其实现原理可以分为：

-   传统统计学

-   机器学习（非深度学习）

-   深度学习

按照预测步长来分，可以分为：

-   单步预测：一次预测一个时间步骤

-   多步预测：一次预测多个时间步骤

按照输入变量区分，可分为：

-   自回归预测：只以预测数据项和时间作为输入

-   协变量预测：也接受其他相关变量（协变量）作为输入

按输出结果分：

-   单点预测：输出结果为一个具体的值

-   概率预测：输出结果为一个具体的概率分布

按目标个数区分，可分为：

-   一元预测：预测目标只有一个

-   多元预测：预测目标有多个

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image412.png){width="5.826388888888889in"
height="1.3791666666666667in"}

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image413.png){width="5.6194444444444445in"
height="2.5104166666666665in"}

**参考**

使用DeepAR进行时间序列预测

[[https://amazonaws-china.com/cn/blogs/china/time-series-prediction-with-deep/]{.underline}](https://amazonaws-china.com/cn/blogs/china/time-series-prediction-with-deep/)

The Complete Guide to Time Series Analysis and Forecasting

[[https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775]{.underline}](https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775)

传统方法：

[[https://zhuanlan.zhihu.com/p/83511434]{.underline}](https://zhuanlan.zhihu.com/p/83511434)

《基于深度学习的时间序列分类》：

[[https://blog.csdn.net/qq\_32796253/article/details/88414656]{.underline}](https://blog.csdn.net/qq_32796253/article/details/88414656)

《深度学习在时间序列分类中的应用》：

[[https://zhuanlan.zhihu.com/p/83130649]{.underline}](https://zhuanlan.zhihu.com/p/83130649)

综述**论文**《Deep learning for time series classification: a review》

[[https://arxiv.org/pdf/1809.04356.pdf]{.underline}](https://arxiv.org/pdf/1809.04356.pdf)

翻译：《基于深度学习时间序列分类研究综述》

[[https://blog.csdn.net/qq\_32796253/article/details/88538231]{.underline}](https://blog.csdn.net/qq_32796253/article/details/88538231)

综述**论文**《Deep Neural Network Ensembles for Time Series
Classification》

[[https://arxiv.org/pdf/1903.06602.pdf]{.underline}](https://arxiv.org/pdf/1903.06602.pdf)

翻译：《用于时间序列分类的集成深度神经网络》

[[https://zhuanlan.zhihu.com/p/60835712]{.underline}](https://zhuanlan.zhihu.com/p/60835712)

深度学习的时间序列预测有没有综述? - Ada的回答

[https://www.zhihu.com/question/405169480/answer/1329637122]{.underline}

数据集
------

时间序列数据集中的每个样本为一个**时间序列（Time
Series）**，每个时间序列分为若干**时间步骤（Time
Step）**，每个时间步骤通常为一个标量或者向量。

衡量标准
--------

传统方法及概念
--------------

时序分析有很多传统（非NN）的方法，下面介绍一些比较主流的。

**参考**

[[https://zhuanlan.zhihu.com/p/83511434]{.underline}](https://zhuanlan.zhihu.com/p/83511434)

[[https://en.wikipedia.org/wiki/Time\_series]{.underline}](https://en.wikipedia.org/wiki/Time_series)

### AR模型

**Auto-regressive
Model**，**自动回归模型**，一种处理时间序列的方法，用同一个变量x的之前各周期，即x~1~到x~t-1~，来预测本期x~t~的取值，并假设它们为一线性关系。

因为这是从回归分析中的线性回归发展而来，只是不用x![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image414.png){width="1.0416666666666666e-2in"
height="1.0416666666666666e-2in"}预测y![IMG\_257](/home/jimzeus/outputs/AANN/images/media/image415.png){width="1.0416666666666666e-2in"
height="2.0833333333333332e-2in"}，而是用x预测x（自己），所以叫做自回归。

X的当期值等于一个或数个前期值的线性组合，加常数项，加随机误差。公式如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image416.png){width="1.757638888888889in"
height="0.49236111111111114in"}

其中：

-   c是常数项

-   ![IMG\_259](/home/jimzeus/outputs/AANN/images/media/image417.png){width="0.12222222222222222in"
    height="0.1527777777777778in"}被假设为平均数等于0，标准差等于![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image418.png){width="0.12777777777777777in"
    height="0.12777777777777777in"}的随机误差值；假设![IMG\_259](/home/jimzeus/outputs/AANN/images/media/image417.png){width="0.12222222222222222in"
    height="0.1527777777777778in"}对于任何的t都不变

**参考**

[[https://zh.wikipedia.org/wiki/%E8%87%AA%E8%BF%B4%E6%AD%B8%E6%A8%A1%E5%9E%8B]{.underline}](https://zh.wikipedia.org/wiki/%E8%87%AA%E8%BF%B4%E6%AD%B8%E6%A8%A1%E5%9E%8B)

### VAR模型

**Vector Auto-regression
Model**，**向量自回归模型**，扩充了只能使用一个变量的AR模型，用在多变量时间序列模型上。

一个VAR(p)模型可以写成为：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image419.png){width="3.5618055555555554in"
height="0.2in"}

其中：

-   c是n × 1常数向量

-   A~i~是n × n矩阵。

-   e~t~是n × 1误差向量，满足：

    -   ![](/home/jimzeus/outputs/AANN/images/media/image420.png){width="0.5708333333333333in"
        height="0.16944444444444445in"}：误差项的均值为0

    -   ![](/home/jimzeus/outputs/AANN/images/media/image421.png){width="0.7541666666666667in"
        height="0.18888888888888888in"} ：误差项的协方差矩阵为Ω（一个n ×
        \'n正定矩阵）

    -   ![](/home/jimzeus/outputs/AANN/images/media/image422.png){width="0.8298611111111112in"
        height="0.22291666666666668in"} （对于所有不为0的k都满足）---误差项不存在自相关

**参考**

[[https://zh.wikipedia.org/wiki/%E5%90%91%E9%87%8F%E8%87%AA%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B]{.underline}](https://zh.wikipedia.org/wiki/%E5%90%91%E9%87%8F%E8%87%AA%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B)

### MA模型

**Moving Average
Model**，**滑动平均模型**，是一种对单一变量时间序列进行建模的方法。移动平均模型和自回归模型都是时间序列中** ARMA模型** 和** ARIMA模型** 的重要组成部分，也是一种特殊情况。与自回归模型不同，移动平均模型总是平稳的。

q阶移动平均模型通常简记为**MA(q)**：

![](/home/jimzeus/outputs/AANN/images/media/image423.png){width="2.8402777777777777in"
height="0.20833333333333334in"}

或：

![](/home/jimzeus/outputs/AANN/images/media/image424.png){width="1.4472222222222222in"
height="0.3993055555555556in"}

其中μ是序列的均值，θ~1~,\..., θ~q~ 是参数，ε~t~,ε~t-1~,\..., ε~t−q~都是**白噪声**。

**参考**

[[https://zh.wikipedia.org/wiki/%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%E6%A8%A1%E5%9E%8B]{.underline}](https://zh.wikipedia.org/wiki/%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%E6%A8%A1%E5%9E%8B)

### 白噪声

**白噪声**，是一种功率谱密度为常数的随机信号或随机过程。即**此信号在各个频段上的功率一致**。

由于白光是由各种频率（颜色）的单色光混合而成，因而此信号的平坦功率谱性质称为"白色"，此信号也因此得名为白噪声。相对的，其他不具有这一性质的噪声信号则称为**有色噪声**。

理想的白噪声具有无限带宽，因而其能量是无限大，这在现实世界是不可能存在的。实际上，人常常将有限带宽的平整信号视为白噪声，以方便进行数学分析。

**参考**

[[https://zh.wikipedia.org/wiki/%E7%99%BD%E9%9B%9C%E8%A8%8A]{.underline}](https://zh.wikipedia.org/wiki/%E7%99%BD%E9%9B%9C%E8%A8%8A)

### ARMA模型

**Auto-regressive moving average
model**，**自回归滑动平均模型**。是研究时间序列的重要方法，由**自回归模型**（AR模型）与**移动平均模型**（MA模型）为基础混合构成。

自回归模型AR(p)：

![](/home/jimzeus/outputs/AANN/images/media/image425.png){width="1.573611111111111in"
height="0.43819444444444444in"}

移动平均模型MA(q)：

![](/home/jimzeus/outputs/AANN/images/media/image426.png){width="1.5173611111111112in"
height="0.4326388888888889in"}

**自回归滑动平均模型**ARMA(p,q)包含了p个自回归项和q个移动平均项：

![](/home/jimzeus/outputs/AANN/images/media/image427.png){width="2.638888888888889in"
height="0.4708333333333333in"}

有时ARMA模型可以用滞后算子L来表示，L^i^ X~t~ =
X~t-1~。这样AR(p)模型可以写成：

![](/home/jimzeus/outputs/AANN/images/media/image428.png){width="3.1347222222222224in"
height="0.6590277777777778in"}

MA(q)模型可以写成：

![](/home/jimzeus/outputs/AANN/images/media/image429.png){width="2.8895833333333334in"
height="0.6256944444444444in"}

因此，ARMA(p, q)模型可以表示为：

![](/home/jimzeus/outputs/AANN/images/media/image430.png){width="3.3381944444444445in"
height="0.6104166666666667in"}

**参考**

[[https://zh.wikipedia.org/wiki/ARMA%E6%A8%A1%E5%9E%8B]{.underline}](https://zh.wikipedia.org/wiki/ARMA%E6%A8%A1%E5%9E%8B)

### ARIMA模型

**ARIMA模型**（**Auto-regressive Integrated Moving Average
Model**，**差分整合移动平均自回归模型**），时间序列的预测模型之一。AR、MA、ARMA模型都可以视作ARIMA模型的特例。

**ARIMA(p, d,
q)**中，p为回归项数，q为滑动平均项数，d为使之成为平稳序列的差分次数（阶数）：

![](/home/jimzeus/outputs/AANN/images/media/image431.png){width="3.604861111111111in"
height="0.5736111111111111in"}

其中L 是滞后算子（Lag operator），
![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image432.png){width="0.7208333333333333in"
height="0.12013888888888889in"}

**参考**

[[https://zh.wikipedia.org/wiki/ARIMA%E6%A8%A1%E5%9E%8B]{.underline}](https://zh.wikipedia.org/wiki/ARIMA%E6%A8%A1%E5%9E%8B)

### ARFIMA模型

Auto-regressive fractionally integrated moving average，

**参考**

[[https://en.wikipedia.org/wiki/Autoregressive\_fractionally\_integrated\_moving\_average]{.underline}](https://en.wikipedia.org/wiki/Autoregressive_fractionally_integrated_moving_average)

### ARCH模型

**Auto-regressive conditional heteroskedasticity
model**，**自回归条件异方差模型**。解决了传统的计量经济学对时间序列变量的第二个假设（方差恒定）所引起的问题。这个模型是获得2003年诺贝尔经济学奖的计量经济学成果之一。

**参考**

[[https://zh.wikipedia.org/wiki/ARCH%E6%A8%A1%E5%9E%8B]{.underline}](https://zh.wikipedia.org/wiki/ARCH%E6%A8%A1%E5%9E%8B)

[[https://zhuanlan.zhihu.com/p/21962996]{.underline}](https://zhuanlan.zhihu.com/p/21962996)

### DTW

**Dynamic Time
Warping**，**动态时间规整**，是一种衡量两个长度不同的时间序列的相似度的方法。

**参考**

[[https://blog.csdn.net/zouxy09/article/details/9140207]{.underline}](https://blog.csdn.net/zouxy09/article/details/9140207)

[[https://en.wikipedia.org/wiki/Dynamic\_time\_warping]{.underline}](https://en.wikipedia.org/wiki/Dynamic_time_warping)

### COTE

**论文**

[[https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7069254]{.underline}](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7069254)

### HIVE-COTE (2016ICDM)

**论文**

[[https://core.ac.uk/download/pdf/77027925.pdf]{.underline}](https://core.ac.uk/download/pdf/77027925.pdf)

神经网络方法
------------

本节介绍基于神经网络的时间序列预测方法。

### WaveNet (201609) (TODO)

wavenet是

**论文**

[[https://arxiv.org/pdf/1609.03499.pdf]{.underline}](https://arxiv.org/pdf/1609.03499.pdf)

**代码**

[[https://github.com/ibab/tensorflow-wavenet]{.underline}](https://github.com/ibab/tensorflow-wavenet)

**参考**

谷歌WaveNet 源码详解

[[https://zhuanlan.zhihu.com/p/24568596]{.underline}](https://zhuanlan.zhihu.com/p/24568596)

### DeepAR (201704) 

DeepAR是Amazon提出来的基于深度学习的时间序列预测方法，被收入GluonTS。

DeepAR相较于传统的TS方法，区别如下：

-   纳入了对额外特征的考虑，也接受**其他协变量**作为输入（传统方法只以预测数据项和时间作为输入）

-   其预测目标是序列在每个时间步骤上的**概率分布**（传统方法是单点预测）

下图左为训练过程，右为预测过程

![](/home/jimzeus/outputs/AANN/images/media/image433.png){width="5.752777777777778in"
height="1.675in"}

训练时，在每个实践步骤t，网络的输入特征包括x~i,t~、上一个实践步骤的取值z~i,t-1~，以及上一个时间步骤的隐状态h~i,t-1~。计算当前的隐状态的函数为:

**h~i,t~=h(h~i,t-1~,z~i,t-1~,x~i,t~)**

进而计算似然函数l(z\|θ)的参数θ~i,t~=θ(h~i,t~)，最后将最大对数似然函数作为Loss：

![](/home/jimzeus/outputs/AANN/images/media/image434.png){width="2.0520833333333335in"
height="0.41388888888888886in"}

来学习网络的参数。

DeepAR本质上是一个RNN，具体来说是多层的LSTM（参考gluonts里的实现）

**论文**

[[https://arxiv.org/pdf/1704.04110.pdf]{.underline}](https://arxiv.org/pdf/1704.04110.pdf)

**参考**

[[https://amazonaws-china.com/cn/blogs/china/time-series-prediction-with-deep/]{.underline}](https://amazonaws-china.com/cn/blogs/china/time-series-prediction-with-deep/)

[[https://www.jianshu.com/p/8a900b9ad3d3]{.underline}](https://www.jianshu.com/p/8a900b9ad3d3)

### Deep state(201800) (TODO)

**论文**

[[https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf]{.underline}](https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf)

### ForGAN（201903）

**论文**

[[https://arxiv.org/pdf/1903.12549.pdf]{.underline}](https://arxiv.org/pdf/1903.12549.pdf)

gitlab：

[[https://git.opendfki.de/koochali/forgan]{.underline}](https://git.opendfki.de/koochali/forgan)

**参考**

### Deep Factor (201905) (TODO)

**论文**

[[https://arxiv.org/pdf/1905.12417.pdf]{.underline}](https://arxiv.org/pdf/1905.12417.pdf)

**参考**

### Informer（202012）

**论文**

[[https://arxiv.org/pdf/2012.07436v3.pdf]{.underline}](https://arxiv.org/pdf/2012.07436v3.pdf)

**代码**

[[https://github.com/zhouhaoyi/Informer2020]{.underline}](https://github.com/zhouhaoyi/Informer2020)

**参考**

AAAI21最佳论文Informer：效果远超Transformer的长序列预测神器！

[[https://mp.weixin.qq.com/s/RRv-DVm6SguQ5GC5oruf8Q]{.underline}](https://mp.weixin.qq.com/s/RRv-DVm6SguQ5GC5oruf8Q)

框架
----

目前只有gluonts一个比较系统的时间序列框架

### gluonts

gluonts是亚马逊推出的**时间序列（Time
Series）**线下建模工具包。其中包括：

-   用于创建模型的组件

-   数据加载、训练和处理工具

-   一些已有数据集

-   一些已有模型

-   图表绘制和评估

Gluonts的主要数据结构包括：

-   **Dataset**：数据集，gluonts中的dataset用于提供访问数据的统一接口

-   **Estimator**：可训练的模型

-   **Predictor**：不可训练的模型

-   **Trainer**：训练器，主要描述了一系列训练的参数，并提供了训练的核心函数，被传递给Estimator

-   **Evaluator**：提供了（对模型的）多尺度评价

    以及：

<!-- -->

-   **Distribution**：描述一个具体的**概率分布**的类

-   **DistributionOutput**：产出**Distribution**的类

-   **Forecast**：描述**预测**的类（以采样、分位数等方式）

**论文**

Gluonts: Probabilistic Time Series Models in Python

[[https://arxiv.org/pdf/1906.05264.pdf]{.underline}](https://arxiv.org/pdf/1906.05264.pdf)

**官方文档**

[[https://gluon-ts.mxnet.io/]{.underline}](https://gluon-ts.mxnet.io/)

Tutorial

[[https://gluon-ts.mxnet.io/examples/basic\_forecasting\_tutorial/tutorial.html]{.underline}](https://gluon-ts.mxnet.io/examples/basic_forecasting_tutorial/tutorial.html)

An Overview of GluonTS

[[https://github.com/aws-samples/amazon-sagemaker-time-series-prediction-using-gluonts]{.underline}](https://github.com/aws-samples/amazon-sagemaker-time-series-prediction-using-gluonts)

**代码**

[[https://github.com/awslabs/gluon-ts.git]{.underline}](https://github.com/awslabs/gluon-ts.git)

#### 基本

-   mx.block：网络中用到的各种单元结构（MXNet架构）

    -   cnn：

        -   CasualConv1D：一维的因果卷积（指的是output\[i\]不依赖input\[i+1\]）

        -   DilatedCasualGated：膨胀的因果卷积

        -   ResidualSequential：

-   core：

#### dataset

gluonts.dataset为数据集相关的包，主要包括的数据类型有：

-   TrainDatasets：包括测试集和训练集的完整数据集

-   Dataset：单个数据集（测试集、训练集）

-   DataEntry：单个样本

-   DataBatch：

-   DataLoader：数据加载，迭代出DataBatch

##### 数据结构

class TrainDatasets(NamedTuple):

metadata: MetaData

train: Dataset

test: Optional\[Dataset\] = None

一个**完整的数据集**类型为TrainDatasets，父类为命名元组（NamedTuple），包含了**训练集**train和**测试集**test（均为Dataset类型），以及**元数据**（MetaData类型）

class BasicFeatureInfo(pydantic.BaseModel):

name: str

class CategoricalFeatureInfo(pydantic.BaseModel):

name: str

cardinality: str

class MetaData(pydantic.BaseModel):

freq：str

target：Optional\[BasicFeatureInfo\] = None

feat\_static\_cat: List\[CategoricalFeatureInfo\] = \[\]

feat\_static\_real: List\[BasicFeatureInfo\] = \[\]

feat\_dynamic\_real: List\[BasicFeatureInfo\] = \[\]

feat\_dynamic\_cat: List\[CategoricalFeatureInfo\] = \[\]

prediction\_length: Optional\[int\] = None

类MetaData描述了数据集的元信息

-   **feat\_static\_cat**：可以用于对记录所属的组进行编码的分类特征数组。分类要素必须编码为基于0的正整数序列。例如，分类域{R，G，B}可以编码为{0，1，2}。来自每个分类域的所有值都必须在训练数据集中表示。

-   **feat\_static\_real**：

-   **feat\_dynamic\_real**：代表自定义要素时间序列（动态要素）向量的浮点值或整数数组。如果设置此字段，则所有记录必须具有相同数量的内部数组（相同数量的特征时间序列）。此外，每个内部数组必须具有与关联target值相同的长度
    。例如，如果目标时间序列代表不同产品的需求，则feat\_dynamic\_real可能是布尔时间序列，它指示是否对特定产品应用了促销。

-   **feat\_dynamic\_cat**：

    Dataset = Iterable\[DataEntry\]

    class FileDataset(Dataset):

    class ListDataset(Dataset):

    一个**训练集**或者**测试集**的类型为Dataset

<!-- -->

-   实际格式为DataEntry（即字典，key为字串，value为任意类型数据）的Iterable（迭代容器）。

-   有两种dataset：FileDataset和ListDataset，均为DataSet的子类

-   如果想自己实现一个训练集或者测试集，需要：

    -   生成gluonts.dataset.common.Dataset的子类

    -   子类实现\_\_iter\_\_()，返回DataEntry的迭代(iterator)，每次迭代给出一个数据样本DataEntry

    -   子类实现\_\_len\_\_()

    -   字典中要有"start"和"target"字段

        DataEntry = Dict\[str, Any\]

        DataEntry表示数据集中的单个样本，即一个**时间序列（Time
        Series）**，这是一个**泛型别名类型**，实际是一个字典dict，其key为字符串，value为任意类型。

字典key对某些str有缺省的定义，在gluonts.dataset.field\_names.FieldName中：

-   必须有的字典key：

    -   "start"：时间序列的开始时间

    -   "target"：时间序列的值

-   可选的key：

    -   "feat\_static\_cat"：

    -   "feat\_static\_real"：

    -   "feat\_dynamic\_cat"：

    -   "feat\_dynamic\_real"：

-   通过各种变换（Transformation）获得的key：

    -   "time\_feat"：

    -   "feat\_dynamic\_const"：

    -   "feat\_dynamic\_age"：通常给AddAgeFeature变换存放输出结果

    -   "observed\_values"：通常给AddObeservedValueIndicator变换存放输出的结果

    -   "is\_pad"：

    -   "forecast\_start"：

        DataEntry\["target"\]

        每个时间序列分为若干**时间步骤（Time
        Step）**，每个时间步骤通常为一个标量或者向量。是DataEntry\["target"\]字段的一个元素。

        DataBatch = Dict\[str, Any\]

        描述一个batch的数据的类型为DataBatch，同DataEntry一样也是一个**泛型别名类型**，为一个key为字符串，value为任意类型的字典。

        class DataLoader(Iterable\[DataBatch\]):

        data\_iterable: Iterable\[DataEntry\] \#关联的数据集

        batch\_size: int

        stack\_fn: Callable\#

        num\_workers: int \#用于数据预处理的并行线程数，缺省为None

        num\_prefetch: int \#预取的batch数，缺省为None

        decode\_fn: Callable \#

        DataLoader为数据加载器的类，用于指定的数据集中迭代数据，迭代出DataBatch。

##### 包结构

-   dataset：数据集相关

    -   common：数据集的类

        -   DataEntry：Dict\[str,
            > Any\]，描述单个时间序列（即一个样本）的数据帧

        -   DataBatch：Dict\[str, Any\]，

        -   Dataset：Iterable\[DataEntry\]，DataEntry的迭代，单个数据集

        -   ListDataset：数据格式为dict的数据集，Dataset的子类

            -   data\_iter：DataEntry的迭代

            -   freq：每个时间步骤的长度

            -   one\_dim\_target：

        -   FileDataset：从JSON文件加载的数据集，Dataset的子类

            -   path：数据集文件路径

            -   freq：每个时间步骤的长度

            -   one\_dim\_target：

            -   cache：

        -   MetaData：数据集的元信息，包括prediction\_length等

        -   TrainDatasets：描述一个完整的、包括训练集和测试集的数据集。为NamedTuple的子类，包括三个元素：

            -   metadata：元信息，MetaData类型

            -   train：训练集，Dataset类型

            -   test：测试集，Dataset类型

            -   save()：保存

        -   load\_datasets()：加载数据集，返回TrainDatasets

    -   field\_names：DataEntry中字典key的缺省定义

        -   FieldName：定义了DataEntry中各个有特殊含义的字段的字段名

            -   ITEM\_ID：字段名"item\_id"

            -   START：时间序列的开始时间，DataEntry中的字段名为"start"

            -   TARGET：时间序列的值，字段名为"target"

            -   FEAT\_STATIC\_CAT：字段名"feat\_static\_cat"

            -   FEAT\_STATIC\_REAL：字段名"feat\_static\_real"

            -   FEAT\_DYNAMIC\_CAT：字段名"feat\_dynamic\_cat"

            -   FEAT\_DYNAMIC\_REAL：字段名"feat\_dynamic\_real"

            -   FEAT\_TIME：字段名"time\_feat"

            -   FEAT\_CONST：字段名"feat\_dynamic\_cost"

            -   FEAT\_AGE：字段名"feat\_dynamic\_age"

            -   OBESERVED\_VALUES：字段名"obsevered\_values"

            -   IS\_PAD：字段名"is\_pad"

            -   FORECAST\_START：字段名"forecast\_start"

            -   TARGET\_DIM\_INDICATOR：字段名"tareget\_dim\_indicator"

    -   loader：数据加载器，用于返回DataBatch的迭代

        -   DataLoader：所有加载器的基类

            -   parallel\_data\_loader：核心成员，实现了数据的迭代

            -   \_\_init\_\_()：构造函数

            -   \_\_iter\_\_()：迭代函数

        -   TrainDataLoader()：返回训练数据集（DataLoader实例）

        -   ValidationDataLoader()：返回验证数据集（DataLoader实例）

        -   InferenceDataLoader()：返回推理数据集（DataLoader实例）

    -   utils：数据集工具

        -   to\_pandas()：将dict转换为pandas.Series格式

    <!-- -->

    -   repository：已经封装好的数据集，其中的datasets.get\_dataset()函数常用

        -   \_\*：各种已有数据集的生成函数（来自网络）

        -   datasets：

            -   datasets\_recipes：预设部分参数的**数据集生成函数**的字典

            -   **get\_datasets()**：根据关键字，生成某数据集，返回TrainDataset

    -   artificial：人工生成的时间序列数据集

        -   \_base.ComplexSeasonalTimeSeries：

            -   num\_series：时间序列的样本数量

            -   prediction\_length：预测的时间步骤

            -   freq\_str：频率，缺省为"H"（小时）

            -   length\_low：最短的时间序列长度（必须高于prediction\_length）

            -   lenght\_high：最长的时间序列长度

            -   min\_val：时间序列的最小值

            -   max\_val：时间序列的最大值

            -   is\_integer：

            -   proportion\_missing\_values：

            -   is\_noise：是否添加噪音

            -   is\_scale：是否

            -   percentage\_unique\_timestamps：

            -   is\_out\_of\_bounds\_date：

#### distribution

mx.distribution（老版本中的distribution）目录中包含了分布相关的内容（MXNet架构）。虽然代码行数不多，但与概率论及统计学结合比较紧密，且代码比较晦涩难懂。与此相关的一些概念可以参考：《[[概念定义
\> NN相关 \> 分布Distribution]{.underline}](\l)》和《[[概念定义 \>
概率论]{.underline}](\l)》。

gluonts中最常用的分布是**StudentT（学生t分布）**，参考《[[概念定义 \>
概率论 \>学生t分布]{.underline}](\l)》。

**混淆注意！**

Distribution和DistributionOutput的区别在于：

-   DistributionOutput描述某类型的分布（泊松分布、正态分布等），它和模型绑定，经过特定数据集的训练之后，吐出一个Distribution

-   Distribution描述了一个参数已定的具体分布，它的参数经由训练确定

##### 数据结构

class Distribution：

distribution.py中只包含一个类Distribution，用于描述一个具体（即参数已定）的分布：

-   是个基类，其子类则是不同类型的分布，比如**泊松分布**、**高斯分布**等

-   包含了各种**决定该分布的参数**，比如

    -   mean：期望值（均值）

    -   stddev：标准差

    -   variance：方差，stddev的平方

    -   batch\_shape：batch的形状，抽象方法

    -   batch\_dim：batch的维度，len(batch\_shape())

    -   event\_shape：event的形状

    -   event\_dim：event的维度，标量则为0

    -   all\_dim：batch\_dim + event\_dim

    -   \...\...

-   提供了各种关于该分布的函数，比如

    -   sample(num\_samples)：从该分布中抽签，数量为num\_samples

    -   cdf(x)：在x的**CDF**（**累积分布函数**），抽象方法

    -   crps(x)：在x的**CRPS**（**连续概率分级评分**），抽象方法

    -   quantile()：分位数

    -   log\_prob(x)：在点x的log密度，抽象方法

    -   loss(x)：在点x的损失，缺省为负的log\_prob()

    -   prob(x)：在点x的密度，缺省为log\_prob().exp()

    -   \...\...

        class StudentT(Distribution):

        描述学生t分布的StudentT是Distribution的子类

<!-- -->

-   有三个标量参数用于描述一个学生t分布：

    -   mu（μ）：期望值，形为batch\_shape\*event\_shape

    -   sigma（σ）：标准差，形为batch\_shape\*event\_shape

    -   nu（ν）：自由度，形为batch\_shape\*event\_shape

-   是标量，因此batch\_shape为mu.shape，而event\_dim为0

    class TransformedDistribution(Distribution):

    transformed\_distribution.py定义了TransformedDistribution，描述基于一个基础分布，经过一系列变换之后得到的分布。

    class ArgProj(gluon.HybridBlock):

    class Output:

    class DistributionOutput(Output):

    distribution\_output.py定义了DistributionOutput，以及另外两个相关的类，用于描述分布的输出：

<!-- -->

-   ArgProj：MXNet神经网络模块，用于将**网络输出**转化为**分布参数**

    -   是gluon.HybridBlock的子类

    -   输入是全连接层（即神经网络的输出），输出是Distribution的参数。

    -   内部处理过程由两部分组成，此二者皆作为参数被传入构造函数：

        -   先是若干并行的全连接层，输入均是网络输出，输出为若干**中间参数**（参数的个数、名称和维度来自参数args\_dim）

        -   后接一个映射函数，来自参数domain\_map，其输入为**中间参数**，输出为**分布参数**

    -   在抽象基类Output的成员函数get\_args\_proj()中生成该类的实例并返回。args\_dim和domain\_map都是Output的成员，但都在具体的子类中被赋值。

-   Output

    -   是一个**抽象基类**，DistributionOutput和BijectionOutput的父类

    -   成员domain\_map和args\_dim，参考ArgProj，**均由子类赋值**

        -   args\_dim：Distribution的参数，字典类型，参数个数为字典长度，key为输出的参数名，value为参数的维度

        -   domain\_map()：映射函数，输入和输出都是分布参数，其中做了处理

    -   成员函数get\_args\_proj()：根据domain\_map和args\_dim，返回一个ArgProj的实例

-   DistributionOutput：

    -   Output的子类

    -   用于产生具体的（即带参数的）分布（即Distribution的实例）

    -   抽象基类，子类是具体的分布输出，比如：

        -   学生T分布StudentTOutput

        -   泊松分布输出PoissonOutput

        -   伽玛分布输出GammaOutput

        -   \...

    -   成员distr\_cls为具体的分布类型，比如StudentT，Poisson，Gamm等

    -   成员函数distribution()：

        -   输入为分布的参数distr\_args

        -   输出为带参数的Distribution

        -   缺省返回distr\_cls(distr\_args)，即根据**分布参数**直接构造一个分布

        -   可能被子类覆盖

            class StudentTOutput(DistributionOutput):

            args\_dim: Dict\[str, int\] = {\"mu\": 1, \"sigma\": 1,
            \"nu\": 1}

            distr\_cls: type = StudentT

            def domain\_map(cls, F, mu, sigma, nu):

            StudentTOutput是DistributionOutput的子类，也是gluonts中最常用的分布，用于产生学生t分布。

            这几个关于分布输出的类有点绕，下面以学生t分布为例，解构一下：

<!-- -->

-   ArgProj是神经网络模块，由若干平行Dense层（args\_dim）和转换函数（domain\_map）组成，输入为glutonts神经网络的输出，经过Dense层输出中间参数，在经过domain\_map输出分布参数。args\_dim和domain\_map是构造函数的参数。

-   Output类的大意是：神经网络输出-\>某种参数，主要向外提供了成员函数**get\_args\_proj**()用于返回ArgProj，此外有成员domain\_map和args\_dim的声明作为ArgProj的参数，但是没有定义，需子类定义

-   DistributionOutput是Output的子类，定义了**distribution**()函数，该函数返回根据输入的分布参数返回一个分布，这里也没有实现domain\_map和args\_dim

-   StudentTOutput作为DistributionOutput的子类，终于实现了**domain\_map**和**args\_dim**，供基类中的函数回调。有的分布会覆盖distribution()的实现。

    以上这些类总体向外提供了get\_args\_proj和distribution两个函数，使用者通过get\_args\_proj()得到**分布参数**（的函数），再将这些参数传入distribution()得到具体的分布。

    类继承关系如下：

-   Output

    -   DistributionOutput

        -   StudentTOutput

        -   GaussianOutput

        -   \....

    -   BijectionOutput

        -   BoxCoxTransformOutput

##### 包结构

-   distribution：分布相关的包

    -   distribution.Distribution：抽象基类，描述一个具体的概率分布，相关概念可以参考《[[概念定义 \>
        > NN相关 \>
        > 分布Distribution]{.underline}](\l)》和《[[概念定义 \>
        > 概率论]{.underline}](\l)》

        -   batch\_shape()：batch shape，抽象方法，需子类实现

        -   batch\_dim：batch shape的维度，len(batch\_shape)

        -   event\_shape()：event shape，抽象方法，需子类实现

        -   event\_dim：event\_shape的维度，len(event\_shape)

        -   all\_dim：所有张量的总维度，batch\_dim + event\_dim

        -   log\_prob()：根据x计算概率（以e为底的对数），抽象方法，需子类实现

            -   x：描述事件，其形状为batch\_shape \* event\_shape

        -   loss()：根据x计算loss，直接返回负的log\_prob()

        -   prob()：根据x计算概率，返回exp(log\_prob())

        -   crps()：

        -   sample()：返回若干次抽签，其形状为num\_samples \*
            > batch\_shape \* event\_shape

            -   num\_samples：返回的抽签数

        -   mean：平均值（即期望值）

        -   stddev：标准差

        -   variance：方差

    <!-- -->

    -   distribution\_output.ArgProj：是HybridBlock的子类，实例是Callable的，最终会调用到hybrid\_forward()，ArgProj的实例用于将Dense层转换为Distribution的**参数**。在Output.get\_args\_proj()中返回其实例。

        -   \_\_init\_\_()：构造函数，根据args\_dim生成self.proj（Dense层的组合），保存domain\_map，这些成员在hybrid\_forward()中被使用

            -   args\_dim：字典，key为str类型的Dense层名，value为该层单元数

            -   domain\_map：传入的映射函数

            -   dtype：

            -   prefix：表示层名前缀的字符串，缺省为None

        -   proj：若干层Dense层，构造函数中被赋值

        -   hybrid\_forward()：实际的转换过程。根据self.proj和self.domain\_map生成并返回所需的Distribution参数（输入x \>
            > proj \> domain\_map）

    -   distribution\_output.Output：抽象基类，给出分布

        -   get\_args\_proj()：返回一个ArgProj对象

        -   args\_dim：get\_args\_proj()所返回的ArgProj的参数，具体由子类定义

        -   domain\_map()：需子类实现的抽象方法，域的映射，将输入张量转换到合理的**形状**和**值域**，是ArgProj对象的最后一个步骤

    -   distribution\_output.DistributionOutput：抽象基类，Output的子类，通过处理网络的输出结果，输出Distribution。

        -   distr\_cls：具体的DistributionOutput的子类类型

        -   distribution()：返回一次具体的分布Distribution。

        -   domain\_map()：父类Output的抽象方法，由具体的子类实现

    <!-- -->

    -   poisson.Poisson：Distribution的子类，泊松分布

    <!-- -->

    -   poisson.PoissonOutput：DistributionOutput的子类

        -   args\_dim：覆盖Output的args\_dim，值为{"rate":1}

        -   domain\_map()：Soft Plus函数，被父类Output传入ArgProj中使用

        -   distribution()：返回一个Poisson

    -   student\_t.StudentT：**学生t分布**，Distribution的子类

        -   \_\_init\_\_()：构造函数

            -   mu：期望值的张量，形状为batch\_shape \* event\_shape

            -   sigma：标准差的张量，形状为batch\_shape \* event\_shape

            -   nu：自由度的张量，形状为batch\_shape \* event\_shape

    -   student\_t.StudentTOutput：学生t分布的输出类

        -   domain\_map()：

#### trainer

mx.trainer目录中包含了训练相关的内容。

##### 数据结构

class Trainer:

mx/trainer/\_base.py（老版本是trainer/\_base.py）中的类Trainer定义了一个**网络应该如何被训练**。这里的参数主要包括两个部分：

-   一部分定义了训练的样本的数量，包括：

    -   epochs：训练多少epoch，缺省为100

    -   batch\_size：一个batch中包含多少个样本，缺省为32

    -   num\_batchs\_per\_epoch：每个epoch中多少batch，缺省为50

-   另外一部分定义了梯度该如何更新：

    -   learning\_rate：学习率，缺省为0.001（10^-3^)

    -   learning\_rate\_decay\_factor：学习率衰减系数，缺省为0.5

    -   patience：等待多少轮之后开始降低学习率

    -   minimum\_learning\_rate：最小学习率，缺省为10^-5^

    -   clip\_gradient：

    -   weight\_decay：权值衰减系数

-   除了参数之外，Trainer主要定义了\_\_call\_\_()函数（即Trainer对象可以被直接调用），这也是训练过程的核心函数，在Estimator的train()函数中被间接调用。

#### model

model是模型相关的包，主要包括两种模型的基类Estimator（可训练的模型）和Predictor（不可训练的模型，由Estimator训练出来），以及各种继承了这两者的具体的（神经网络或非神经网络）模型。

##### 数据结构

以下是模型相关的数据结构，主要的基类包括：

-   **Estimator**：估计器，可训练的模型

-   **Predictor**：预测器，不可训练的模型

    -   经Estimator训练得来（Estimator.train()返回）

-   **Trainer**：训练器，主要描述了一系列训练的参数，及训练过程，被传递给Estimator

    在已有模型的情况下，最基本的流程为：

1.  设置Trainer，传递给Estimator

2.  estimator.train()，参数为训练dataset，训练出Predictor返回

3.  predictor.predict()，参数为测试dataset

##### Estimator

class Estimator:

Estimator是所有可训练模型的抽象基类，子类包括GluonEstimator、DummyEstimator。

class DummyEstimator(Estimator):

DummyEstimator的构造函数直接传入一个predictor，train并不训练，直接返回这个predictor。

class GluonEstimator(Estimator):

GluonEstimator是Estimator的主要子类，也是所有gluon格式算法模型的父类：

-   \_\_init\_\_()：构造函数，传入trainer（Trainer的实例）

-   定义了三个抽象方法（需要子类实现）：

    -   create\_transformation()

    -   create\_training\_network()

    -   create\_predictor()

-   train\_model()：GluonEstimator的主要函数，流程为：

    -   由create\_tranformation()生成transformation（处理输入数据）

    -   根据训练数据集，经过transformation，生成TrainDataLoader的实例

    -   根据验证数据集生成ValidationDataLoader的实例

    -   由create\_training\_network()生成训练网络trained\_net

    -   调用self.trainer()进行训练

    -   由create\_predictor()生成Predictor

    -   返回transformation、trained\_net、predictor

-   train()：调用train\_model()，并仅返回predictor

    每个具体的算法都对应一个Estimator的子类，其中需要实现3个基类中的抽象方法。而每个算法也都对应一个训练网络和一个预测网络，分别用于训练和推理。

##### Predictor

所有不可训练网络的基类Predictor及其子类的继承关系如下：

-   Predictor：

    -   RepresentablePredictor：

        -   NPTS、Seasonal Naive、R
            > Forecast、Naive2、Prophet、trivial下的各种

    -   GluonPredictor：

        -   SymbolBlockPredictor：

            -   TPP（Temporal Point Process）算法、Rotbaum Tree

        -   RepresentableBlockPredictor：

            -   Canonical、Transformer、Deep Factor、DeepAR、Seq2Seq、N
                > beats、Simple Feedforward、Deep State、GP
                > Forecaster、GPVAR、WaveNet算法均以次为Predictor

        -   TPP（Temporal Point Process）算法、Rotbaum Tree

    -   FallbackPredictor：ConstantValuePredictor、MeanPredictor

    -   ParallelizedPredictor：Not Used

    -   Localizer：Not Used

        class Predictor:

        def predict(self, dataset: Dataset, \*\*kwargs) -\>
        Iterator\[Forecast\]:

        model.predictor.Predictor是所有predictor（训练完成的模型）的抽象基类

<!-- -->

-   成员变量包括：

    -   prediction\_length：预测的长度，单位为freq

    -   freq：时间步骤的频率，比如'h'

    -   lead\_time：

-   其子类需实现抽象方法predict()，返回Forecast的迭代

-   直接子类包括RepresentablePredictor和GluonPredictor

    class RepresentablePredictor(Predictor):

    def predict\_item(self, item: DataEntry) -\> Forecast:

    model.predictor.RepresentablePredictor是Predictor的一个子类：

<!-- -->

-   用于给各种非Gluon模型作为基类，比如NPTS、Seasonal Naive等

-   实现了predict()函数，但仅仅是将传入的参数dataset的每个迭代作为参数调用了predict\_item()

-   predict\_item()是需要子类实现的抽象方法，参数为dataset的迭代

    class GluonPredictor(Predictor):

    class RepresentableBlockPredictor(GluonPredictor):

    mx.model.predictor.GluonPredictor是基于gluon的模型的基类：

-   有如下成员变量：

    -   prediction\_net：网络模型

    -   input\_transform：输入前的变换

    -   output\_transform：输出前做的变换

    -   forecast\_generator：缺省为SampleForecastGenerator

-   predict()：根据dataset建立InferenceDataLoader，并通过forecast\_generator迭代Forecast

    mx.model.predictor.RepresentableBlockPredictor是GluonPredictor的子类，也是众多具体的gluon模型的predictor的基类。

##### 非Gluon模型

class ConstantPredictor(RepresentablePredictor):

model.trivial.ConstantPredictor每次都输出同样的一个Forecast

-   成员变量samples保存用于建立SampleForecast的采样，初始化时赋值

-   predict\_item()每次返回一个由samples赋值建立的SampleForecast的实例

    class ConstantValuePredictor(RepresentablePredictor,
    FallbackPredictor):

    model.trivial.ConstantValuePredictor每次输出一个由固定常量构成的Forecast

-   成员变量value用于保存该常量，初始化时候被赋值

-   predict\_item()中先根据value生成sample，再根据它生成SampleForecast返回

    class MeanPredictor(RepresentablePredictor, FallbackPredictor):

    model.trivial.MeanPredictor基于传入数据集的最后context\_length个实践步骤的期望值和标准差，生成Sample

-   成员变量：

    -   context\_length：用于定义生成mean和std的数据集时间步骤长度

    -   num\_samples：生成采样的数量

    -   prediction\_length：预测的长度

-   predict\_item()：根据context\_length算出mean和std，再根据mean和std的正态分布，以num\_sample和prediction\_length的形状采样出预测的张量，以SampleForecast形式返回

##### SFF模型

以**Simple Feed Forward算法**为例：

class SimpleFeedForwardEstimator(GluonEstimator)：

-   成员create\_transformation()：返回仅有一个InstanceSplitter的Transformation

-   成员create\_training\_network()：返回SimpleFeedForwardTrainingNetwork

-   create\_predictor()：

    -   实例化一个SimpleFeedForwardPredictionNetwork，网络参数来自参数

    -   加上transformation，构成RepresentableBlockPredictor，返回

        class SimpleFeedForwardNetworkBase(mx.gluon.HybridBlock):

        class
        SimpleFeedForwardTrainingNetwork(SimpleFeedForwardNetworkBase):

        class
        SimpleFeedForwardPredictionNetwork(SimpleFeedForwardNetworkBase):

<!-- -->

-   SFF的训练网络和预测网络继承同一个基类SimpleFeedForwardNetworkBase，由MLP和Distribution构成

    -   成员变量mlp为全连接层的MLP，来自于构造函数的参数

    -   get\_distr()函数输入为网络输入，经过MLP和DistributionOutput，返回分布

<!-- -->

-   训练网络SimpleFeedForwardTrainingNetwork：

    -   hybrid\_forward()调用get\_distr()，根据分布返回损失值

-   预测网络SimpleFeedForwardPredictorNetwork：

    -   hybrid\_forward()调用get\_distr()，根据分布返回抽签

##### DeepAR模型

以**DeepAR算法**为例：

class DeepAREstimator(GluonEstimator):

-   create\_transformation()：返回一个Transformation

-   create\_training\_network()：返回DeepARTrainingNetwork

-   create\_predictor()：返回transformation和构成的

    -   实例化DeepARPredictionNetwork为预测网络

    -   将训练网络（来自函数参数）的参数传给这个预测网络

    -   根据transformation（来自函数参数）和预测网络，生成一个RepresentableBlockPredictor作为Predictor返回

        class DeepARNetwork(mx.gluon.HybridBlock):

        class DeepARTrainingNetwork(DeepARNetwork):

        class DeepARPredictionNetwork(DeepARNetwork):

<!-- -->

-   DeepARNetwork是DeepAR算法的训练网络和预测网络的基类：

    -   get\_lagged\_subsequences()

        -   被self.unroll\_encoder()调用

        -   被DeepARPredictionNetwork.sampling\_decoder()调用

    -   unroll\_encoder()

        -   被DeepARTrainingNetwork.distribution()调用

        -   被DeepARPredictionNetwork.hybrid\_forward()调用

-   DeepAR算法的训练网络是DeepARTrainingNetwork：

    -   distribution()：

        -   被self.hybrid\_forward()调用

    -   hybrid\_forward()：

-   DeepAR算法的预测网络是DeepARPredictionNetwork：

    -   sampling\_decoder()：

        -   被self.hybrid\_forward()调用

    -   hybrid\_forward()：

##### Transformer模型

以**Transformer算法**为例：

class TransformerEstimator(GluonEstimator):

-   成员encoder是编码器，TransformerEncoder的实例

-   成员decoder为解码器，TransformerDecoder的实例

-   create\_transformation()：返回一个Transformation

-   create\_training\_network()：返回TransformerTrainingNetwork

-   create\_predictor()：返回transformation和构成的

    -   实例化TransformerPredictionNetwork为预测网络

    -   将trained\_network（来自函数参数）的参数传给这个预测网络

    -   根据transformation（来自函数参数）和预测网络，生成一个RepresentableBlockPredictor作为Predictor返回

        class TransformerNetwork(mx.gluon.HybridBlock):

        class TransformerTrainingNetwork(TransformerNetwork):

        class TransformerPredictionNetwork(TransformerNetwork):

<!-- -->

-   TransformerNetwork是抽象基类，包括两个函数

    -   get\_lagged\_subsequence()

        -   被create\_network\_input()调用

        -   被TransformerPredictionNetwork.sampling\_decoder()调用

    -   create\_network\_input()

        -   被两个子类的hybrid\_forward()调用

-   TransformerTrainingNetwork是训练网络

    -   hybrid\_forward()

-   TransformerPredictionNetwork是预测网络

    -   sampling\_decoder()

    -   hybrid\_forward()

##### 包结构

-   model：模型相关

    -   forecast：预测结果

        -   Forecast：抽象类，表示预测结果

            -   start\_date：预测结果的开始时间

            -   freq：同Estimator.freq

            -   item\_id：

            -   info：

            -   prediction\_length：预测多少个时间步骤

            -   mean：平均值

            -   plot()：将预测结果绘制成图表（与matplotlib.pyplog结合）

                -   prediction\_intervals：所绘制的预测的置信区间，为0-100间的浮点数的List，越小表示置信度越高的部分。

                -   show\_mean：是否显示

                -   color：颜色

        -   SampleForecast：Forecast的子类

            -   num\_samples：每个预测的样本个数

            -   samples：用于表示预测结果的张量，ndarray类型，如果每个时间步骤
                > 为标量，则形如 （num\_samples,
                > prediction\_length）；如
                > 果每个时间步骤为target\_dim维向量，则形如
                > （num\_samples, prediction\_length, target\_dim）

            -   mean：（num\_samples个样本的）平均值

    -   forecast\_generator：

        -   ForecastGenerator：

        -   DistributionForecastGenerator：

        -   QuantileForecastGenerator：

        -   SampleForecastGenerator：

    -   predictor：预测器，不可训练的模型，由Estimator.train()生成

        -   Predictor：抽象类，表示不可训练模型

            -   \_\_init\_\_()：构造函数

                -   freq：时间粒度，每两个数据帧之间的时间间隔

                -   prediction\_length：预测输出的长度，即预测几个时间步骤

                -   lead\_time：

            -   predict()：虚函数，预测并返回Forecast的迭代

            -   serialize()：序列化，保存到指定位置

            -   deserialize()：反序列化，类方法，从指定位置读取predictor

        -   GluonPredictior：gluon格式模型的基类，Predictor的子类

            -   \_\_init\_\_()：构造函数

                -   prediction\_net：预测网络（例：DeepARPredictionNetwork）

                -   forecast\_generator：缺省为SampleForecastGenerator实例

                -   output\_tranform：对输出数据的变换

            -   predict()：父类predict()的实现，调用forecast\_generator

        -   RepresentableBlockPredictor：

    -   estimator：估计器，可训练的模型

        -   Estimator：抽象类，表示可训练模型

            -   freq：时间粒度，每两个数据帧之间的时间间隔

            -   prediction\_length：预测输出的长度，即预测几个时间步骤

            -   lead\_time：

            -   train()：子类须实现的虚函数，训练模型并返回Predictor

                -   training\_data：训练数据集（Dataset）

                -   validation\_data：验证数据集（Dataset）

        -   GluonEstimator：继承Estimator，其子类为具体算法的Estimator

            -   \_\_init\_\_()：

                -   trainer：训练器，类型为Trainer

            -   from\_hyperparameters()：类方法

            -   create\_transformation()：子类实现的抽象方法，返回一个Transformation，用于在相应的Estimator处理数据**之前**做变换。

            -   create\_training\_network()：子类实现的抽象方法，返回**训练用网络**

            -   create\_predictor()：子类实现的抽象方法，返回**predictor**

            -   train\_model()：训练模型，仅被train()调用。流程中
                > 先后调用上述几个子类实现的抽象函数，返回结果

            -   train()：直接调用train\_model()，返回predictor

    <!-- -->

    -   simple\_feedforward：全连接层构成的Simple Feed Forward的模型

        -   \_network：包含了Simple Feed Forward对应的MXNet的网络模型

            -   SimpleFeedForwardNetworkBase：mx.gluon.HybridBlock的子类，下面训练和推理两个网络模型的基类，实现了全Dense层的网络模型。

                -   \_\_init\_\_()：构造函数

                    -   num\_hidden\_dimensions：MLP的隐藏层维度

                    -   prediction\_length：预测长度

                    -   context\_length：被用于预测的背景长度

                    -   batch\_normalization：是否使用BN

                    -   mean\_scaling：

                    -   distr\_output：输出结果的分布类型，DistributionOutput类型，处理网络输出结果，输出分布，缺省为StudentTOutput。

                -   mlp：根据构造函数参数创建的MLP，私有成员

                -   get\_distr()：根据输入数据past\_target，通过mlp推理出结果，再返回分布（根据构造函数的参数distr\_output）

            -   SimpleFeedForwardTrainingNetwork：用于训练的SFF网络模型，SimpleFeedForwardNetworkBase的子类

                -   hybrid\_forward()：返回loss值

            -   SimpleFeedForwardPredictionNetwork：用于推理的SFF网络模型，SimpleFeedForwardNetworkBase的子类

                -   hybrid\_forward()：返回一批结果

        <!-- -->

        -   \_estimator.SimpleFeedForwardEstimator：GluonEstimator的子类，MLP结构的estimator（可训练模型）

            -   \_\_init\_\_()：构造函数

                -   freq：频率

                -   prediction\_length：预测长度

                -   trainer：训练参数，缺省为Trainer的缺省实例

                -   num\_hidden\_dimensions：各隐藏层的节点数，缺省为None（对应成员变量为\[40,40\]）

                -   context\_length：影响预测结果的时间步骤，缺省为None（对应成员变量为prediction\_length）

                -   distr\_output：分布的类型，缺省为StudentTOutput

                -   batch\_normalization：是否使用BN，缺省为False

                -   mean\_scaling：缺省为True

                -   num\_parallel\_sample：缺省为100

            -   create\_transformation()：返回一个只有InstanceSplitter的Chain

            -   create\_training\_network()：以成员变量为参数（参考构造函数的参数），返回一个SimpleFeedForwardTrainingNetwork

            -   create\_predictor()：返回predictor

    -   deepar：DeepAR算法

        -   \_estimator.DeepAREstimator：

    <!-- -->

    -   waevnet：

        -   \_estimator.WaveNetEstimator：

-   kernels：

-   nursery：

-   shell：

-   support：

-   testutil：

-   time\_feature：

-   trainer：

    -   \_base.Trainer：训练器，包括关于训练的一系列参数以及训练函数，包括：

        -   \_\_init\_\_()：构造函数，传入参数

            -   ctx：CPU还是GPU

            -   epoch：缺省为100

            -   batch\_size：缺省为32

            -   num\_batchs\_per\_epoch：缺省为50

            -   learning\_rate：缺省为0.001

            -   learning\_rate\_decay\_factor：缺省为0.5

            -   patience：

            -   minimum\_learning\_rate

            -   clip\_gradient

            -   weight\_decay

            -   init

        -   \_\_call\_\_()：训练

            -   net：输入的训练网络，例如DeepARTrainingNetwork

            -   input\_names：上述网络各层的名称

            -   train\_iter：训练数据加载器，TrainDataLoader类型

            -   validation\_iter：验证数据加载器，ValidationDataLoader类型

#### forecast

model/forecast.py描述了几个forecast相关的类。Forecast是用于描述结果的形式之一

class Quantile(NamedTuple):

value: float

name: str

Quantile用于描述分位数分割点（quantile
level，指**累计分布函数**的值，参考《[[概念定义 \> 概率论 \>
分位数]{.underline}](\l)》）

-   是一个**命名元组**，用于提供两种格式来描述分割点

    -   value为介于0到1之间的float，为累计分布函数的值

    -   name则是对应的字符串，比如value为0.2，则name为"0.2"。

-   用于规范化不同的描述格式，可以通过以下成员函数得到：

    -   from\_float()：输入是0到1的浮点数

    -   from\_str()：输入是字串，可以是"p20"或者"0.20"这两种形式

    -   parse()：输入是float或str或Quantile，返回对应的Quantile对象

        class Forecast:

        model.forecast.Forecast是用于表示预测的抽象基类：

<!-- -->

-   是SampleForecast、QuantileForecast、DistributionForecast等的父类

-   包括如下成员变量：

    -   start\_date：pd.Timestamp，预测的开始时间戳

    -   freq: str，预测时间段的频率

    -   item\_id: Optional\[str\]

    -   info: Optional\[Dict\]，Forecast额外提供的信息

    -   prediction\_length: int，预测的时间长度，单位为freq

    -   mean: np.ndarray，均值

    -   index：时间索引，是一个（从start\_date开始，间隔为freq的，长为prediction\_length的）pandas.DatetimeIndex

-   包括如下成员函数：

    -   plot()：

    -   median()：参数为0.5的quantile()函数返回

    -   as\_json\_dict()：

    -   quantile\_ts()：quantile加上index的pands.Series形式

-   包括如下需子类实现的纯虚函数：

    -   quantile()：计算分位数，参数为float或者str（比如0.3,"0.3","p30",表示累计分布函数CDF的值），返回一个ndarray，形为(时间步骤,变量元数)，即CDF取到该值时，各个时间步骤的随机变量取值

    -   dim()：返回单个时间步骤下的数据维数（多变量的变量元数）

    -   copy\_dim()：返回（样本和时间步骤下）某个指定的子维度（sub-dimension）

    -   copy\_aggregate()：利用传入的agg\_fun函数，对Forecast中的数据做聚合（通常是**求和**或者**求平均值**）

        class SampleForecast(Forecast):

        SampleForecast是Forecast的子类，用样本形式来表示预测的结果。

<!-- -->

-   覆盖父类的成员变量：

    -   mean：所有样例的均值

-   除父类之外的成员变量包括：

    -   num\_samples：样本的数量

    <!-- -->

    -   samples：形状为(num\_samples,
        > prediction\_length)，或者在多变量情况下形为(num\_samples,
        > prediction\_length, target\_dim)的数组

    -   mean\_ts：mean和index组成的pandas.Series格式

<!-- -->

-   类内实现的成员函数包括：

    -   quantile()：

    -   dim()：

    -   copy\_dim()：当samples的阶（即len(sample.shape)）至少为2，第一个是各个样本，第二个是各个时间步骤，如果大于2的情况下，则返回指定第三个，否则直接返回samples

    -   copy\_aggregate()：如果样本是单元变量（即samples的阶len(sample.shape)为2），则直接返回samples，否则调用传入的agg\_fun对样例做聚合

-   覆盖父类的函数包括：

    -   as\_json\_dict()：

        class QuantileForecast(Forecast):

        QuantileForecast是Forecast的子类，用**Quantile**和**均值**来表示预测。

<!-- -->

-   除父类之外的成员变量：

    -   forecast\_arrays：ndarray类型，表示预测的array，其第一阶的维度（shape\[0\]）同forecast\_keys数组的长度

    -   forecast\_keys：List\[str\]类型，其元素为从"0.1"到"0.9"的Quantile，或者"mean"字符串，每一个对应于forecast\_arrays中的一项。

    -   \_forecast\_dict：forecat\_keys和forecast\_arrays一一对应形成的字典

    -   \_nan\_out：np.nan构成的长度为prediction\_length的ndarray

<!-- -->

-   覆盖父类的成员变量：

    -   mean：返回\_forecast\_dict\["mean"\]，没有则返回"p50"对应的分位数

<!-- -->

-   实现父类的纯虚函数：

    -   quantile()：根据分割点（quantile
        > level，float或者str）从\_forecast\_dict得到对应的分位数，如果没有则返回\_nan\_out

    -   dim()：返回数据维度

        -   如果forecast\_arrays的阶为2，则是单元变量，维度为1

        -   否则是多元变量，维度为forecast\_arrays.shape\[1\]。FIXME：因为这里forecast\_arrays的形为\[num\_samples,
            > target\_dim, prediction\_length\]?

-   覆盖父类的函数：

    -   plot()：

        class DistributionForecast(Forecast):

        DistributionForecast是Forecast的子类，根据**分布**来描述预测

<!-- -->

-   除父类之外的成员变量包括：

    -   distribution：Distribution，对应的分布

    -   mean\_ts：mean和index组成的pandas.Series格式

-   覆盖父类的成员变量包括：

    -   mean：返回distribution的mean（期望值）

<!-- -->

-   实现父类的纯虚函数：

    -   quantile()：调用distribution.quantile()实现

-   除父类之外，DistributionForecast自己的成员函数：

    -   to\_sample\_forecast()：通过distribution.sample()函数进行抽样，返回SampleForecast

#### transform

transform是用于对数据做变换的包。

Transformation是所有变换的**抽象基类**，

-   transform：对数据做变换

    -   \_base.Transformation：抽象类，所有变换的基类

        -   \_\_call\_\_()：Transformation子类的实例都是Callable的

            -   data\_it：DataEntry的迭代（Iterable）

            -   is\_train：训练还是推理，bool值

            -   返回：DataEntry的迭代（Iterable）

    -   \_base.Chain：Transformation的子类，串接各种变换，构造函数的输入是Transformation的List

    -   \_base.FlatMapTransformation：Transformation的子类，对输入数据集的每个迭代（DataEntry），迭代出0个至若干个结果（但不将之合并）

        -   \_\_call\_\_()：对输入的每个迭代（DataEntry），都迭代出0至若干个输出，具体的输出由flatmap\_transform()生成

        -   flatmap\_transform()：抽象方法，需要子类实现

            -   data：单条数据，DataEntry类型

    -   \_base.MapTransformation：抽象基类，Transformation的子类，对输入数据集的每个迭代（DataEntry），迭代一个输出

        -   \_\_call\_\_()：对输入（Iterable\[DataEntry\]）的每个迭代（DataEntry），都迭代一个输出，具体处理过程通过map\_transform实现

        -   map\_transform()：具体的处理函数，抽象方法，需要子类实现

            -   data：单条数据，DataEntry类型

            -   is\_train：训练或推理，bool类型

    -   \_base.SimpleTransformation：抽象类，MapTransformation的子类

        -   map\_transform()：实现了父类的抽象方法，实现仅忽略了参数is\_train，直接调用抽象方法transform()

        -   transform()：抽象方法，需要子类实现

            -   data：单条数据，DataEntry类型

    -   feature.AddAgeFeature：MapTransformation的子类，根据target\_field的长度（如果is\_train为False则加上pred\_length），输出从0增长的Age字段到output\_field。如果log\_scale，则output\_field字段对数增长，否则为线性增长。

        -   target\_field：DataEntry中的目标字段（即DataEntry
            > dict的key）

        -   output\_field：保存输出结果的DataEntry字段，通常"feat\_dynamic\_age"

        -   pred\_length：

        -   log\_scale：bool型，True则为对数增长，否则是线性增长

        -   map\_transform()：具体的实现函数，实现父类的抽象方法

            -   data：输入的DataEntry

            -   is\_train：bool型，如果为False则output\_field的长度要加上pred\_length

    -   feature.AddObservedValuesIndicator：SimpleTransformation的子类，检查样本target\_field字段（DataEntry的某字段，通常是numpy张量）的值是否有缺失，输出结果（同形bool张量，False表示元素的值缺失）到output\_field字段。如果convert\_nans为True，则将所有缺失元素替换为dummy\_value。

        -   target\_field：DataEntry中的目标字段（即DataEntry
            > dict的key）

        -   output\_field：保存输出结果的DataEntry字段，通常为"observed\_values"

        -   dummy\_value：用于替换的值

        -   convert\_nans：是否转换空值元素

        -   transform()：具体的实现函数，实现父类的抽象方法

    -   feature.AddTimeFeatures：

    -   split.InstanceSplitter：FlatMapTransformation的子类，

#### evaluation

##### 数据结构

class Evaluator:

##### 包结构

-   evaluation：预测结果的评价

    -   \_base.Evaluator：评价器，用于衡量预测结果的准确性

        -   quantile：分位数，缺省为0.1, 0.2, \...0.9

        -   sensonality：

        -   alpha：

        -   calculate\_owa：

        -   num\_workers：执行评估的线程数，缺省为CPU核心数

        -   chunk\_size：

        -   \_\_call\_\_()：执行评估

            -   ts\_iterator：测试数据集（包括数据和验证）

            -   fcst\_iterator：需要衡量的预测结果，Forecast的迭代

            -   num\_series：时间序列样本的数量

            -   返回tuple：

                -   dict：衡量结果（各种统计方式）

                -   pandas.DataFrame：具体每个时间序列的衡量结果

    -   \_base.MultivariateEvaluator：多尺度评价器，

    -   backtest：

        -   make\_evaluation\_prediction()：做预测/评估

            -   dataset：需要评估的**测试数据集**，类型为Dataset

            -   predictor：用于评估的预测器，类型为Predictor

            -   num\_samples：对每个时间序列给出的预测结果数量

            -   返回tuple：

                -   Forecast的迭代器：预测结果，每个迭代是一个

                -   pandas.Series的迭代器

        -   backtest\_metrics()：

实践应用
========

应用类的方向不是一个理论研究方向，而是基于某个或者若干个理论方向的实际应用、工程化、数据集、系统集成等等。

车牌识别
--------

车牌识别用到的研究方向包括目标检测（车牌检测）和OCR（车牌识别）

### 数据集

#### CCPD

Chinese City Parking Dataset，用于车牌识别的大型数据集，由中科大构建。

优点：

-   包含各种复杂环境，不同天气、车牌倾斜、模糊、过亮过暗等等

-   数据量大，包含了25万张车牌

缺点：

-   每张图片只包含了一张车牌

-   由于所有数据采集都是在合肥（中科大所在地），因此车牌以徽A居多

##### 格式

车牌的ground
truth直接标记在文件名中，文件名包含7个部分，以"-"分隔，分别是：

-   车牌与整张图片的占比

-   倾斜角度，包括水平和垂直的倾斜角度，以"\_"分隔

-   Bounding Box的左上角、右下角，以"\_"分隔

-   车牌四角位置的坐标

-   车牌号，具体格式参考官网说明

-   亮度

-   模糊度

官网：[[https://github.com/detectRecog/CCPD]{.underline}](https://github.com/detectRecog/CCPD)

### 代码：EasyPR

EasyPR是比较早期的一个中文车牌识别项目。

**代码**

[[https://github.com/liuruoze/EasyPR]{.underline}](https://github.com/liuruoze/EasyPR)

### 代码：HyperLPR

HyperLPR也是github上比较目前仍在维护

**代码**

[[https://github.com/szad670401/HyperLPR]{.underline}](https://github.com/szad670401/HyperLPR)

无人机识别
----------

无人机识别是图像分类、目标检测、视频目标跟踪、视频分类的一个具体应用。

### 数据集

无人机数据集总览：

[[https://zhuanlan.zhihu.com/p/53151892]{.underline}](https://zhuanlan.zhihu.com/p/53151892)

SSD（Stanford Drone
Dataset）：[[http://cvgl.stanford.edu/projects/uav\_data/]{.underline}](http://cvgl.stanford.edu/projects/uav_data/)

Okutama Action
Dataset：[[http://okutama-action.org/]{.underline}](http://okutama-action.org/)

视频监控异常检测
----------------

Anomaly Detection in Surveillance
Video，也不算是个研究方向，而应该是视频理解（比如动作识别、时序动作检测、时空动作检测的特化）

### Real-World Anomaly Detection in Surveillance Videos (201801) (TODO)

**论文**

[[https://arxiv.org/pdf/1801.04264.pdf]{.underline}](https://arxiv.org/pdf/1801.04264.pdf)

**参考**

[[https://zhuanlan.zhihu.com/p/34553884]{.underline}](https://zhuanlan.zhihu.com/p/34553884)

红绿灯识别
----------

### 数据集

[[https://github.com/ytzhao/Robotics/wiki/TS-and-TL-Dataset]{.underline}](https://github.com/ytzhao/Robotics/wiki/TS-and-TL-Dataset)

[[https://blog.csdn.net/weixin\_42419002/article/details/100605115\#5\_\_ApolloScape\_\_49]{.underline}](https://blog.csdn.net/weixin_42419002/article/details/100605115#5__ApolloScape__49)

#### Lara交通灯识别

巴黎的信号灯数据集

[[http://www.lara.prd.fr/benchmarks/trafficlightsrecognition]{.underline}](http://www.lara.prd.fr/benchmarks/trafficlightsrecognition)

#### WPI数据集

包括交通灯、行人和车道检测

[[http://computing.wpi.edu/dataset.html]{.underline}](http://computing.wpi.edu/dataset.html)

#### Bosch交通灯数据集

[[https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset]{.underline}](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset)

### 基于OpenCV的红绿灯识别

**参考**

[[https://zhuanlan.zhihu.com/p/93867116]{.underline}](https://zhuanlan.zhihu.com/p/93867116)

**代码**

[[https://github.com/ZhiqiangHo/code-of-csdn/tree/master/Traffic%20Light%20Detection%20using%20Python%20OpenCV]{.underline}](https://github.com/ZhiqiangHo/code-of-csdn/tree/master/Traffic Light Detection using Python OpenCV)

### 用深度学习识别交通灯

**参考**

[[https://zhuanlan.zhihu.com/p/24955921]{.underline}](https://zhuanlan.zhihu.com/p/24955921)

**代码**

元学习
======

本章介绍了和深度学习本身相关的一些研究方向

迁移学习
--------

以下是王晋东的《迁移学习简明手册》：

[[https://zhuanlan.zhihu.com/p/35352154]{.underline}](https://zhuanlan.zhihu.com/p/35352154)

[[http://jd92.wang/assets/files/transfer\_learning\_tutorial\_wjd.pdf]{.underline}](http://jd92.wang/assets/files/transfer_learning_tutorial_wjd.pdf)

### Fine-tuning

Transfer learning和Fine
tuning的区别和联系是什么？在Tensorflow语境里，Fine tuning是Transfer
learning的最后步骤，比如在预训练的图像分类CNN上迁移学习一个分类网络：

1.  去掉top（include\_top=False），将base model冻结，增加分类的Dense层

2.  训练分类的Dense层

3.  解冻base model的后面几层，训练。这步被叫做Fine tuning

知识蒸馏
--------

**参考**

[[https://zhuanlan.zhihu.com/p/81467832]{.underline}](https://zhuanlan.zhihu.com/p/81467832)

集成学习
--------

**集成学习（ensemble
learning）**，并不是一个单独的机器学习算法，而是通过构建并结合多个机器学习器来完成学习任务。集成学习往往被视为一种**元算法（meta-algorithm）**。

集成学习可以用于分类问题集成，回归问题集成，特征选取集成，异常点检测集成等等，可以说所有的机器学习领域都可以看到集成学习的身影。

对于训练集数据，我们通过训练若干个个体学习器（learner），通过一定的结合策略，就可以最终形成一个强学习器，以达到博采众长的目的。

也就是说，集成学习有两个主要的问题需要解决：

-   如何得到若干个个体学习器

-   如何选择一种结合策略，将这些个体学习器集合成一个强学习器。

而个体学习器按照个体学习器之间是否存在依赖关系可以分为两类：

-   个体学习器之间存在强依赖关系，一系列个体学习器基本都需要串行生成，代表算法是boosting系列算法

-   个体学习器之间不存在强依赖关系，一系列个体学习器可以并行生成，代表算法是**bagging**和**随机森林（Random
    Forest）**系列算法。

**参考**

常用的模型集成方法介绍：bagging、boosting 、stacking

[[https://zhuanlan.zhihu.com/p/65888174]{.underline}](https://zhuanlan.zhihu.com/p/65888174)

### Boosting

**Boosting**，指的是一系列**集成学习元算法**（**ensemble meta
algorithm**），用于减小监督式学习中误差的。

**Boosting**通过一系列**弱学习算法**（**weak
learner**，准确率较低的学习算法，可能结果只比随机分类略好）的集合来生成一个**强学习算法**（**strong
learner**）.

**参考**

#### AdaBoost

**Adaptive Boosting**，是第一个成功的Boost算法，用于二元分类。

AdaBoost方法的自适应在于：前一个分类器分错的样本会被用来训练下一个分类器。

AdaBoost方法是一种迭代算法，在每一轮中加入一个新的弱分类器，直到达到某个预定的足够小的错误率。每一个训练样本都被赋予一个权重，表明它被某个分类器选入训练集的概率。如果某个样本点已经被准确地分类，那么在构造下一个训练集中，它被选中的概率就被降低；相反，如果某个样本点没有被准确地分类，那么它的权重就得到提高。通过这样的方式，AdaBoost方法能"聚焦于"那些较难分（更富信息）的样本上。在具体实现上，最初令每个样本的权重都相等，对于第k次迭代操作，我们就根据这些权重来选取样本点，进而训练分类器Ck。然后就根据这个分类器，来提高被它分错的的样本的权重，并降低被正确分类的样本权重。然后，权重更新过的样本集被用于训练下一个分类器Ck。整个训练过程如此迭代地进行下去。

**参考**

[[http://www.uml.org.cn/sjjmwj/2019030721.asp]{.underline}](http://www.uml.org.cn/sjjmwj/2019030721.asp)

### Bagging

Bagging是Bootstrapping ，

Bootstrapping（自助法）是统计学中的一种抽样方法：利用有限的样本，经由多次重复抽样（从给定训练集中有放回的均匀抽样，也就是说，每当选中一个样本，其等可能的被再次选中并被再次添加回训练集中），建立起足以代表母体样本分布的新样本。

### 随机森林

在[机器学习]{.underline}中，**随机森林（Random
Forrest）**是一个包含多个**[决策树]{.underline}**的[分类器]{.underline}，并且其输出的类别是由个别树输出的类别的[众数]{.underline}而定。

模型压缩
--------

模型压缩可以使得模型在时间和空间两个维度上更加的节省，通常来说，模型压缩分成两个主要的方式：

-   **prune：剪枝**，改变模型的结构

-   **quantize：量化**，将float型权重改为int或binary等更为简单和易于计算的形式

### 剪枝

#### Activation Rank Filter(1607)

Activation APo Rank Filter：基于指标
APoZ（平均百分比零）的剪枝过滤器，该指标测量（卷积）图层激活中零的百分比。

Activation Mean Rank Filter：基于计算输出激活最小平均值指标的剪枝过滤器

**论文**

[[https://arxiv.org/abs/1607.03250]{.underline}](https://arxiv.org/abs/1607.03250)

#### L1Filter和L2Filter(1608)

在卷积层中具有最小 L1 权重规范和L2权重规范的剪枝过滤器（用于 Efficient
Convnets 的剪枝过滤器）。

**论文**

[[https://arxiv.org/pdf/1608.08710.pdf]{.underline}](https://arxiv.org/pdf/1608.08710.pdf)

#### Slim(1708)

通过修剪 BN 层中的缩放因子来修剪卷积层中的通道。

**论文**

[[https://arxiv.org/pdf/1708.06519.pdf]{.underline}](https://arxiv.org/pdf/1708.06519.pdf)

#### AGP(1710)

自动的逐步剪枝（是否剪枝的判断：基于对模型剪枝的效果）

**论文**

[[https://arxiv.org/pdf/1710.01878.pdf]{.underline}](https://arxiv.org/pdf/1710.01878.pdf)

#### Lottery Ticket(1803)

反复修剪模型

**论文**

[[https://arxiv.org/pdf/1803.03635.pdf]{.underline}](https://arxiv.org/pdf/1803.03635.pdf)

#### FPGM(1811)

**论文**

[[https://arxiv.org/pdf/1811.00250.pdf]{.underline}](https://arxiv.org/pdf/1811.00250.pdf)

### 量化

#### BNN (201602)

**论文**Binarized Neural Network

[[https://arxiv.org/pdf/1602.02830.pdf]{.underline}](https://arxiv.org/pdf/1602.02830.pdf)

#### DoReFa (201606)

**论文**

[[https://arxiv.org/pdf/1606.06160.pdf]{.underline}](https://arxiv.org/pdf/1606.06160.pdf)

#### QAT(2018CVPR)

**论文**

[[http://openaccess.thecvf.com/content\_cvpr\_2018/papers/Jacob\_Quantization\_and\_Training\_CVPR\_2018\_paper.pdf]{.underline}](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)

#### XGBoost

e**X**treme **G**radient **Boost**ing，

**代码**

[[https://github.com/dmlc/xgboost]{.underline}](https://github.com/dmlc/xgboost)

#### CatBoost

#### LightGBM

**代码**

[[https://github.com/Microsoft/LightGBM]{.underline}](https://github.com/Microsoft/LightGBM)

#### GBDT

AutoML
------

**AutoML，Automated Machine
Learning，自动机器学习**，是自动化应用机器学习于解决现实世界问题的过程。AutoML覆盖了从原始数据处理到可部署模型的整个过程。高等级的AutoML可以使得普通用户也可以使用机器学习。

针对某个问题的解决，传统神经网络需要人类来进行网络模型的选择，算法的选择，数据的预处理，特征值的提取和选择，超参数的优化等等操作。AutoML则希望将以上这些流程全部自动完成。

AutoML可以做的工作包括：

-   数据准备：data preparation数据预处理、清洗

-   特征工程：feature engineering，选择、构造、提取合适的特征

-   **网络架构搜索：NAS（Network Architecture Search）**

-   超参数优化：hyperparameter optimization

-   模型压缩：model compression

-   结果分析：result analysis，分析模型输出结果

其中NAS被视为相对更有技术含量的部分。

目前的AutoML系统包括：

-   Auto-sklearn: sklearn的AutoML实现

-   Cloud AutoML: 谷歌的自动机器学习系统（收费）

-   AutoKeras: Keras的AutoML实现，被视为Cloud AutoML的开源替代

-   NNI：微软推出的AutoML开源包，后端支持Pytorch、Tensorflow、sklearn等

-   AutoGluon：Amazon推出的AutoML系统

目前这些AutoML系统支持的主要任务见下表（Y表示支持）：

  -------------- ---------------- -------------- -------------- ----- --------------
                 AutoKeras        Cloud AutoML   AutoGluon      NNI   Auto sklearn
  图像分类       Y                Y ( +Edge)     Y                    
  图像回归       Y                                                    
  目标检测                        Y ( +Edge)     Y                    
  视频分类                        Y                                   
  目标追踪                        Y                                   
  文本分类       Y                Y (NLP)        Y                    
  文本回归       Y                                                    
  结构数据处理   Y (分类+回归）   Y              Y (表格预测)         
  机器翻译                        Y                                   
  -------------- ---------------- -------------- -------------- ----- --------------

这里有篇文章讨论四者的区别：

[[https://medium.com/\@santiagof/auto-is-the-new-black-google-automl-microsoft-automated-ml-autokeras-and-auto-sklearn-80d1d3c3005c]{.underline}](https://medium.com/@santiagof/auto-is-the-new-black-google-automl-microsoft-automated-ml-autokeras-and-auto-sklearn-80d1d3c3005c)

**参考**

[[https://www.jiqizhixin.com/articles/2018-11-07-18]{.underline}](https://www.jiqizhixin.com/articles/2018-11-07-18)

[[https://www.automl.org/automl/]{.underline}](https://www.automl.org/automl/)

[[https://www.jianshu.com/p/b59fe5dd1b93]{.underline}](https://www.jianshu.com/p/b59fe5dd1b93)

### AutoKeras(开源)

AutoKeras是德州A&M大学的团队研发的开源项目，是Google Cloud
AutoML的替代品。

官网：

[[https://autokeras.com/]{.underline}](https://autokeras.com/)

**论文**

Auto-Keras: An Efficient Neural Architecture Search System

[[https://arxiv.org/pdf/1806.10282.pdf]{.underline}](https://arxiv.org/pdf/1806.10282.pdf)

**代码**

[[https://github.com/keras-team/autokeras]{.underline}](https://github.com/keras-team/autokeras)

AutoKeras目前支持的任务包括：

-   Image Classification

-   Image Regression

-   Text Classification

-   Text Regression

-   Structured Data Classification

-   Structured Data Regression

#### Python包

AutoKeras的python包名为autokeras，具体结构如下：

-   autokeras

    -   auto\_model

        -   AutoModel

    -   encoder

        -   Encoder

        -   LabelEncoder

        -   OneHotEncoder

        -   serialize()

        -   deserialize()

    -   hypermodel

        -   

    -   meta\_model

        -   Assembler

        -   ImageAssembler

        -   StructuredDataAssembler

        -   TimeSeriesAssembler

        -   assemble()

    -   task

        -   SupervisedImagePipeline, SupervisedStructuredDataPipeline,
            SupervisedTextPipleline：分别是以下几个class的基类

        -   ImageClassifier

        -   ImageRegressor

        -   StructuredDataClassifier

        -   StructuredDataRegressor

        -   TextClassifier

        -   TextRegressor

        -   TimeSeriesForecaster

    -   tuner

        -   AutoTuner

        -   RandomSearch

        -   Hyperband

        -   BayesianOptimization

        -   GreedyOracle

        -   Greedy

    -   utils

### NNI（Microsoft开源）

**NNI (Neural Network
Intelligence)** 是一个轻量但强大的工具包，帮助用户自动的进行[特征工程](https://github.com/microsoft/nni/blob/master/docs/zh_CN/FeatureEngineering/Overview.md)，[神经网络架构搜索](https://github.com/microsoft/nni/blob/master/docs/zh_CN/NAS/Overview.md)，[超参调优](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Tuner/BuiltinTuner.md)以及[模型压缩](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Compressor/Overview.md)。

NNI并没有提供其他AutoML（AutoKeras、AutoGluon）所提供的具体任务实现（比如ImageClassification、TextClassification之类），而是提供了一些相对更加底层且灵感的工具。

NNI 管理自动机器学习 (AutoML) 的 Experiment，调度运行由调优算法生成的
Trial
任务来找到最好的神经网络架构和/或超参，支持各种训练环境，如[本机](https://github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/LocalMode.md)，[远程服务器](https://github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/RemoteMachineMode.md)，[OpenPAI](https://github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/PaiMode.md)，[Kubeflow](https://github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/KubeflowMode.md)，[基于
K8S 的 FrameworkController（如，AKS
等)](https://github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/FrameworkControllerMode.md)，以及其它云服务。

支持Pytorch，Tensorflow，Keras，MXNet，sklearn等框架。

文档：[[https://nni.readthedocs.io/en/latest/Overview.html]{.underline}](https://nni.readthedocs.io/en/latest/Overview.html)

中文文档：[[https://nni.readthedocs.io/zh/latest/Overview.html]{.underline}](https://nni.readthedocs.io/zh/latest/Overview.html)

**代码**

[[https://github.com/microsoft/nni]{.underline}](https://github.com/microsoft/nni)

[[https://github.com/microsoft/nni/blob/master/README\_zh\_CN.md]{.underline}](https://github.com/microsoft/nni/blob/master/README_zh_CN.md)

#### 概念

-   Trial：是将一组超参数在模型上的一次尝试，实现trial有两种方式，NNI
    API和NNI Annotation

-   Experiment：实验，实验是一次找到模型的最佳超参组合，或最好的神经网络架构的任务。
    它由 Trial 和自动机器学习算法所组成。

-   Configuration：配置是来自搜索空间的一个参数实例，每个超参都会有一个特定的值。

-   Annotation：标记，NNI的一种语法，通过加入一些注释，就可以启动NNI，完全不影响代码原先的逻辑。

-   Tuner：调参算法，Tuner从trial接收结果（metrics），评估一组网络结构/超参的性能，然后将下一组网络结构/超参发给trial。NNI有内置的tuner，也可以自定义。

-   Assessor：评估算法，为了节省资源，可通过创建Assessor来配置提前终止策略。NNI有内置的Assessor，用户也可以自定义assessor。

-   Advisor：是Tuner和Assessor的结合体

-   训练平台：是 Trial 的执行环境。 根据 Experiment
    的配置，可以是本机，远程服务器组，或其它大规模训练平台（如，OpenPAI，Kubernetes）。

#### 实现

NNI提供了两种模式来实现AutoML：

-   NNI
    API：在训练的trial文件中显式调用nni（比如通过nni.get\_next\_parameter()来得到下一组参数），需要3个文件：

    -   config.yml：主配置文件，其中定义了trial
        command（执行trial文件），以及搜索空间文件，以及其他各种配置（比如tuner、assessor的选择等）。

    -   xxxx.py：trial文件，由普通的训练主程序增加对nni接口的调用实现，实现了单次训练（一个Trial）

    -   search\_space.json：定义了超参数的搜索空间

-   NNI Annotation：在训练的主程序增加Annotation，需要2个文件：

    -   config.yml：主配置文件，其中定义了trial
        command（执行trial文件），及其他各种配置。

    -   xxxx.py：trial文件，由普通的训练主程序增加Annotation（以注释形式，因此即便没有nni环境也可以正常执行单次训练）来定义搜索空间

#### Python包

NNI的Python包名为nni：

-   nni

    -   tuner.Tuner：Tuner的基类

    -   xxx\_tuner：各内置tuner

    -   xxx\_advisor：各内置advisor

    -   assessor.Assessor：Assessor的基类

    -   xxx\_assessor：各内置assessor

    -   nas：网络架构搜索算法

        -   tensorflow：目前不支持

        -   pytorch：Pytorch的NAS

            -   base\_mutator.BaseMutator：

            -   base\_trainer.BaseTrainer：

            -   mutator.Mutator：

            -   trainer.Trainer：

            -   classic-nas.mutator.ClassicMutator：NAS算法

            -   enas：ENAS算法

                -   trainer.EnasTrainer：

                -   mutator.EnasMutator：

            -   darts：DARTS算法

                -   mutator.DartsMutator：

                -   trainer.DartsTrainer：

            -   pdarts：P-DARTS算法

                -   mutator.PdartsMutator：

                -   trainer.PdartsTrainer：

            -   random.mutator.RandomMutator

            -   spos

                -   trainer.SPOSSupernetTrainer

                -   mutator.SPOSSupernetTrainingMutator

                -   evolution.SPOSEvolution

    -   compression：模型压缩算法，主要包括**剪枝算法**和**量化算法**

        -   tensorflow：tensorflow的实现

            -   compressor：tensorflow压缩算法的抽象类

                -   Compressor：tensorflow压缩算法的基类

                -   Pruner：tensorflow剪枝算法的基类，继承Compressor

                -   Quantizer：tensorflow量化算法的基类，继承Compressor

            -   builtin\_pruners：tf内置的剪枝算法，包括LevelPruner、AGP\_Pruner和FPGMPruner

            -   builtin\_quantizers：tf内置的量化算法，包括NaiveQuantizer、QAT\_Quantizer和DoReFaQuantizer

        -   torch：Pytorch的实现

            -   compressor：Pytorch压缩算法的抽象类

                -   Compressor：Pytorch压缩算法的基类

                -   Pruner：Pytorch剪枝算法的基类，继承Compressor

                -   Quantizer：Pytorch量化算法的基类，继承Compressor

                -   QuantGrad：

            -   pruners：剪枝算法，包括LevelPruner、AGP\_Pruner、SlimPruner和LotteryTicketPruner

            -   quantizers：量化算法，包括NaiveQuantizer、QAT\_Quantizer、DoReFaQuantizer和BNNQuantizer

            -   activation\_rank\_filter\_pruners：

                -   ActivationRankFilterPruner：以下两种剪枝算法的基类

                -   ActivationAPoZRankFilterPruner

                -   ActivationMeanRankFilterPruner

            -   weight\_rank\_filter\_pruners：

                -   WeightRankFilterPruner：以下三种剪枝算法的基类

                -   L1FilterPruner：

                -   L2FilterPruner：

                -   FPGMPruner：

    -   trial：被import进nni模块，在trial文件中被调用

        -   get\_next\_parameter()：用于在NNI API模式中得到下一组超参数

        -   get\_current\_parameter()

        -   report\_intermediate\_result()

        -   report\_final\_result()

        -   get\_experiment\_id()

        -   get\_trial\_id()

        -   get\_sequence\_id()

    -   smartparam：NNI Annotation的超参数选择，被import进nni模块

        -   choice()：超参数是输入的参数之一

        -   randint()：超参数是round(uniform(low, high))的点

        -   uniform()：超参数是low和high之间均匀分布的某个值

        -   loguniform()：超参数是exp(uniform(low, high))的点

    -   feature\_engineering：特征工程

        -   feature\_selector.FeatureSelector：特征选择算法的抽象类

        -   gbdt\_selector.gbdt\_selector.GBDTSelector：

        -   gradient\_selector：

### Cloud AutoML（Google）

谷歌推出的在线AutoML服务，要钱。

官网：[[https://cloud.google.com/automl/]{.underline}](https://cloud.google.com/automl/)

目前支持的任务包括：

-   AutoML Vision

    -   Image Classification：图像分类

    -   Image Classification (Edge)：图像分类（边缘计算）

    -   Object Detection：目标检测

    -   Object Detection (Edge)：目标检测（边缘计算）

-   AutoML Video Intelligence

    -   Classification：视频分类

    -   Object Tracking：目标追踪

-   AutoML Natural Language：自然语言处理

-   AutoML Translation：机器翻译

-   AutoML Tables：结构数据处理

### AutoGluon（Amazon）

AutoGluon是亚马逊推出的AutoML框架。

官网:

[[https://autogluon.mxnet.io/]{.underline}](https://autogluon.mxnet.io/)

github:

[[https://github.com/awslabs/autogluon]{.underline}](https://github.com/awslabs/autogluon)

AutoGluon目前支持的任务包括：

-   Tabular Prediction：表格预测，根据表格的其他列，预测某个列的值

-   Image Classification：传统的图像分类任务

-   Object Detection：传统的目标检测任务

-   Text Classification：文本分类，比如情感分类

#### Python包

-   autogluon.task

    -   base

        -   base\_task

            -   BaseDataset

            -   BaseTask

        -   base\_predictor.BasePredictor

    -   image\_classification

        -   image\_classification.ImageClassification

        -   classifier.Classifier

        -   dataset

            -   RecordDataset

            -   NativeImageFolderDataset

            -   ImageFolderDataset

        -   losses：

        -   metrics：

        -   nets：

            -   Identity

            -   ConvBNRelu

            -   ResUnit

        -   pipeline：

        -   utils：

    -   object\_detection

        -   object\_detection.ObjectDetection：

        -   detector.Detector

        -   dataset：数据集载入

            -   get\_dataset()：

            -   base.DatasetBase：数据集loader的基类

            -   coco.COCO：COCO格式数据集loader

            -   voc.CustomVOCDetection：

        -   nets：

        -   pipeline：

        -   utils：

    -   tabular\_prediction：

        -   tabular\_prediction.TabularPrediction：

        -   predictor.TabularPredictor：

        -   dataset.TabularDataset：加载数据集，返回统一的数据集格式

    -   text\_classification：

        -   text\_classification.TextClassification：

        -   predictor.TextClassificationPredictor：

        -   dataset：

        -   transforms.BERTDatasetTransform

        -   network：

        -   pipeline：

-   autogluon.

### 算法：NAS

NAS（Neural Architecture Search），

#### NAS (201611)

这篇文章提出了**NAS**这个概念，即Neural Architecture
Search，通过强化学习寻找最优的网络架构，包括一个Image
Classification的卷积部分，和RNN的一个Cell（类似LSTM）。

由于现在的神经网络一般采用堆叠block的方式搭建而成，这种堆叠的超参数可以通过一个序列来表示。而这种序列的表示方式正是RNN所擅长的工作。

所以，NAS会使用一个RNN构成的控制器（controller）以概率 p随机采样一个网络结构 A，接着在CIFAR-10上训练这个网络并得到其在验证集上的精度R，然后在使用R更新控制器的参数，如此循环执行直到模型收敛，如图所示：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image435.jpeg){width="4.31875in"
height="1.9881944444444444in"}

首先我们考虑最简单的CNN，即只有卷积层构成。那么这种类型的网络是很容易用控制器来表示的。即将控制器分成N段，每一段由若干个输出，每个输出表示CNN的一个超参数，例如Filter的高，Filter的宽，横向步长，纵向步长以及Filter的数量，如图所示：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image436.jpeg){width="4.9118055555555555in"
height="1.7590277777777779in"}

NAS同样可以用来进行生成一个RNN的Cell, 传统的LSTM的计算图如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image437.jpeg){width="1.9208333333333334in"
height="2.1131944444444444in"}

和LSTM一样，NAS-RNN也需要输入一个Ct-1并输出一个Ct，并在控制器的最后两个单元中控制如何使用Ct-1以及如何计算Ct。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image438.jpeg){width="5.126388888888889in"
height="1.0930555555555554in"}

上面例子是使用"base 2"的超参作为例子进行讲解的，在实际中使用的是base
8，得到下图两个RNN单元。左侧是不包含max和sin的搜索空间，右侧是包含max和sin的搜索空间（控制器并没有选择sin）。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image439.jpeg){width="4.070138888888889in"
height="1.7666666666666666in"}

参考

[[https://zhuanlan.zhihu.com/p/52471966]{.underline}](https://zhuanlan.zhihu.com/p/52471966)

**论文**

[[https://arxiv.org/pdf/1611.01578.pdf]{.underline}](https://arxiv.org/pdf/1611.01578.pdf)

#### NASNet (201707)

**论文**

[[https://arxiv.org/pdf/1707.07012.pdf]{.underline}](https://arxiv.org/pdf/1707.07012.pdf)

#### PNASNet (201712)

**论文**

[[https://arxiv.org/pdf/1712.00559.pdf]{.underline}](https://arxiv.org/pdf/1712.00559.pdf)

#### ENAS (201802)

**论文**

Efficient Neural Architecture Search Via Parameter Sharing

[[https://arxiv.org/pdf/1802.03268.pdf]{.underline}](https://arxiv.org/pdf/1802.03268.pdf)

#### AmoebaNet (201802)

**参考**

[[https://zhuanlan.zhihu.com/p/57489362]{.underline}](https://zhuanlan.zhihu.com/p/57489362)

**论文**

[[https://arxiv.org/pdf/1802.01548.pdf]{.underline}](https://arxiv.org/pdf/1802.01548.pdf)

#### DARTS (201806)

**论文**

DARTS: Differentiable Architecture Search

[[https://arxiv.org/pdf/1806.09055.pdf]{.underline}](https://arxiv.org/pdf/1806.09055.pdf)

#### P-DARTS (201904)

**论文**

Progressive Differentiable Architecture Search: Bridging the Depth Gap
between Search and Evaluation

[[https://arxiv.org/pdf/1904.12760.pdf]{.underline}](https://arxiv.org/pdf/1904.12760.pdf)

#### EvaNet (201811)

**论文**

[[https://arxiv.org/pdf/1811.10636.pdf]{.underline}](https://arxiv.org/pdf/1811.10636.pdf)

代码：

[[https://github.com/google-research/google-research/tree/master/evanet]{.underline}](https://github.com/google-research/google-research/tree/master/evanet)

#### AssembleNet (201905)

**论文**

[[https://arxiv.org/pdf/1905.13209.pdf]{.underline}](https://arxiv.org/pdf/1905.13209.pdf)

#### TinyVideoNetworks (201910)

**论文**

[[https://arxiv.org/pdf/1910.06961.pdf]{.underline}](https://arxiv.org/pdf/1910.06961.pdf)

### 算法：特征工程

#### Gradient Selector(1908)

Gradient Feature Selector，基于梯度搜索算法的特征选择。

1.  该方法扩展了一个近期的结果，即在亚线性数据中通过展示计算能迭代的学习（即，在迷你批处理中），在线性的时间空间中的特征数量
    D 及样本大小 N。

2.  这与在搜索领域的离散到连续的放松一起，可以在非常大的数据集上进行高效、基于梯度的搜索算法。

3.  最重要的是，此算法能在特征和目标间为 N \> D 和 N \< D
    都找到高阶相关性，这与只考虑一种情况和交互式的方法所不同。

**论文**

Feature Gradients: Scalable Feature Selection via Discrete Relaxation

[[https://arxiv.org/pdf/1908.10382.pdf]{.underline}](https://arxiv.org/pdf/1908.10382.pdf)

多示例学习
----------

**多示例学习**，**Multi-Instance
Learning**，**MIL**，在20世纪90年代在机器学习领域中提出的方法。在MIL中，"包"被定义为多个示例的集合，其中"正包"中至少包含一个正示例，而"负包"中则只有负示例（此处示例的概念与样本相同，以下不区分）。MIL的目的是得到一个分类器，使得对于待测试的示例，可以得到其正负标签。

半监督学习
----------

Semi-supervised Learning

**参考**

[[https://zhuanlan.zhihu.com/p/33196506]{.underline}](https://zhuanlan.zhihu.com/p/33196506)

[[https://zhuanlan.zhihu.com/p/138085660]{.underline}](https://zhuanlan.zhihu.com/p/138085660)

硬件支持
========

由于神经网络的运算特点，使用GPU相较于CPU更加高效。传统的GPU公司NVIDIA和嵌入式芯片设计公司ARM因此都推出了一些相应的神经网络接口，以便直接对接上层的神经网络框架。

除此之外，神经网络运算和图形计算还是有些区别，比如神经网络计算对精度的要求并不高，8位精度就能满足要求（只是使用网络的时候，训练网络的时候还是需要浮点精度），因此很多公司有NPU（Neural
Processing Unit）推出，这其中包括Google设计并仅供自己使用的TPU（Tensor
Processing Unit）。

基本来说，适合运行神经网络的处理器可以分为三类：

-   **独立的处理器**：比如谷歌的TPU、Intel Nervana NNP、Mobileye Myriad
    2等

-   **基于GPU的产品**：Nvidia Tesla系列、AMD Raedon Instinct系列等

-   **协处理器**：高通Hexagon DSP、Apple Neural Engine、谷歌Pixel Visual
    Core、ARM ML Processor等，此外三星和华为的SoC都有内部的NPU协处理器

[https://en.wikipedia.org/wiki/AI\_accelerator]{.underline}

关于GPU和CPU之间的关系，有一个很形象的比喻，CPU像是一个大学教授，而GPU像是一个班的小学生，小学生自然没有大学教授厉害，但是如果要完成1000道四则运算，反倒是小学生快点。而NPU的处理单元则像是一大帮幼儿园大班的小朋友，因为只需要算5以内的加法。

GPGPU
-----

**图形处理单元上的通用计算（General-purpose computing on graphics
processing
units，GPGPU）**，是利用处理图形任务的**图形处理器**来计算原本由**中央处理器**处理的通用计算任务。这些通用计算任务通常与图形处理没有任何关系。由于现代图形处理器有强大的并行处理能力和可编程流水线，令图形处理器也可以处理非图形数据。特别是在面对**单指令流多数据流（SIMD）**且数据处理的运算量远大于数据调度和传输的需要时，通用图形处理器在性能上大大超越了传统的中央处理器应用程序。

GPGPU的概念在2012年神经网络开始复兴之前便被提出和实现，GPGPU的目的并不仅仅是为了神经网络，而是为了以一种通用的方式把GPU的强大计算能力提供出来。

OpenCL
------

OpenCL（Open Computing Language，开放计算语言）是一个为异构平台编写程序的框架，此异构平台可由CPU、GPU、DSP、FPGA或其他类型的处理器与硬件加速器所组成。OpenCL由一门用于编写kernels（在OpenCL设备上运行的函数）的语言（基于C99）和一组用于定义并控制平台的API组成。OpenCL提供了基于任务分割和数据分割的并行计算机制。

CUDA（NVIDIA）
--------------

**CUDA（Compute Unified Device Architecture，统一计算架构）**是由[NVIDIA]{.underline}所推出的一种GPGPU技术。CUDA可以兼容[OpenCL]{.underline}或者自家的C-编译器。无论是CUDA
C-语言或是OpenCL，指令最终都会被驱动程序转换成PTX代码，交由显示核心计算。

cuDNN（NVIDIA）
---------------

**CUDA Deep Neural
Network**，是NVIDIA推出的DNN的GPU加速库，cuDNN提供了专门针对类似卷积、池化、正规化等DNN计算的高性能实现。

CMSIS-NN则是其中专为神经网络的应用而提供的统一抽象接口。

TPU（Google）
-------------

**TPU（Tensor Processing
Unit，张量处理单元）**，Google于2016年5月发布的专为机器学习和TensorFlow定制的专用集成电路。第一代TPU提供高吞吐量的低精度运算（8位），面向使用或运行模型，而不是训练模型。Google宣布他们已经在数据中心中运行TPU长达一年多，发现它们对机器学习提供一个[数量级]{.underline}更优的每瓦特性能。

2017年5月，Google推出第二代TPU，并在Google Compute
Engine中使用。第二代TPU提供最高180TFLOPS的性能，组装成64个TPU的集群时提供最高11.5
PFLOPS的性能。相较于第一代，TPU
v2不仅可以计算整数，也可以计算浮点数，这使得它不仅仅可以用于使用网络，也可以被用于训练网络模型。

2018年5月，Google推出第三代TPU，Google宣布这一代TPU比上一代的性能翻倍。

Linux下Nvidia/CUDA/cuDNN的安装
------------------------------

Nvidia驱动、CUDA和cuDNN，首先搞清楚这三者的关系

1.  你得有一块NV的显卡，才需要装NVidia的驱动。

2.  在有了显卡和驱动的情况下，才可以安装CUDA（在显卡和驱动版本支持的情况下）

3.  cuDNN基于CUDA的神经网络的驱动，是在CUDA的基础上安装的

### NVidia驱动

Nvidia驱动可以从Nvidia官网下载安装，也可以从系统包安装

#### Nvidia官网

**下载**

[[https://www.nvidia.com/Download/index.aspx]{.underline}](https://www.nvidia.com/Download/index.aspx)

从官网下载对应的可执行安装程序NVIDIA-Linux-x86\_64-xxx.xx.run（64位）

**安装NVidia驱动**

service lightdm stop \#首先需要停止 X server，某些情况下需要手动杀死Xorg

./NVIDIA-Linux-x86\_64-xxx.xx.run \#执行安装脚本

service lightdm start \#启动 X server

如果装完之后出现循环登录，可以卸载之后添加**\--no-opengl-files**选项重新安装一次

**卸载NVidia驱动**

./NVIDIA-Linux-x86\_64-xxx.xx.run --uninstall \#用下载的安装程序进行卸载

或者

nvidia-uninstall \#用安装出来的可执行脚本卸载

#### Ubuntu官方

**Ubuntu安装**

apt install nvidia-xxx \#xxx为版本，例如nvidia-418

**Ubuntu卸载**

apt reomove --purge nvidia-xxx

### CUDA

CUDA同理，可以从NV官网直接下载或者通过Linux系统apt方式安装

#### NVidia官网

**下载**

[[https://developer.nvidia.com/cuda-downloads]{.underline}](https://developer.nvidia.com/cuda-downloads)

**安装CUDA**

官网提供了四种CUDA安装方式：

1.  runfile(local)：下载可执行文件到本地，运行安装

2.  deb(local)：增加一个本地的仓库，而该仓库的建立通过安装一个deb实现，通过apt
    install安装该仓库中的包

3.  deb(network)：增加一个远程的仓库，通过apt install安装该仓库中的包

4.  cluster(local)：.tar.gz文件

    **卸载CUDA**

apt purge \--autoremove cuda

#### Ubuntu官方

**Ubuntu安装**

apt install nvidia-cuda-toolkit

**Ubuntu卸载**

apt purge \--autoremove nvidia-cuda-toolkit

### CuDNN

CuDNN也是一样，安装来源可以是NV官网或者Linux系统

#### Nvidia官网

**下载CuDNN**

[[https://developer.nvidia.com/rdp/cudnn-archive]{.underline}](https://developer.nvidia.com/rdp/cudnn-archive)

下载内容包括三个，运行库，开发库，代码样例和使用说明，都是deb的形式

**安装CuDNN**

dpkg -i libcudnn7\_7.6.3.30-1+cuda10.1\_amd64.deb

#### Ubuntu官方

**安装**

apt install libcudnn7

Keras中GPU/CPU的切换
--------------------

import tensorflow as tf

from keras import backend as K

CPU\_config = tf.ConfigProto(intra\_op\_parallelism\_threads=4,\\\
inter\_op\_parallelism\_threads=4, allow\_soft\_placement=True,\\\
device\_count = {\'CPU\' : 1, \'GPU\' : 0})

GPU\_config = tf.ConfigProto(intra\_op\_parallelism\_threads=4,\\\
inter\_op\_parallelism\_threads=4, allow\_soft\_placement=True,\\\
device\_count = {\'CPU\' : 1, \'GPU\' : 1})

session = tf.Session(config=CPU\_config)\
K.set\_session(session)

Nvidia GPU
==========

图形渲染流水线
--------------

**Graphics Rendering
Pipeline（图形渲染流水线**，或者叫做**图形渲染管线）**，其主要功能是将一个三维场景（给定的虚拟相机、三维物体、光源、照明模式、纹理等），生成一副二维图像。

**图形渲染管线**类似CPU中的**指令流水线**，其使得图形渲染的过程并发，也就是说GPU中的不同部分同时工作，处理不同的图形数据。比如当第一帧的图像执行到第三阶段的时候，第二帧的数据在第二阶段处理，而第三帧的数据在第一阶段处理，因而无须等第一帧的数据全部处理完毕之后才可处理第二帧数据。毫无疑问，图形渲染管线的速度由其中最慢的那个阶段决定。

渲染管线可以被分为四个大的阶段：

-   应用程序阶段（Application Stage）

-   几何阶段（Geometry Processing Stage）

-   光栅化阶段（Rasterization Stage）

-   像素处理阶段（Pixel Processing Stage）

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image440.jpeg){width="5.7131944444444445in"
height="2.1638888888888888in"}

其中应用程序阶段的工作是由**应用程序**在CPU上完成的，而几何阶段和光栅化阶段的工作通常是在GPU上完成的。

**参考**

《Real-Time Rendering 3rd》 提炼总结

[[https://zhuanlan.zhihu.com/p/26527776]{.underline}](https://zhuanlan.zhihu.com/p/26527776)

实时渲染第四版翻译：第二章图形渲染管道

[[https://zhuanlan.zhihu.com/p/97586708]{.underline}](https://zhuanlan.zhihu.com/p/97586708)

### 应用程序阶段

**应用程序阶段（Applicaton
Stage）**一般是图形渲染管线概念上的第一个阶段。应用程序阶段是通过软件方式来实现的阶段（应用程序在CPU上执行），开发者能够对该阶段发生的情况进行完全控制，可以通过改变实现方法来改变实际性能。其他阶段，他们全部或者部分建立在硬件基础上，因此要改变实现过程会非常困难。

而与此相对的，应用程序阶段基于软件实现的一个结果是它不能被划分成一些子阶段，像图形处理，光栅化，像素处理阶段那样。然而，为了提升性能，这个阶段通常在多个处理器核心上并行的执行。

在应用程序阶段的末端，将需要在屏幕上（具体形式取决于具体输入设备）显示出来绘制的**几何体**（也就是绘制图元，rendering
primitives，如点、线、矩形等）输入到绘制管线的下一个阶段（**几何阶段**）。

### 几何阶段

**从几何阶段**（**Geometry
Stage**）主要负责大部分多边形操作和顶点操作。可以将这个阶段进一步划分成如下几个功能阶段：

-   **模型视点变换 Model & View
    Transform**：模型变换的目的是将模型变换到适合渲染的空间当中，而视图变换的目的是将摄像机放置于坐标原点，方便后续步骤的操作。

-   **顶点着色Vertex
    Shading**：顶点着色的目的在于确定模型上顶点处材质的光照效果

-   **投影Projection**：投影阶段就是将模型从三维空间投射到了二维的空间中的一个过程。投影阶段也可以理解为将视体变换到一个对角顶点分别是(-1,-1,-1)和(1,1,1)单位立方体内的过程
    。

-   **裁剪Clipping**：裁剪阶段的目的，就是对部分位于视体内部的图元进行裁剪操作

-   **屏幕映射Screen
    Mapping**：屏幕映射阶段的主要目的，就是将之前步骤得到的坐标映射到对应的屏幕坐标系上

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image441.png){width="5.708333333333333in"
height="0.8951388888888889in"}

#### 模型视点变换 (Model & View Transform)

**模型变换（Model Transform）**的目的是将模型变换到适合渲染的空间当中。

**视点变换（View
Transform）**目的就是要把相机放在原点，然后进行视点校准，使其朝向Z轴负方向，y轴指向上方，x轴指向右边。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image442.png){width="4.329861111111111in"
height="1.5048611111111112in"}

#### 顶点着色 (Vertex Shading)

确定材质上的光照效果的操作被称为**着色（shading）**。

通常，这些计算中的一些在几何阶段期间在模型的顶点上执行，而其他计算可以在每像素光栅化（per-pixel
rasterization）期间执行。可以在每个顶点处存储各种材料数据，诸如点的位置，法线，颜色或计算着色方程所需的任何其它数字信息。顶点着色的结果（其可以是颜色，向量，纹理坐标或任何其他种类的阴着色数据）计算完成后，会被发送到光栅化阶段以进行插值操作。

顶点着色还包括几个可选的子阶段，包括：

-   曲面细分
    Tessellation：想象一下你有一个弹球物体。如果你仅仅用一系列三角形来代表弹球，你会遇到质量或性能的问题。你的球或许在5米外看起来很好，但是靠近每个独立的三角形，尤其是沿着轮廓，轮廓会变得可见。如果为了改进质量你用更多的三角形代表球，当球离得很远，只在屏幕上占很少的像素，你会浪费很可观的处理时间和内存。使用曲面细分，曲面可以通过适当数量的三角形来生成。曲面细分又由三个子阶段构成，分别是**外壳着色
    Hull Shading**，**曲面细分 Tessellation**，和**域着色 Domain
    Shading**

<!-- -->

-   **几何着色 Geometry
    Shading**：着色器早于曲面细分着色器，因此在GPU上更常见。就像曲面细分着色器一样，它接收各种类型的图元然后产生新的顶点。这是一个更简单的阶段在于它创建顶点的范围是受限的并且输出图元的类型也有更多的限制。几何着色器有多种用途，其中最受欢迎的是生成粒子。想象一下模拟烟花爆炸。每个火球可以用一个顶点来代表。**几何着色器**可以把每个顶点变成正方形（由两个三角形组成），该正方形面向观察者，覆盖几个像素，因此为我们提供了一个更可信的图元用来着色。

-   **流输出 Stream
    Output**：这个阶段使我们把GPU当作几何引擎来使用。取而代之的把我们处理过的顶点发往剩余的管道渲染到屏幕上，在这里我们可以选择把这些输出到一个数组里，以便将来处理。这些数据可以被CPU或是GPU使用，在之后的过程中。此阶段通常用于粒子模拟，比如我们烟花的粒子。

    这三个阶段是以这样的顺利来执行的──**曲面细分**，**几何着色**，和**流输出**，每个都是可选的。无论哪个可选的管道被使用（也可能都没使用），如果我们继续沿着管道，我们会有一系列齐次坐标的顶点将要用来被检查相机是否能看到它们。

#### 投影 (Projection)

投影（Projection）就是将模型从三维空间投射到了二维的空间中的过程。

光照处理之后，渲染系统就开始进行投影操作，即将视体变换到一个对角顶点分别是(-1,-1,-1)和(1,1,1)单位立方体（unit
cube）内，这个单位立方体通常也被称为规范立方体（Canonical View
Volume，CVV）。

目前，主要有两种投影方法，即：

-   正交投影（orthographic projection，或称parallel projection）。

-   透视投影（perspective projection）。

#### 裁剪(Clipping)

只有当图元完全或部分存在于视体（也就是上文的规范立方体，CVV）内部的时候，才需要将其发送到光栅化阶段，**裁剪**就是对部分位于视体内部的图元进行裁剪操作。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image443.png){width="5.639583333333333in"
height="1.9229166666666666in"}

#### 屏幕映射 (Screen Mapping)

**屏幕映射（Screen
Mapping）**的主要目的，就是将之前步骤得到的坐标映射到对应的屏幕坐标系上。

进入到这个阶段时，坐标仍然是三维的（但显示状态在经过投影阶段后已经成了二维），每个图元的x和y坐标变换到了屏幕坐标系中，屏幕坐标系连同z坐标一起称为窗口坐标系。

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image444.png){width="5.558333333333334in"
height="1.851388888888889in"}

### 光栅化阶段 & 像素处理

给定经过变换和投影之后的顶点，颜色以及纹理坐标（均来自于几何阶段），给每个像素（Pixel）正确配色，以便正确绘制整幅图像。这个过个过程叫**光栅化（rasterization）**或扫描变换（scan
conversion），即从二维顶点所处的屏幕空间（所有顶点都包含Z值即深度值，及各种与相关的着色信息）到屏幕上的像素的转换。

与几何阶段相似，该阶段细分为几个功能阶段：

-   **三角形设定（Triangle
    Setup）**：这个阶段主要计算三角形的一些数据。这些数据用于**三角形遍历**阶段，也被几何阶段产生的插值的着色器数据所使用。这个阶段由固定的硬件执行。

-   **三角形遍历（Triangle
    Traversal）**：找到哪些采样点或像素在三角形中的过程。

-   **像素着色（Pixel
    Shading）**：像素着色阶段的主要目的是计算所有需逐像素计算操作的过程。

-   **融合（Merging）**：融合阶段的主要任务是**合成**当前储存于缓冲器中的由之前的像素着色阶段产生的片段颜色。此外，融合阶段还负责**可见性问题（Z缓冲相关）**的处理。

    这四个阶段又被分为两个部分：**光栅化（Rasterization）**和**像素处理（Pixel
    Processing）**：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image445.jpeg){width="5.20625in"
height="1.3680555555555556in"}

#### 三角形设定

三角形设定阶段主要用来计算三角形表面的差异和三角形表面的其他相关数据。该数据主要用于扫描转换（scan
conversion），以及由几何阶段处理的各种着色数据的插值操作所用。
该过程在专门为其设计的硬件上执行。

#### 三角形覆盖

在三角形遍历阶段将进行逐像素检查操作，检查该像素处的像素中心是否由三角形覆盖，而对于有三角形部分重合的像素，将在其重合部分生成片段（fragment）。

#### 像素着色

**像素着色（Pixel Shading）**的主要目的是计算所有需逐像素操作的过程。

所有逐像素的着色计算都在像素着色阶段进行，使用插值得来的着色数据作为输入，输出结果为一种或多种将被传送到下一阶段的颜色信息。纹理贴图操作就是在这阶段进行的。

像素着色是在可编程GPU内执行的，在这一阶段有大量的技术可以使用，其中最常见，最重要的技术之一就是**纹理贴图（Texturing）**。简单来说，纹理贴图就是将指定图片"贴"到指定物体上的过程。而指定的图片可以是一维，二维，或者三维的，其中，自然是二维图片最为常见：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image446.png){width="4.923611111111111in"
height="2.729861111111111in"}

#### 融合

每个像素的信息都储存在颜色缓冲器中，而颜色缓冲器是一个颜色的矩阵列（每种颜色包含红、绿、蓝三个分量）。融合阶段的主要任务是合成当前储存于缓冲器中的由之前的像素着色阶段产生的片段颜色。不像其它着色阶段，通常运行该阶段的GPU子单元并非完全可编程的，但其高度可配置，可支持多种特效。

此外，这个阶段还负责可见性问题的处理。这意味着当绘制完整场景的时候，颜色缓冲器中应该还包含从相机视点处可以观察到的场景图元。对于大多数图形硬件来说，这个过程是通过Z缓冲（也称深度缓冲器）算法来实现的。Z缓冲算法非常简单，具有O(n)复杂度（n是需要绘制的像素数量），只要对每个图元计算出相应的像素z值，就可以使用这种方法。

GPU的构成
---------

GPU的构成和**图形渲染流水线**基本对应，大致如下：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image447.jpeg){width="5.506944444444445in"
height="1.28125in"}

可以看到分别对应图形渲染流水线中的某个部分，其中绿色表示**高度可编程**，蓝色表示是**固定流程**，黄色则介于两者之间，虽然不可编程但是**高度可配置**。虚线表示**可选阶段**。

在早期GPU的参数中，可以看到核心结构的参数为a:b:c:d，这四个值分别是：

**顶点着色器：像素着色器：纹理单元：光栅单元**

后期的GPU中（从GeForce
8系列开始），由于统一着色器架构的出现，变成了a:b:c，三个值分别为：**流处理器：纹理单元：光栅单元**

其中**流处理器（Unified
Shader）**即统一渲染结构的着色器（顶点着色器/几何着色器/像素着色器）

### 着色器

计算机图形学领域中，**着色器**（**shader**）是一种计算机程序，原本用于进行图像的浓淡处理（计算图像中的光照、亮度、颜色等），但近来，它也被用于完成很多不同领域的工作，比如处理CG特效、进行与浓淡处理无关的[影片后期处理](https://zh.wikipedia.org/w/index.php?title=%E5%BD%B1%E7%89%87%E5%90%8E%E6%9C%9F%E5%A4%84%E7%90%86&action=edit&redlink=1)、甚至用于一些与计算机图形学无关的其它领域。

GPU的**可编程图形流水线**已经全面取代传统的固定流水线，可以使用着色器语言对其编程。构成最终图像的像素、顶点、纹理，它们的位置、色相、饱和度、亮度、对比度也都可以利用**着色器**中定义的算法进行动态调整。调用**着色器**的外部程序，也可以利用它向**着色器**提供的外部变量、纹理来修改这些**着色器**中的参数。

随着图形处理器的进步，OpenGL和Direct3D等主要的图形软件库都开始支持着色器。第一批支持着色器的
GPU
仅支持**像素着色器**，但随着开发者逐渐认识到着色器的强大，很快便出现了**顶点着色器**。2000年，第一款支持可编程像素着色器的显卡 Nvidia
GeForce 3（NV20）问世。Direct3D 10 和 OpenGL 3.2 则引入了几何着色器。

着色器可以分为**二维着色器**和**三维着色器**，目前二维着色器只有**像素着色器**一种，而三维着色器则包括**顶点着色器**和**几何着色器**等，即：

-   二维着色器

    -   像素着色器（像素处理-\>像素着色阶段）

-   三维着色器

    -   顶点着色器（几何阶段-\>顶点着色阶段）

    -   几何着色器（几何阶段-\>几何着色阶段）

    -   曲面细分着色器Tessellation Shader（几何阶段-\>曲面细分阶段）

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image448.png){width="6.111111111111111in"
height="1.3777777777777778in"}

### 统一着色器/流处理器

**Unified
Shader**，也叫**统一渲染单元**，NVidia管它叫**流处理器**（**Stream
Processor，SP**）或者**CUDA核心**（**CUDA Core**）。

微软在DirectX
10提出了统一着色器的概念，统一了各种着色器，NVidia则从GeForce
8系列（Tesla微架构）开始，使用了**统一着色器**架构，不再区分具体的着色器。

### CUDA核心

**CUDA
Core**，这是NVidia独有的定义（CUDA是NVidia的商标），即可以运行CUDA程序的核心，基本对应于通用概念中的**统一着色器/流处理器**。

### 纹理单元

也叫**纹理映射单元**（**TMU，Texture Mapping
Unit**），TMU能够旋转，调整大小和扭曲位图图像（执行纹理采样），以作为理放置在给定3D模型的任意平面上。此过程称为纹理制图映射。

### 光栅单元

**光栅单元**（**ROP，Raster Operation
Pipline**），也叫渲染输出单元（Render Output
Unit）。是GPU渲染过程的最后步骤之一。**图形管线**取像素（每个像素是一个无量纲点），和纹理像素信息，并处理它，经由特定的矩阵和向量运算，变成最终像素或深度值。此过程对应图形渲染流水线中的**光栅化阶段**。

### 光线追踪核心

**光追核**，**Ray Tracing Core（RT
Core）**，专门用于处理光线追踪运算的硬件。NVidia从RTX开始（即RTX
20系列，之前没有RT核的都叫GTX）加入了**RT核**。

### 张量核心

**Tensor Core**，是部分NVidia
GPU中提供的，用于进行矩阵运算的硬件（GeForce系列是从RTX
20系列开始使用第二代张量核心）。更多用在专用的用于进行神经网络运算的GPU中，有的没有显示输出，俗称无头卡，比如Tesla系列。

### GPU大核

**Streaming
Multiprocessor**，**SM**，也叫**GPU大核**。在NVidia不同世代的GPU中也有被叫做SMM、SMX等，有时候被**误翻译**成**流处理器**。

一个GPU中包含一个到若干个SM，比如RTX 2080 Ti包含68个SM。

一组线程（Thread Block）会被打包交给一个SM进行处理，这个Thread
Block会被切成若干Thread Warp，在SM中的执行粒度是一个Thread Warp。

SM中包含了：

-   成千上万个寄存器

-   Load/Store Units

-   各种cache：

    -   线程共享的内存

    -   L1 Cache

    -   Constant Cache

    -   纹理Cache

-   Warp调度器，用于进行Warp间的切换

-   执行单元（流处理器，CUDA核）

    -   整数和单精度浮点执行单元

    -   双精度浮点执行单元

    -   Special Funtion Unit（SFU）

-   纹理单元

-   光栅单元

-   Tensor Core（Volta、Ampere微架构）

这些不同的组成部分在NVidia每一代不同的微架构都有调整。

Pascal微架构：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image449.jpeg){width="5.574305555555555in"
height="4.042361111111111in"}

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image450.png){width="6.299305555555556in"
height="3.7180555555555554in"}

Volta微架构：

![IMG\_256](/home/jimzeus/outputs/AANN/images/media/image451.jpeg){width="3.7381944444444444in"
height="4.945833333333334in"}

**参考**

NVidia流处理器发展史

[[https://zhuanlan.zhihu.com/p/139801202]{.underline}](https://zhuanlan.zhihu.com/p/139801202)

[[https://www.zhihu.com/question/267104699/answer/320361801]{.underline}](https://www.zhihu.com/question/267104699/answer/320361801)

NVidia微架构 (Microarchitecture)
--------------------------------

目前，图形硬件正在朝**[统一着色器](https://zh.wikipedia.org/w/index.php?title=%E7%BB%9F%E4%B8%80%E7%9D%80%E8%89%B2%E5%99%A8%E6%A8%A1%E5%9E%8B&action=edit&redlink=1)架构（Unified
Shading Architecture）**发展。统一着色器模型的好处是更加灵活。

自GeForce8系列开始采用**统一着色器架构**之后，NVIDIA的GPU微架构的codename依次是：

**Tesla**（并非指产品系列Tesla）、**Fermi**、**Kepler**、**Maxwell**、**Pascal**，之后分为**Turing**（消费级产品使用）和**Volta**、**Ampere**（专业产品/图形工作站）

### Pascal架构

采用**帕斯卡微架构**的显卡包括Tesla P100、Titan X、GeForce GTX
10系列、Quadro P系列

**参考**

[[https://en.wikipedia.org/wiki/Pascal\_(microarchitecture)]{.underline}](https://en.wikipedia.org/wiki/Pascal_(microarchitecture))

### Volta架构

**伏特微架构**用于图形工作站等专业设备，采用**伏特架构**的产品包括Titan
V、Quadro GV100、Titan V CEO Edition，Tesla V100

**参考**

[[https://en.wikipedia.org/wiki/Volta\_(microarchitecture)]{.underline}](https://en.wikipedia.org/wiki/Volta_(microarchitecture))

### Turing架构

采用**图灵架构**的产品包括GeForce GTX 16系列、GeForce RTX 20系列、Quadro
RTX 系列、Tesla T4

**参考**

[[https://en.wikipedia.org/wiki/Turing\_(microarchitecture)]{.underline}](https://en.wikipedia.org/wiki/Turing_(microarchitecture))

### Ampere架构

采用**安培架构**的产品包括GeForce RTX
30系列、A100（属于之前的Tesla产品线，但不再使用Tesla这个名字）

**参考**

[[https://en.wikipedia.org/wiki/Ampere\_(microarchitecture)]{.underline}](https://en.wikipedia.org/wiki/Ampere_(microarchitecture))

NVidia产品系列及型号
--------------------

NVIDIA将显示核心分为三大系列：

-   GeForce系列：用于提供家庭娱乐

-   Quadro系列：用于专业绘图设计

-   Tesla系列：用于大规模的并联电脑运算，从基于Ampere微架构的A100开始弃用了Tesla这个系列名称，改用同微架构一样的Ampere作为产品名称

**混淆注意！**

1.  这里的Tesla是**产品系列**的名称，请注意不要和NVidia的**GPU微架构**Tesla混淆。之所以使用Tesla的原因应该是因为从Tesla微架构开始使用**统一着色器架构**。

2.  Tesla的产品先后采用了若干种微架构，比如P100（Pascal微架构），V100（Volta微架构），T4（Turing微架构）

3.  从A100（Ampere微架构）开始，不再使用Tesla作为其产品名称，而改为使用微架构名称Ampere作为产品名称。

### GeForce

GeForce系列是NVidia推出的针对桌面和移动个人电脑的GPU。其首个型号是1999年推出的GeForce
256，它之前的型号为Riva TNT 2。历年推出过的型号包括：

-   **GeForce256**：1999年推出

-   **GeForce 2系列**：2000年，包括GeForce 2 MX、GTS、Pro、Ti等型号

-   **GeForce 3系列**：2001年，包括GeForce 3、GeForce 3 Ti等型号

-   **GeForce 4系列**：2002年，包括GeForce MX 4x0、PCX 4xx0、Ti
    4xx0等型号

-   **GeForce FX系列**：2003年，包括GeForce FX 5xx0和PCX 5xx0等型号

-   **GeForce 6系列**：2004年，包括GeForce 6xx0等型号

-   **GeForce 7系列**：2005年，包括GeForce 7xx0等型号

-   **GeForce 8系列**：2006年，包括GeForce 8xx0等型号

-   **GeForce 9系列**：2008年，包括GeForce 9xx0等型号

-   **GeForce 200系列**：2008年，包括GeForce 2xx、G2xx、GTS2xx、GTX2xx

-   **GeForce 100系列**：2009年，系GeForce
    9系列重命名而成，包括G1xx、G1xxM、GT1xx、GT1xxM、GTS1xx、GTS1xxM

-   **GeForce 300系列**：2009年，也是之前的GPU重命名，包括GeForce
    3xx、3xxM、GT3xx、GT3xxM、GTS3xxM

-   **GeForce 400系列**：2010年，包括GeForce
    4xx、4xxM、GT4xx、GT4xxM、GTS4xx、GTX4xx、GTX4xxM

-   **GeForce 500系列**：2010年，包括GeForce
    5xx、GT5xx、GT5xxM、GT5xxMX、GTX5xx、GTX5xxTi、GTX5xxM

-   **GeForce 600系列**：2012年，包括GeForce GT6xx、GTX6xx、GTX6xxTi

-   **GeForce 700系列**：2013年，包括GeForce
    GT7xx、GTX7xx、GTX7xxTi、GTX Titan旗舰系列

-   **GeForce 800M系列**：2014年，包括GeForce 8xxM、GTX8xxM

-   **GeForce 900系列**：2014年，包括GeForce
    GTX9x0、GTX9x0M、GTX9x0Ti、NVidia Titan X

-   **GeForce 10系列**：2016年，包括GeForce GT10x0、GTX10x0、GTX10x0Ti

-   **GeForce 20系列**：2018年，包括GeForce RTX20x0、RTX20x0 Super、RTX
    20x0Ti、Titan RTX

-   **GeForce
    16系列**：2019年，是20系列去光线追踪功能的弱化版本，型号包括GeForce
    GTX16xx、GTX16xx Super、GTX16xxTi

-   **GeForce 30系列**：2020年，包含 GeForce RTX 30x0

#### GeForce 900系列

GeForce 900系列GPU发布于2014年，采用第二代**Maxwell微架构**。

900系列支持DirectX12，OpenGL4.6，OpenCL1.2，Vulkan1.2。下表列出的部分
900系列GPU制程工艺均为**28纳米**，显存均为**GDDR5**，总线接口均为**PCIe
3.0 x16：**

  --------------------- --------- ------------- ----------------- -------------- ------------ ---------------- ----------------------- --------- ---------- --------- ----------- ------------------ ------------------ ----- -----
  **型号**              **核心\   **晶体管数\   **核心配置**      **时钟频率**   **填充率**   **显示存储器**   **运算性能 (GFLOPS)**   **TDP\    **SLI\                                                                       
                        代号**    &\                                                                                                   （W)**    支持**                                                                       
                                  晶粒面积**                                                                                                                                                                                  

                                                                                                                                                                                                                              

                                                **核心配置\       **二级\        **默认/\     **像素\          **材质\                 **容量\   **带宽\    **接口\   **单精度\   **双精度(加速)**   **半精度(加速)**         
                                                SPs:TMUs:ROPs**   缓存\          加速\        (GP/s)**         (GT/s)**                (GB)**    (GB/s)**   带宽\     (加速)**                                                
                                                                  （MB)**        MHz**                                                                      (bit)**                                                           

                                                                                                                                                                                                                              

                                                                                                                                                                                                                              

  **GeForce GTX 950**   GM206     29.4亿\       768:48:32\        1              1024/\       32.7             49.2                    2,4       106        128       1572        49.1                                  90    2路
                                  227mm2        （6 SMM)                         1188                                                                                                                                         

  **GeForce GTX 960**                           1024:64:32\                      1127/\       39.3             72.1                    2,4       112                  2308        72.1                                  120   2路
                                                （8 SMM）                        1178                                                                                                                                         

  **GeForce GTX 970**   GM204     52亿\         1664:104:56\      1.75           1050/\       54.6             109.2                   3.5\      196\       224       3494        109                                   145   3路
                                  398mm2        （13 SMM）                       1178                                                  +0.5      +28                                                                          

  **GeForce GTX 980**                           2048:128:64\      2              1126/\       72.1             144                     4         224        256       4612        144                                   165   4路
                                                (16 SMM)                         1216                                                                                                                                         

  **GeForce GTX\        GM200     80亿\         3072:192:96\      3              1000/\       96               192                     12        336        384       6144        192                                   250   
  TITAN X**                       601mm2        (24 SMM)                         1089                                                                                                                                         
  --------------------- --------- ------------- ----------------- -------------- ------------ ---------------- ----------------------- --------- ---------- --------- ----------- ------------------ ------------------ ----- -----

#### GeForce 10系列

GeForce 10系列的GPU发布于2016年，采用了**Pascal微架构**。

支持DirectX12，OpenGL4.6，OpenCL1.2，Vulkan1.2。下表列出的部分10系列GPU制程均为**16纳米**，总线接口均为**PCIe
3.0 x16**，均支持**SLI**：

  ------------------------- --------- ------------- ----------------- -------------- ------------ ---------------- ----------------------- ---------- --------- ---------- ---------- --------- ------------- ------------------ ------------------ -----
  **型号**                  **核心\   **晶体管数\   **核心配置**      **时钟频率**   **填充率**   **显示存储器**   **运算性能 (GFLOPS)**   **TDP\                                                                                                   
                            代号**    &\                                                                                                   （W)**                                                                                                   
                                      晶粒面积**                                                                                                                                                                                                    

                                                                                                                                                                                                                                                    

                                                    **核心配置\       **二级\        **默认/\     **存储器\        **像素\                 **材质\    **容量\   **带宽\    **类型**   **接口\   **单精度\     **双精度(加速)**   **半精度(加速)**   
                                                    SPs:TMUs:ROPs**   缓存\          加速\        (MT/s)**         (GP/s)**                (GT/s)**   (GB)**    (GB/s)**              带宽\     (加速)**                                            
                                                                      （MB)**        MHz**                                                                                            (bit)**                                                       

                                                                                                                                                                                                                                                    

                                                                                                                                                                                                                                                    

  **GeForce GTX 1070**      GP104     72亿\         1920:80:48\       2              1506/\       8000             96.4                    180.7      8         256        GDDR5      256       5783\         181\               90\                150
                                      314mm2        （15 SM）                        1683                                                                                                       (6463)        (202)              (101)              

  **GeForce GTX 1070 Ti**   GP104                   2432:152:64\                     1607/\                        102.8                   244.3                                                7816\         244\               122\               180
                                                    （19 SM）                        1683                                                                                                       (8186)        (256)              (128)              

  **GeForce GTX 1080**      GP104                   2560:160:64\                     1607/1733    10000                                    257.1                320        GDDR5X               8228\         257\               128\               
                                                    （20 SM）                                                                                                                                   (8873)        (277)              (139)              

  **GeForce GTX 1080Ti**    GP102     120亿\        3584:224:88\      2.75           1480         11000            130.2                   331.5      11        484                   352       10609-11340   332-354            166-177            250
                                      471mm2        （28 SM）                                                                                                                                                                                       

  **NVIDIA TITAN X**        GP102                   3584:224:96\      3              1417         10000            136                     317.4      12        480                   384       10157\        317\               159\               
                                                    （28 SM）                                                                                                                                   (10974)       (343)              (171)              

  **NVIDIA TITAN Xp**       GP102                   3840:240:96\                     1405         11410            135                     337.2                547.7                           10790\        337\               169\               
                                                    （30 SM）                                                                                                                                   (12150)       (380)              (190)              
  ------------------------- --------- ------------- ----------------- -------------- ------------ ---------------- ----------------------- ---------- --------- ---------- ---------- --------- ------------- ------------------ ------------------ -----

#### GeForce 16系列

GeForce
16系列的GPU发布于2019年，采用**Turing微架构**，16系列在20系列之后推出，与20系列最大的区别在于取消了**RT核**和**张量核心**。

支持DirectX12，OpenGL4.6，OpenCL1.2，Vulkan1.2。下表列出的部分16系列GPU制程均为**12纳米**，总线接口均为**PCIe
3.0 x16**：

  ---------------------- --------- ------------- ----------------- -------------- ------------ ---------------- ----------------------- ---------- ---------- --------- ---------- ---------- --------- ----------- ------------------ ----------- -----
  **型号**               **核心\   **晶体管数\   **核心配置**      **时钟频率**   **填充率**   **显示存储器**   **运算性能 (GFLOPS)**   **TDP\                                                                                                     
                         代号**    &\                                                                                                   （W)**                                                                                                     
                                   晶粒面积**                                                                                                                                                                                                      

                                                                                                                                                                                                                                                   

                                                 **核心配置\       **一级\        **二级\      **默认/\         **存储器\               **像素\    **材质\    **容量\   **带宽\    **类型**   **接口\   **单精度\   **双精度(加速)**   **半精度\   
                                                 SPs:TMUs:ROPs**   缓存\          缓存\        加速\            (MT/s)**                (GP/s)**   (GT/s)**   (GB)**    (GB/s)**              带宽\     (加速)**                       (加速)**    
                                                                   （KB)**        （MB)**      MHz**                                                                                          (bit)**                                              

                                                                                                                                                                                                                                                   

                                                                                                                                                                                                                                                   

  **GTX 1650**           TU117     47亿\         896:56:32\        896            1            1485/\           8000                    53.28      93.24      4         128        GDDR5      128       2661\       83.16\             5322\       75
                                   200mm2        （14 SM）                                     1665                                                                                                     (2984)      (93.24)            (5967)      

  **GTX 1650 (GDDR6)**                                                                         1410/\           12000                   50.88      89.04                192        GDDR6                2527\       79\                5053\       
                                                                                               1590                                                                                                     (2849)      (89)               (5699)      

  **GTX 1650 (TU106)**   TU106     108亿\                                                                                                                                                                                                          90
                                   445mm2                                                                                                                                                                                                          

  **GTX 1650 (TU116)**   TU116     66亿\                                                                                                                                                                                                           75
                                   284mm2                                                                                                                                                                                                          

  **GTX 1650 Super**     TU116                   1280:80:32\       1280                        1530/\                                   55.2       110.4                                                3916\       122\               7832\       100
                                                 （20 SM)                                      1725                                                                                                     (4416)      (138)              (8832)      

  **GTX 1660**           TU116                   1408:88:48\       1408           1.5          1530/\           8000                    73         135        6                    GDDR5      192       4308\       135\               8616\       120
                                                 (22 SM)                                       1785                                                                                                     (5027)      (157)              (10053)     

  **GTX 1660 Super**                                                                                            14000                                                   336        GDDR6                                                           125

  **GTX 1660 Ti**        TU116                   1536:96:48\       1536                        1500/\           12000                   88.6       177.1                288                             4608        144                9216        120
                                                 （24 SM）                                     1770                                                                                                                                                
  ---------------------- --------- ------------- ----------------- -------------- ------------ ---------------- ----------------------- ---------- ---------- --------- ---------- ---------- --------- ----------- ------------------ ----------- -----

#### GeForce 20系列

GeForce
20系列的GPU发布于2018年，采用**Turing微架构**，具有实时光线追踪功能（Ray
Tracing），通过使用RT（Ray Tracing）核心可以加速这一过程。

支持DirectX12，OpenGL4.6，OpenCL1.2，Vulkan1.2。下表列出的部分16系列GPU制程均为**12纳米**，显存类型均为**GDDR6**，总线接口均为**PCIe
3.0 x16**

  -------------------- ------------- ----------------- -------------- ------------ ---------------- ----------------------- ------------------ ---------- ---------- --------- ---------- --------- ----------- ------------------ ----------- ---------- ---------- ----------
  **型号**             **晶体管数\   **核心配置**      **时钟频率**   **填充率**   **显示存储器**   **运算性能 (GFLOPS)**   **光线追踪性能**                                                                                                                         
                       &\                                                                                                                                                                                                                                            
                       晶粒面积**                                                                                                                                                                                                                                    

                                                                                                                                                                                                                                                                     

                                     **核心配置\       **光线\        **张量\      **二级\          **默认/\                **存储器\          **像素\    **材质\    **容量\   **带宽\    **接口\   **单精度\   **双精度(加速)**   **半精度\   **每秒\    **RTX-\    **张量\
                                     SPs:TMUs:ROPs**   追踪\          核心**       缓存\            加速\                   (MT/s)**           (GP/s)**   (GT/s)**   (GB)**    (GB/s)**   带宽\     (加速)**                       (加速)**    光线数\    OPS\       浮点\
                                                       核心**                      （MB)**          MHz**                                                                                 (bit)**                                              (十亿)**   (万亿)**   (万亿)**

                                                                                                                                                                                                                                                                     

                                                                                                                                                                                                                                                                     

  **RTX 2060**         108亿\        1920:120:48\      30             240          3                1365/\                  14000              65.52      163.8      6         336        192       5242\       164\               10483\      5          37         51.6
                       445mm2        (30 SM)                                                        1680                                                                                            (6451)      (202)              (12902)                           

  **RTX 2060 Super**                 2176:136:64\      34             272          4                1470/\                                     90.5       191.4      8         448        256       6123\       191\               12246\      6          41         57.4
                                     (34 SM)                                                        1650                                                                                            (7181)      (224)              (14362)                           

  **RTX 2070**                       2304:114:64\      36             288                           1410/\                                     90.24      203.04                                    6497\       203\               12994\                 45         59.7
                                     (36 SM)                                                        1620                                                                                            (7465)      (233)              (14930)                           

  **RTX 2070 Super**   136亿\        2560:160:64\      40             320                           1605/\                                     102.72     256.8                                     8218\       257\               16435\      7          52         72.5
                       545mm2        （40 SM）                                                      1770                                                                                            (9062       (283)              (18125)                           

  **RTX 2080**                       2944:184:64\      46             368                           1515/\                                     96.96      278.76                                    8920\       279\               17840\      8          60         80.5
                                     （46 SM）                                                      1710                                                                                            (10068)     (315)              (20137)                           

  **RTX 2080 Super**                 3072:192:64\      48             384                           1650/\                  15500              105.6      316.8                496                  10138\      317\               20275\                 63         89.2
                                     （46 SM）                                                      1815                                                                                            (11151)     (349)              (22303)                           

  **RTX 2080 Ti**      186亿\        4352:272:88\      68             544          5.5              1350/\                  14000              118.8      367.2      11        616        352       11750\      367\               23500\      10         78         107.6
                       745mm2        （68 SM)                                                       1545                                                                                            (13448)     (421)              (26896)                           

  **Titan RTX**                      4608:288:96\      72             576          6                1350/\                                     129.6      388.8      24        672        384       12442\      389\               24884\      11         84         130.5
                                     （72 SM）                                                      1770                                                                                            (16312)     (510)              (32625)                           
  -------------------- ------------- ----------------- -------------- ------------ ---------------- ----------------------- ------------------ ---------- ---------- --------- ---------- --------- ----------- ------------------ ----------- ---------- ---------- ----------

其中只有2080系列支持2路NVLink（用于连接CPU和GPU，或多个GPU之间的连接）：

#### GeForce 30系列

GeForce 30系列于2020年9月推出，采用了**Ampere微架构**。其性能是之前的RTX
20系列的两倍。

30系列用的制程是三星的**8纳米**工艺，总线接口是**PCIe 4.0 x16：**

  -------------- -------------- -------------- ---------------- ----------------------- --------- ------------ ---------- --------- ----------- ----------- ----------- ----- -----
  **型号**       **核心配置**   **时钟频率**   **显示存储器**   **运算性能 (GFLOPS)**   **TDP\    **NVLink**                                                                  
                                                                                        （W)**                                                                                

                                                                                                                                                                              

                 **SPs:TMUs**   **默认\        **加速\          **存储器\               **容量\   **带宽\      **类型**   **接口\   **单精度\   **双精度\   **半精度\         
                                （MHz）**      （MHz）**        (MT/s)**                (GB)**    (GB/s)**                带宽\     (加速)**    (加速)**    (加速)**          
                                                                                                                          (bit)**                                             

                                                                                                                                                                              

                                                                                                                                                                              

  **RTX 3070**   5888:368       1500           1730             16000                   8         512          GDDR6      256       17664\      552\        35328\      220   否
                                                                                                                                    (20372)     (637)       (40745)           

  **RTX 3080**   8704:544       1440           1710             19000                   10        760          GDDR6X     320       25068\      783\        50135\      320   否
                                                                                                                                    (29768)     (930)       (59535)           

  **RTX 3090**   10496:656      1400           1700             19500                   24        936          GDDR6X     384       29389\      918\        58778\      350   2路
                                                                                                                                    (35686)     (1115)      (71373)           
  -------------- -------------- -------------- ---------------- ----------------------- --------- ------------ ---------- --------- ----------- ----------- ----------- ----- -----

### Quadro

Quadro系列定位于**专业绘图工作站**领域，用于运行专业的CAD（Computer-Aided
Design）、CGI（Computer-Generated Imagenary）、DCC（Digital Content
Creation）。

多数产品的核心实质上与定位于个人领域的GeForce完全相同，但与GeForce相比Quadro强调与行业软件的兼容性、稳定性以及高效率。其驱动程式对行业软件及编程界面有相应的优化。

-   **Quadro**：第一代产品，基于GeForce 256，专业级图形处理器。

-   **Quadro 2**：基于GeForce 2，Quadro改良型专业级图形处理器。

-   **Quadro DCC**：基于GeForce 3，针对数位内容创作。

-   **Quadro 4**：基于GeForce 4，专业级图形处理器。

-   **Quadro
    FX**：针对台式电脑和行动工作站的专业级图形处理器，提供高效能的专业3D应用处理。例如多媒体，视像模拟等。2010年7月27日以后，Quadro
    FX系列重新划分为Quadro系列。

-   **Quadro
    NVS**：专业级商用图形处理器。提供多显示功能，例如金融交易等。2010年12月1日以后，Quadro
    NVS系列独立成为NVS系列。

-   **Quadro CX**：专为Adobe Creative Suite设计的加速器。

-   **Quadro
    VX**：专为满足中国AutoCAD专业人士的需求而设计的高性价比图形处理器。

-   **Quadro
    Plex**：针对最复杂的重度绘图和运算问题的专业级图形处理器。例如制造业设计，地球科学，数位内容创建等。

-   **Quadro RTX**：加入即时光影追踪Ray tracing技术。而且Ray
    tracing技术能够令显卡进行即时的运算，渲染与光影追踪

### Tesla

Tesla是一个新的（相较于GeForce）显示核心系列品牌，主要用于**服务器高性能电脑运算**，用于对抗AMD的FireStream系列。这是继**GeForce**和**Quadro**之后，第三个显示核心商标。

早期的Tesla分为三个子系列：

-   C：外形类似普通显卡，不设任何显示输出

-   D：Desktop，D870包含两张C870 GPU，可多个设备互联

-   S：Server，外形类似1U服务器，S870包含四张C870，可多个设备互联

下表列出了部分Tesla GPU的参数：

  ----------------------- ------------ ---------- ------------ ---------------- ------------- ----------- --------------- --------- ------------- ---------- ----------------- -------------- ----- ------- --------
  **型号**                **微架构**   **芯片**   **着色器**   **显示存储器**   **运算性能\   **CUDA\     **TDP\          **总线\                                                                           
                                                                                (GFLOPS)**    core\       (W)**           接口**                                                                            
                                                                                              version**                                                                                                     

                                                                                                                                                                                                            

                                                  **CUDA\      **默认\          **加速\       **类型**    **接口\         **容量\   **时钟\       **带宽\    **单精度\         **双精度\                    
                                                  核心\        (MHz)**          (MHz)**                   带宽（bit）**   (GB)**    频率(MHz)**   (GB/s)**   （MAD or FMA)**   (FMA)**                      
                                                  (总共)**                                                                                                                                                  

  **P4**                  Pascal       1× GP104   2560         810              1063          GDDR5       256             8         6000          192        4147--5443        129.6--170.1   6.1   50-75   PCIe

  **P6**                               1× GP104   2048         1012             1506          GDDR5       256             16        3003          192.2      6169              192.8          6.1   90      MXM

  **P40**                              1× GP102   3840         1303             1531          GDDR5       384             24        7200          345.6      10007--11758      312.7--367.4   6.1   250     PCIe

  **P100**                             1× GP100   3584         1328             1480          HBM2        4096            16        1430          732        9519--10609       4760--5304     6     300     NVLink

  **P100 (16 GB card)**                1× GP100                1126             1303                                                                         8071‒9340         4036‒4670            250     PCIe

  **P100 (12 GB card)**                                                                                   3072            12                      549        8071‒9340         4036‒4670                    

  **V100 (mezzanine)**    Volta        1× GV100   5120         N/A              1455          HBM2        4096            16/32     1750          900        14899             7450           7     300     NVLink

  **V100 (PCIe card)**                 1× GV100                N/A              1370                                                                         14028             7014                 250     PCIe

  **T4 (PCIe card)**      Turing       1× TU104   2560         585              1590          GDDR6       256             16        N/A           320        8100              N/A            7.5   70      PCIe
  ----------------------- ------------ ---------- ------------ ---------------- ------------- ----------- --------------- --------- ------------- ---------- ----------------- -------------- ----- ------- --------

API
---

### DirectX

DirectX（Direct eXtension，缩写：DX）是由微软公司创建的一系列专为多媒体以及游戏开发的应用程序接口。旗下包含Direct3D、Direct2D、DirectCompute等等多个不同用途的子部分，因为这一系列API皆以Direct字样开头，所以DirectX（只要把X字母替换为任何一个特定API的名字）就成为这一巨大的API系列的统称。目前最新版本为DirectX
12，随附于Windows 10操作系统之上。

DirectX被广泛用于Microsoft Windows、Microsoft
Xbox电子游戏开发，并且只能支持这些平台。除了游戏开发之外，DirectX亦被用于开发许多虚拟三维图形相关软件。Direct3D是DirectX中最广为应用的子模块，所以有时候这两个名词可以互相代称。

DirectX主要基于C++编程语言实现，遵循COM架构。

-   DirectX 5.0 - 雾化效果，Alpha混合

-   DirectX 6.0 - 纹理映射

-   DirectX 7.0 - 硬件T&L

-   DirectX 8.0 - Shader Model 1.1

-   DirectX 8.1 - Pixel Shader 1.4，Vertex Shader 1.1

-   DirectX 9.0 - Shader Model 2.0

-   DirectX 9.0b - Pixel Shader 2.0b，Vertex Shader 2.0

-   DirectX 9.0c - Shader Model 3.0

-   DirectX 9.0Ex- Windows Vista版本的DirectX 9.0c，Shader Model
    3.0，DXVA 1.0

-   DirectX 10- Shader Model 4.0，Windows Graphic Foundation 2.0，DXVA
    2.0

-   DirectX 10.1- Shader Model 4.1，Windows Graphic Foundation 2.1，DXVA
    2.1

-   DirectX 11- Shader Model
    5.0，Tessellation镶嵌技术，多线程渲染，计算着色器

-   DirectX 12.0 - Windows 10, low-level rendering API, GPGPU

### OpenGL

OpenGL（Open Graphics
Library，开放图形库）是用于渲染2D、3D矢量图形的跨语言、跨平台的应用程序编程接口（API）。这个接口由近350个不同的函数调用组成，用来从简单的图形比特绘制复杂的三维景象。

而另一种程序接口系统是仅用于Microsoft
Windows上的Direct3D。OpenGL常用于CAD、虚拟现实、科学可视化程序和电子游戏开发。

OpenGL的高效实现（利用图形加速硬件）存在于Windows，部分UNIX平台和Mac
OS。这些实现一般由显示设备厂商提供，而且非常依赖于该厂商提供的硬件。

-   OpenGL 1.1 - 纹理对象

-   OpenGL 1.2 - 3D纹理，BGRA压缩象素格式

-   OpenGL 1.3 - 多重渲染，多重采样，纹理压缩

-   OpenGL 1.4 - 深度纹理

-   OpenGL 1.5 - 物体顶点缓冲，遮面查询

-   OpenGL 2.0 - GLSL 1.1，多渲染目标，可编程着色语言，双面模板

-   OpenGL 2.1 - GLSL 1.2，物体像素缓冲，sRGB纹理

-   OpenGL 3.0 - GLSL 1.3，纹理阵列，条件渲染，FBO

-   OpenGL 3.1 - GLSL
    1.4，纹理缓冲对象，统一缓冲对象，符号正常化纹理，基本元素重启，实例化，拷贝缓冲接口

-   OpenGL 3.2 - GLSL
    1.5，着色器可直接处理纹理采样，改进管线可编程设计性

-   OpenGL 3.3 - GLSL 3.3，同OpenGL 4.0，大量新的ARB扩展，使OpenGL
    3.x级别硬件尽可能多的支持OpenGL 4.x级别硬件的特性

-   OpenGL 4.0 - GLSL
    4.0，两种新的着色阶段，增加渲染质量和反锯齿灵活性，数据绘图由外部API负责，加强GPU通用计算，64位双精度浮点着色器

-   OpenGL 4.1 - GLSL
    4.1，支持着色器二进制信息提取和加载，64位浮点组件支持顶点着色器输入，完全兼容于OpenGL
    ES 2.0 APIs

-   OpenGL 4.2 - GLSL
    4.2，允许多种操作的Shader同处在一个级别的纹理单元内，捕捉GPU细分几何图形，绘制多个实例用来改变反馈结果，一个32bit精度的数值可以包含多个8bit和16bit精度数值

-   OpenGL GLSL着色器4.3 -
    4.30利用GPU的并行计算，着色器存储缓冲区对象，高质量的工作/
    EAC纹理压缩，提高存储的安全，多应用鲁棒性扩展

-   OpenGL 4.4 GLSL
    4.40缓冲配置控制，高效异步查询，着色器变量布局，高效的多目标约束，流线型的Direct3D应用程序的移植，无纹理延伸，稀疏纹理扩展

-   OpenGL GLSL 4.5 - 4.50

### OpenGL ES

是OpenGL的子集，用于移动设备

### Mesa

Mesa
3D是一个在MIT许可证下开放源代码的三维计算机图形库，以开源形式实现了OpenGL的应用程序接口。

OpenGL的高效实现一般依赖于显示设备厂商提供的硬件，而Mesa
3D是一个纯基于软件的图形应用程序接口。由于许可证的原因，它只声称是一个"类似"于OpenGL的应用程序接口。

### Vulkan

Vulkan是一个低开销、跨平台的二维、三维图形与计算的应用程序接口（API），最早由科纳斯组织在2015年游戏开发者大会（GDC）上发表。与OpenGL类似，Vulkan针对全平台即时3D图形程序（如电子游戏和交互媒体）而设计，并提供高性能与更均衡的CPU与GPU占用，这也是Direct3D 12和AMD的Mantle的目标。与Direct3D（12版之前）和OpenGL的其他主要区别是，Vulkan是一个底层API，而且能执行并行任务。除此之外，Vulkan还能更好地分配多个CPU核心的使用。

边缘计算
========

本章介绍在边缘计算中，即各种手机、移动设备、嵌入式设备、物联网设备中如何使用神经网络。包括：

-   移动设备上的计算单元

-   各公司的移动计算框架

-   ARM相关介绍

达芬奇NPU（华为）(TODO)
-----------------------

寒武纪（寒武纪）
----------------

国内的AI芯片公司及其产品

TensorRT（NVIDIA）
------------------

**TensorRT**是**NVIDIA**推出的针对自己产品做的加速包。

官网：

[[https://developer.nvidia.com/tensorrt]{.underline}](https://developer.nvidia.com/tensorrt)

**参考**

[[https://zhuanlan.zhihu.com/p/64933639]{.underline}](https://zhuanlan.zhihu.com/p/64933639)

ncnn（腾讯）
------------

ncnn是腾讯开发的移动设备运行库，类似TensorFlow
Lite，但ncnn是个c++的库，只支持NDK。Ncnn库中提供了MXNet、Caffe和ONNX的模型转换到ncnn的工具。

ncnn 是一个为手机端极致优化的高性能神经网络前向计算框架。ncnn
从设计之初深刻考虑手机端的部署和使用。无第三方依赖，跨平台，手机端 cpu
的速度快于目前所有已知的开源框架。

**代码**

[[https://github.com/Tencent/ncnn]{.underline}](https://github.com/Tencent/ncnn)

TNN（腾讯）
-----------

**TNN**是**腾讯**最新推出的新一代开源移动端推理框架

**代码**

[[https://github.com/Tencent/TNN]{.underline}](https://github.com/Tencent/TNN)

MNN（阿里）
-----------

MNN是阿里推出的移动端神经网络推理框架

**代码**

[[https://github.com/alibaba/MNN]{.underline}](https://github.com/alibaba/MNN)

doc:

[[https://www.yuque.com/mnn/en/usage]{.underline}](https://www.yuque.com/mnn/en/usage)

mace（小米）
------------

**代码**

[[https://github.com/XiaoMi/mace]{.underline}](https://github.com/XiaoMi/mace)

Paddle-Lite（百度）
-------------------

**代码**

[[https://github.com/PaddlePaddle/Paddle-Lite]{.underline}](https://github.com/PaddlePaddle/Paddle-Lite)

Google
------

### TensorFlow Lite（Google）

2017年5月，Google宣布从Android
Oreo开始，提供一个专用于Android开发的软件栈TensorFlow Lite。

官网：

[https://www.tensorflow.org/lite]{.underline}

TensorFlow
Lite主要由两个组件构成，解释器（Interpreter）和转换器（Converter）。

#### 基本工作流程

1.  选择模型：选择一个自定义的TensorFlow网络模型，也可以是TF
    Lite里自带的预训练网络（https://www.tensorflow.org/lite/models）

2.  转换模型：如果是自定义的网络，使用转换器转换成TF
    Lite的格式（.tflite）

3.  部署到设备：通过TF Lite
    Interpreter在设备上运行这个模型，可以通过多种语言的API实现。

4.  优化模型：通过模型优化工具对模型进行优化：在尽量不影响准确率的情况下缩减尺寸、提高

    [https://www.tensorflow.org/lite/performance/model\_optimization]{.underline}

#### Converter

转换器（Converter）可以将一些框架下的网络模型转换成解释器可以使用的格式（.tflite文件），转换器可以通过命令行（toco，tflite\_convert）或者Python接口进行调用。

Converter可以转换几种格式的数据为tflite文件：

-   TensorFlow GraphDef（.pb）文件

-   TensorFlow SavedModel

-   Keras模型（.h5）文件

-   TensorFlow 2.0 Concrete Function

简单命令：

tflite\_convert \--keras\_model\_file=MODEL.h5
\--output\_file=MODEL.tflite

**参考**

[[https://tensorflow.google.cn/lite/convert/cmdline\_reference]{.underline}](https://tensorflow.google.cn/lite/convert/cmdline_reference)

[[https://www.tensorflow.org/lite/convert]{.underline}](https://www.tensorflow.org/lite/convert)

[https://www.tensorflow.org/lite/r2/convert]{.underline}

#### Interpreter

解释器（Interpreter）通过以下步骤来运行模型：

1.  加载模型：将描述网络模型的 .tflite文件加载到内存中

2.  数据转换：将输入数据转换为网络模型认可的格式，比如图像的缩放

3.  运行模型：

4.  解释输出：将模型输出的张量转换为应用程序可以利用的格式

解释器可以以库的形式运行在安卓、iOS和Linux上。

**参考**

[https://www.tensorflow.org/lite/guide/inference]{.underline}

#### 模型量化

TFlite和Tensorflow提供了量化（Quantization）这个方法/工具来减小模型的复杂度。

量化通过降低模型权重的精确度（从32位float到16位float，或者8位整数）来实现对推理时间、模型大小、读写时间的降低，其代价是模型准确性的下降。许多CPU等硬件提供SIMD指令的支持，可以大幅提高量化后模型的运算时间。

目前量化有两种方式：

-   训练后量化（Post-training
    quantization）：对已有的训练好的模型进行量化。

-   量化训练（Quantization-aware
    training）：由于在训练时就针对量化做优化，可以使得模型的准确性下降最小，但这只适用于部分CNN架构。

##### 训练后量化

训练后量化针对一个已经训练好的模型进行转换，有三种转换方式：

+---------------------------+-----------------+-----------------+
| 方式                      | 优化结果        | 硬件            |
+---------------------------+-----------------+-----------------+
| 权重量化                  | 模型大小变为1/4 | CPU             |
|                           |                 |                 |
| Weights quantization      | 2-3倍速度提升   |                 |
+---------------------------+-----------------+-----------------+
| 全整数量化                | 模型大小变为1/4 | CPU, Edge TPU等 |
|                           |                 |                 |
| Full integer quantization | 3倍以上速度提升 |                 |
+---------------------------+-----------------+-----------------+
| 16位浮点量化              | 模型大小变为1/2 | CPU/GPU         |
|                           |                 |                 |
| Float 16 quantization     | 潜在的GPU加速   |                 |
+---------------------------+-----------------+-----------------+

###### 权重量化：

Weights quantization，将权重量化为8位整数，代码如下：

import tensorflow as tf\
converter =
tf.lite.TFLiteConverter.from\_saved\_model(saved\_model\_dir)\
converter.optimizations = \[tf.lite.Optimize.OPTIMIZE\_FOR\_SIZE\]\
tflite\_quant\_model = converter.convert()

权重量化也可以通过执行converter时增加参数实现：

tflite\_convert \\\
  \--output\_file=/tmp/foo.tflite \\\
  \--graph\_def\_file=/tmp/some\_quantized\_graph.pb \\\
  \--inference\_type=QUANTIZED\_UINT8 \\\
  \--input\_arrays=input \\\
  \--output\_arrays=MobilenetV1/Predictions/Reshape\_1 \\\
  \--mean\_values=128 \\\
  \--std\_dev\_values=127

###### 全整数量化

Full integer quantization of weights and
activations，将权重和激活函数量化为8位整数，代码如下：

import tensorflow as tf\
\
def representative\_dataset\_gen():\
  for \_ in range(num\_calibration\_steps):\
    \# Get sample input data as a numpy array in a method of your
choosing.\
    yield \[input\]\
\
converter =
tf.lite.TFLiteConverter.from\_saved\_model(saved\_model\_dir)\
converter.optimizations = \[tf.lite.Optimize.DEFAULT\]\
converter.representative\_dataset = representative\_dataset\_gen\
tflite\_quant\_model = converter.convert()

###### 16位浮点量化

Float16 quantization of weights，将权重量化为16位浮点，代码如下：

import tensorflow as tf\
converter =
tf.lite.TFLiteConverter.from\_saved\_model(saved\_model\_dir)\
converter.optimizations = \[tf.lite.Optimize.DEFAULT\]\
converter.target\_spec.supported\_types = \[tf.lite.constants.FLOAT16\]\
tflite\_quant\_model = converter.convert()

#### 预训练模型

TF Lite官网自带了五种功能的预训练网络模型：

-   **图像分类（Image
    Classification）**：给图像分类，包括人、动物、植物、活动、地点等。模型会输出一组0到1之间的数字（和为1），表示各种类别的可能性。图像分类的缺省模型是MobileNetV1。

> （[https://www.tensorflow.org/lite/models/image\_classification/overview]{.underline}）

-   **目标检测（Object
    Detection）**：检测图像中的若干物体，用矩形框标示出位置。模型的输出为一组数据，其中每个数据由类别、得分（置信度）、位置三部分构成，位置则包括上、下、左、右四个数据。目标检测的缺省模型是用coco数据集训练的MobileNetV1。

（[https://www.tensorflow.org/lite/models/object\_detection/overview]{.underline}）

-   **智能问答（Smart
    Reply）**：智能问答可以根据聊天内容自动生成回复，而且是上下文相关的。

    （[https://www.tensorflow.org/lite/models/smart\_reply/overview]{.underline}）

-   **姿势评估（Pose
    Estimation）**：评估图像/视频中人的姿势，或者说是定位其中人的各个身体节点（包括鼻、眼、耳、肩、肘、腕、臀、膝、踝）的位置。姿势评估用的模型是PoseNet。

    （[https://www.tensorflow.org/lite/models/pose\_estimation/overview]{.underline}）

-   **图像分割（Segmentation）**：图像分割有点类似目标检测，其区别在于标示的方式不是矩形框，而是物体的准确轮廓。图像分割用的缺省模型是DeepLabV3。

    （[https://www.tensorflow.org/lite/models/segmentation/overview]{.underline}）

    其他已经在TF Lite中可以工作的预训练好的模型：

[https://www.tensorflow.org/lite/guide/hosted\_models]{.underline}

或者可以从TensorFlow Hub上自行下载需要的模型进行转换：

[https://www.tensorflow.org/hub]{.underline}

#### Transfer Learning

迁移学习（Transfer
Learning），指的是在已经训练好的网络上做适度训练，以便解决一个类似的问题，比如用于辨识小轿车的模型可以通过少量训练用于辨识卡车。迁移学习的好处是不用从头对网络进行训练，大大减少了训练时间和训练数据。

**参考**

[[https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/\#1]{.underline}](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/#1)

#### 交叉编译 label\_image 

label\_image是TF lite里自带的一个对tflite的C++使用样例，位于：

tensorflow/tensorflow/lite/examples/label\_image

对其在x86 android上进行交叉编译的步骤如下：

-   修改tensorflow/tensorflow/WORKSPACE文件，增加：

    android\_ndk\_repository(

    name = \"androidndk\", \# Required. Name \*must\* be \"androidndk\".

    path = \"/home/xxx/ndk\", \# Optional. Can be omitted if
    \`ANDROID\_NDK\_HOME\` environment variable is set.

    )

    其中path指向ndk的所在目录

-   在tensorflow/中执行编译：

    bazel build //tensorflow/lite/examples/label\_image:label\_image

    \--crosstool\_top=//external:android/crosstool \--cpu=x86

    \--host\_crosstool\_top=\@bazel\_tools//tools/cpp:toolchain
    \--cxxopt=\"-std=c++11\"

    可进行在x86
    android上的交叉编译，如果编译中出现std::round相关的错误，去掉
    std即可。

-   将程序和mobilenet网络模型上传到设备上的同一个目录下：

    -   tensorflow/tensorflow/bazel-bin/tensorflow/lite/examples/label\_image/label\_image

    -   examples/lite/examples/image\_classification/android/app/src/main/assets下的tflite网络模型文件和label文件

### NNAPI（Google）

NNAPI（Neural Network
API）是Google推出的Anroid的C语言API，用于提高神经网络在安卓设备上的计算效率。NNAPI位于上层的机器学习库（TensorFlow
Lite、Caffe2等）的下层，这些上层机器学习库在设备外搭建并训练神经网络模型，之后通过NNAPI在安卓设备上运行。

通常App不会直接调用NNAPI，而是直接使用上层的机器学习库。而这些机器学习库会通过调用NNAPI来实现加速。

基于App的需求和硬件的条件，NNAPI
runtime会机动的分配计算载荷到设备的计算部件上，这些计算部件可以是专用的NN硬件、GPU以及DSP，对于没有这些计算部件（或者驱动）的设备，NNAPI
runtime会优化代码并在CPU上执行。

![https://developer.android.com/ndk/images/nnapi/nnapi\_architecture.png](/home/jimzeus/outputs/AANN/images/media/image452.png){width="4.295833333333333in"
height="3.636111111111111in"}

NNAPI和ARMNN的相同之处在于，都是为了将上层的神经网络框架和下层的嵌入式硬件桥接起来，区别在于NNAPI仅为Android提供，下层可以由多种硬件实现。而ARMNN仅封装下层的ARM硬件，而可以供给不同的上层使用。

**参考**

[https://developer.android.com/ndk/guides/neuralnetworks]{.underline}

Facebook
--------

### Pytorch Mobile (Facebook)

Pytorch Mobile是Facebook推出的移动计算框架，支持QNNPACK和ARM CPU。

官网：

[[https://pytorch.org/mobile/home/]{.underline}](https://pytorch.org/mobile/home/)

### QNNPACK (Facebook)

Quantized Neural Network
PACKage，Facebook为移动计算优化的开源高性能内核库

**代码**

[[https://github.com/pytorch/QNNPACK]{.underline}](https://github.com/pytorch/QNNPACK)

ARM
---

ARM，Advanced RISC
Machine，英国的一家公司，及其推出的RISC体系架构处理器家族，适用于移动通信领域。ARM本身不生产销售芯片，而是销售自己的设计授权。客户可以选择：

-   IP Core授权：直接购买ARM设计的核心，并基于此设计SoC

-   指令集架构授权：基于某版本的架构自己设计核心，并基于此设计SoC。

### 指令集/架构/核心

首先来清晰一下**架构**的概念。

在谈到体系结构时，有若干种/层分类方式，非常绕人，其中有很多都可以被叫做"架构（architecture）"，从ARM这个体系结构本身，到最终手机/移动设备上的主板。而这些架构的概念，是有概念上和层次上的区分的。目前并没有一个通用的名词列表去定义这些概念，以下的名词定义和解释主要基于ARM官网和维基百科。

**混淆注意！**

1.  请注意ARMv7和ARM7的区别，前者是指令集版本（或者称为ARM架构版本），后者是指一个核心家族或者核心（Core
    Family或者Core）

2.  核心的划分被分为两个阶段，ARMv6之前官方核心是按照数字划分的（ARM1-ARM11），ARMv7之后被称为Cortex核心

下面是具体内容：

-   **RISC**和**CISC架构**，这是CPU的一个二分类方式，**RISC（Reduced
    Instruction Set
    Computing，精简指令计算机）**的代表就是ARM，此外还有MIPS、PowerPC、RISC-V、SPARC等。**CISC（Complex
    Instruction Set
    Computing，复杂指令计算机）**的代表则是x86体系结构，此外还有Motorola
    68000、PDP-10、System Z等。

-   各种体系结构，比如**ARM体系结构（ARM
    Architecture）**，这里指整个ARM处理器家族。或者著名的x86体系结构

-   **ARM架构（ARM
    Architecture）**划分，也被称为**指令集版本**或者**指令集架构**，目前版本从ARMv1到ARMv8。

    -   ARMv6之前的部分**指令集版本**又有不同的子版本，比如ARMv5有ARMv5TE、ARMv5TEJ两个子版本。

    -   **Profile**，在ARMv6之前并没有做此种划分，因此被归为Classic，ARMv7开始（以及ARMv6-M）开始将**ARM架构**分为三种，即A（Application），R（Real-time），M（Microcontroller），比如ARMv6-M，ARMv7-M，ARMv7-A，ARMv8-R，ARMv8-A等等。

-   **核心家族（Core Family）**，处理器核心的划分，分为官方（ARM
    Holdings）和第三方（比如三星、高通等）。

    -   官方核心的划分在ARMv6之前跟指令集版本部分对应，也有一定的交错（比如ARM7，ARM9、ARM11）。

        -   根据不同的配置又有细分子类，比如ARM7可分为ARM7、ARM7T、ARM7EJ三个子类

            -   T表示支持Thumb指令

            -   J表示支持Jazzele

            -   D表示支持JTAG Debug

            -   M表示支持fast Multiplier

            -   I表示支持enhanced ICE

            -   S表示Synthesizable

    -   ARMv7之后则按照Profile分为Cortex-A、Cortex-R、Cortex-M。

    -   第三方设计的Core指的是其他公司在拿到指令集架构授权之后自己设计的Core，苹果、三星、都有做第三方设计

-   **片上系统（SoC）**，这个才是真正能在手机主板上看到的封装好的芯片，功能类似PC中的主板，其中主处理器即是上条中的Core。和x86架构的PC不一样，考虑到便携性，手机等移动设备的集成度更高，一个SoC里不仅包含了处理器，还有很多外设比如DMA控制器、LCD控制器、内存控制器、摄像头接口、Flash等等。

以下是详细的**指令集版本**和不同**Core
Family**，以及所包括的**Core**的列表，可以看到两者之间相关，但也有一定交错：

  ------------ ----------------- ------------ ------------------- --------------------- ------------------------ -------------------
  指令集版本   官方Core family   官方Core     第三方Core Family   第三方Core                                     

  ARMv1        ARM1              ARM1                                                                            

  ARMv2        ARMv2             ARM2         ARM2                                                               

               ARMv2a                         ARM250              Amber (Open Source)   Amber23\                 
                                                                                        Amber25                  

                                 ARM3         ARM3                                                               

  ARMv3        ARM6              ARM60\                                                                          
                                 ARM600\                                                                         
                                 ARM610                                                                          

               ARM7              ARM7         ARM700\                                                            
                                              ARM710\                                                            
                                              ARM710a                                                            

  ARMv3        ARMv4T                         ARM7T               ARM7TDMI\                                      
                                                                  ARM710T\                                       
                                                                  ARM720T\                                       
                                                                  ARM740T\                                       
                                                                  ARM7TDMI-S                                     

                                 ARM9         ARM9T               ARM9TDMI\                                      
                                                                  ARM920T\                                       
                                                                  ARM922T\                                       
                                                                  ARM940T                                        

                                 SecureCore   SC100                                                              

               ARMv4             ARM8         ARM810              StrongARM(DEC)        SA-110\                  
                                                                                        SA-1100                  

                                                                  Faraday (Faraday)     FA510\                   
                                                                                        FA526\                   
                                                                                        FA626                    

  ARMv5        ARMv5TE           ARM10        ARM10E              ARM1020E\                                      FA606TE\
                                                                  ARM1022E                                       FA626TE\
                                                                                                                 FMP626TE\
                                                                                                                 FA726TE

                                 ARM9         ARM9E               ARM946E-S\            XScale (Intel/Marvell)   XScale\
                                                                  ARM966E-S\                                     Bulverde\
                                                                  ARM968E-S\                                     Monahans
                                                                  ARM996HS                                       

               ARMv5TEJ                                           ARM926EJ-S                                     

                                 ARM7         ARM7EJ              ARM7EJ-S                                       

                                 ARM10        ARM10E              ARM1026EJ-S                                    

  ARMv6        ARMv6             ARM11        ARM1136J(F)-S                                                      

               ARMv6T2                        ARM1156T2(F)-S                                                     

               ARMv6Z                         ARM1176JZ(F)-S                                                     

               ARMv6K                         ARM11MPCore                                                        

               ARMv6-M           SecureCore   SC000                                                              

                                 Cortex-M     Cortex-M0\                                                         
                                              Cortex-M0+\                                                        
                                              Cortex-M1                                                          

  ARMv7        ARMv7-M           SecureCore   SC300                                                              

                                 Cortex-M     Cortex-M3                                                          

               ARMv7E-M                       Cortex-M4                                                          

                                              Cortex-M7                                                          

               ARMv7-R           Cortex-R     Cortex-R4\                                                         
                                              Cortex-R5\                                                         
                                              Cortex-R7\                                                         
                                              Cortex-R8                                                          

               ARMv7-A           Cortex-A     Cortex-A\           Cortex-A5\            Snapdragon (Qualcomm)    Scorpion\
                                              (32bit)             Cortex-A7\                                     Krait
                                                                  Cortex-A8\                                     
                                                                  Cortex-A9\                                     
                                                                  Cortex-A12\                                    
                                                                  Cortex-A15\                                    
                                                                  Cortex-A17                                     

                                                                                        Ax (Apple)               A6(Swift)

  ARMv8        ARMv8-A                                            Cortex-A32            Snapdragon (Qualcomm)    Kryo

                                              Cortex-A\           Cortex-A34\           Denver (NVidia)          Denver
                                              (64bit)             Cortex-A35\                                    
                                                                  Cortex-A53\                                    
                                                                  Cortex-A57\                                    
                                                                  Cortex-A72\                                    
                                                                  Cortex-A73                                     

                                                                                        K12 (AMD)                K12

                                                                                        Exynos (Samsung)         M1/M2 (Mongoose)\
                                                                                                                 M3(Meerkat)

                                                                                        Ax (Apple)               A7(Cyclone)\
                                                                                                                 A8(Typhoon)\
                                                                                                                 A9(Twister)

               ARMv8.1-A                                                                                         A10(Hurricane)

               ARMv8.2-A                                          Cortex-A55\                                    A11(Monsoon)
                                                                  Cortex-A65E\                                   
                                                                  Cortex-A75\                                    
                                                                  Cortex-A76\                                    
                                                                  Cortex-A77                                     

                                 Neoverse     NeoverseN1\         Exynos (Samsung)      M4(Cheetah)              
                                              NeoverseE1                                                         

               ARMv8.3-A                      Ax (Apple)          A12(Vortex)                                    

               ARMv8.4-A                                          A13(Lightning)                                 

               ARMv8-M           Cortex-M     Cortex-M23\                                                        
                                              Cortex-M33\                                                        
                                              Cortex-M35P                                                        

               ARMv8-R           Cortex-R     Cortex-R52                                                         
  ------------ ----------------- ------------ ------------------- --------------------- ------------------------ -------------------

**参考**

[[https://en.wikipedia.org/wiki/List\_of\_ARM\_microarchitectures]{.underline}](https://en.wikipedia.org/wiki/List_of_ARM_microarchitectures)

[[https://en.wikipedia.org/wiki/ARM\_architecture]{.underline}](https://en.wikipedia.org/wiki/ARM_architecture)

  ---------------- ------------------------------------------------- ---------------------------------
  **特性**         **ARM V8**                                        **ARM V7**
  指令集           64位指令集 AArch64， 并且兼容32位指令集 AArch32   32位指令集 A32 和16位指令集 T16
  支持地址长度     64位                                              32位
  通用寄存器       31个 x0-x30（64位）或者 w0-w30（32位）            15个, r0-r14 (32位)
  异常模式         4层结构 EL0-EL3                                   2层结构vector table
  NEON             默认支持                                          可选支持
  LAPE             默认支持                                          可选支持
  Virtualization   默认支持                                          可选支持
  big.LITTLE       支持                                              支持
  TrustZone        默认支持                                          默认支持
  SIMD寄存器       32个 X 128位                                      32个 X 64位
  ---------------- ------------------------------------------------- ---------------------------------

#### Armv7

从V7版本后开始变成了Cortex架构。

-   Cortex-A系列:
    应用处理器，主要用于移动计算、智能手机、车载娱乐、自动驾驶、服务器、高端处理器等领域。时钟频率超过1GHZ,支持Linux、Android、Windows等完整操作系统需要的内存管理单元MMU。

-   Cortex-R系列：实时处理器，可用于无线通讯的基带控制、汽车传动系统、硬盘控制器等。时钟频率200HZ到大于1GHZ，多数不支持MMU，具有MPU、Cache和其他针对工业设计的存储器功能。响应延迟非常低，不支持完整版本的Linux和Windows，支持RTOS，

-   Cortex-M系列：微控制器处理器，时钟频率较低容易使用，应用于单片机和深度嵌入式市场。

#### Armv8

ARM
V8是ARM公司的第一款64位处理器架构，包括AArch64和AArch32二种主要执行状态。其中前者引入了一套新的指令集"A64"专门用于64位处理器，后者后者用来兼容现有的32位ARM指令集。目前我们看到的Cortex-A53,
Cortex-A57（现在被A72替代了）二款处理器便属于Cortex-A50系列，首次采用64位V8架构，是ARM在2012年下半年发布的二款产品。

**参考**

[[https://blog.csdn.net/weixin\_42325069/article/details/84070376]{.underline}](https://blog.csdn.net/weixin_42325069/article/details/84070376)

### big.LITTLE (ARM)

**big.LITTLE**是ARM公司推出的使用异种处理器的技术，"big"处理器有更好的计算性能，用于执行对性能要求高的任务，"LITTLE"处理器则更加节能，用于处理其他对性能没有高要求的任务。

Armv8中的处理器组合：

-   big处理器：Cortex-A73，Cortex-A75，Cortex-A76

-   LITTLE处理器：Cortex-A53,Cortex-A55

### ARM NN（ARM）

ARM NN（ARM Neural
Network）是ARM推出的一套开源软件/工具，使得神经网络可以高效的运行在ARM设备上，它桥接了上层的神经网络框架（TensorFlow，Caffe，TensorFlow
Lite，ONNX）和下层的ARM系列硬件（Cortex CPU、Mali GPU、ARM
ML处理器）。这样，在其它主机上训练的网络，可以通过ARM
NN在ARM硬件上无缝运行。

![https://pic4.zhimg.com/80/v2-0e82efbf7e6c0b555dfeb8bc7021e89f\_hd.jpg](/home/jimzeus/outputs/AANN/images/media/image453.jpeg){width="2.457638888888889in"
height="3.323611111111111in"}

ARM NN通过Compute
Library使用Cortex-A，Mali和ML处理器，通过CMSIS-NN使用Cortex-M处理器：

![preview](/home/jimzeus/outputs/AANN/images/media/image454.jpeg){width="3.5729166666666665in"
height="2.020138888888889in"}

ARM NN也提供对谷歌的Android NNAPI的支持：

![https://pic3.zhimg.com/80/v2-3dff358d961331e07d8b642cd581649a\_hd.jpg](/home/jimzeus/outputs/AANN/images/media/image455.jpeg){width="2.127083333333333in"
height="2.792361111111111in"}

#### ARM Compute Library

ARM Compute Library是一个专为ARM的
CPU和GPU优化过的底层函数的集合，这个函数集基于Cortex-A、Mali和ARM的ML处理器，向外提供了图像处理、计算机视觉和机器学习的功能。

#### CMSIS-NN

CMSIS（Cortex Microcontroller Software Interface
Standard）是ARM的Cortex-M系列处理器提供的具体硬件无关的HAL层接口，其基本组件是CMSIS-Core，除此之外还有CMSIS-SVD、CMSIS-RTOS、CMSIS-DSP、CMSIS-Driver、CMSIS-Packs、CMSIS-DAP。

CMSIS-NN则是其中专为神经网络的应用而提供的统一抽象接口。

### Neon(ARM)

Neon是ARM提供的一个SIMD架构，工作在Cortex-A系列和Cortex-R52处理器上

Vulkan
------

Vulkan是一个低开销、跨平台的二维、三维图形与计算的应用程序接口（API），最早由科纳斯组织在2015年游戏开发者大会（GDC）上发表。与OpenGL类似，Vulkan针对全平台即时3D图形程序（如电子游戏和交互媒体）而设计，并提供高性能与更均衡的CPU与GPU占用，这也是Direct3D 12和AMD的Mantle的目标。与Direct3D（12版之前）和OpenGL的其他主要区别是：

-   Vulkan是一个底层API，而且能执行并行任务。

-   Vulkan还能更好地分配多个CPU核心的使用。

NNIE(TODO)
----------

AidLearning
-----------

**Aid Learning
FrameWork**是一个在Android手机上构建了一个带图形界面的Linux系统，同时支持GUI，Python以及AI编程。这意味着当它安装时，你的Android手机拥有一个可以在其中运行AI程序的Linux系统。

现在Aid
Learning已经完美支持**Caffe，Tensorflow，Mxnet，ncnn，Keras，pytorch,
opencv**这些框架。此外提供了一个名为Aid\_code的AI编码开发工具。它可以通过在Aid
Learning框架上使用Python（支持Python2和Python3）来提供可视化的AI编程体验。

官网： [[http://aidlearning.net/]{.underline}](http://aidlearning.net/)

**代码**[[https://github.com/aidlearning/AidLearning-FrameWork]{.underline}](https://github.com/aidlearning/AidLearning-FrameWork)

其它
====

NN上的工作
----------

![NN\_roadmap](/home/jimzeus/outputs/AANN/images/media/image456.png){width="5.7659722222222225in"
height="3.7868055555555555in"}

神经网络类型
------------

网络类型是神经网络（NN、DNN）的最基本分类，只包含全连接层（或者叫FC层、Dense层）的网络被称为多层感知机。包含卷积层的神经网络被称为卷积神经网络。包含循环层的被称为循环神经网络。

### 多层感知机（MLP）

**多层感知器（Multilayer
Perceptron,MLP）**是一种前向结构的**[人工神经网络]{.underline}**，映射一组输入向量到一组输出向量。MLP可以被看作是一个**有向图**，由多个的节点层所组成，每一层都全连接到下一层。除了输入节点，每个节点都是一个带有非线性激活函数的神经元（或称处理单元）。一种被称为[反向传播算法]{.underline}的[监督学习]{.underline}方法常被用来训练**MLP**。 **MLP**是[感知器]{.underline}的推广，克服了感知器不能对[线性不可分]{.underline}数据进行识别的弱点。

从神经网络的角度来看，**MLP就是最基本的DNN，即都是全连接层的DNN**。

MLP在80年代的时候曾是相当流行的机器学习方法，拥有广泛的应用场景，譬如语音识别、图像识别、机器翻译等等，但自90年代以来，MLP遇到来自更为简单的支持向量机的强劲竞争。近来，由于**[深度学习]{.underline}**的成功，MLP又重新得到了关注。

### 卷积神经网络（CNN）

**卷积神经网络（Convolutional Neural
Network, CNN）**是一种**[前馈神经网络]{.underline}**，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。

卷积神经网络由一个或多个**卷积层**和顶端的**全连接层**（对应经典的神经网络）组成，同时也包括关联权重和**池化层**（pooling
layer）。这一结构使得卷积神经网络能够利用输入数据的二维结构。与其他深度学习结构相比，卷积神经网络在**图像和[语音识别]{.underline}**方面能够给出更好的结果。这一模型也可以使用[反向传播算法]{.underline}进行训练。相比较其他深度、前馈神经网络，卷积神经网络需要考量的参数更少，使之成为一种颇具吸引力的深度学习结构。

### 递归神经网络（RNN）

**递归神经网络（RNN，Recurrent Neural
Network），**也叫**循环神经网络，**是**神经网络**的一种。

RNN的定义包括两个概念，广义的定义是指宏观结构，**基础RNN**、**LSTM**和**GRU**用的都是这个宏观结构，狭义的是指**基础RNN**的微结构。

时间递归神经网络可以描述动态时间行为，因为和**前馈神经网络（feedforward
neural
network）**接受较特定结构的输入不同，RNN将状态在自身网络中循环传递，因此可以接受更广泛的时间序列结构输入。手写识别是最早成功利用RNN的研究结果。

单纯的RNN因为无法处理随着递归，权重指数级爆炸或梯度消失的问题（Vanishing
gradient
problem），难以捕捉长期时间关联；而结合不同的[LSTM]{.underline}可以很好解决这个问题。

### 生成对抗网络（GAN）

**生成对抗网络（Generative Adversarial Network，GAN）**是**非监督学习**的一种方法，通过让两个神经网络相互博弈的方式进行学习。

生成对抗网络由一个**生成网络**与一个**判别网络**组成。生成网络从潜在空间（latent
space）中随机采样作为输入，其输出结果需要尽量模仿训练集中的真实样本。判别网络的输入则为真实样本或生成网络的输出，其目的是将生成网络的输出从真实样本中尽可能分辨出来。而生成网络则要尽可能地欺骗判别网络。两个网络相互对抗、不断调整参数，最终目的是使判别网络无法判断生成网络的输出结果是否真实。

生成对抗网络常用于生成以假乱真的图片。此外，该方法还被用于生成视频、三维物体模型等。

### 受限玻尔兹曼机（RBM）

**受限玻尔兹曼机（restricted Boltzmann machine,
RBM）**是一种可通过输入数据集学习概率分布的随机生成神经网络。受限玻兹曼机在降维、分类、协同过滤、特征学习和主题建模中得到了应用。根据任务的不同，受限玻兹曼机可以使用监督学习或无监督学习的方法进行训练。

**受限玻兹曼机**是一种**玻兹曼机**的变体，但限定模型必须为二分图，由一个显层（Visible
Layer）和一个隐层（Hidden
Layer）构成，显层和隐层之间的神经元为双向全连接。

![图片包含 文字
描述已自动生成](/home/jimzeus/outputs/AANN/images/media/image457.png){width="1.9361111111111111in"
height="2.061111111111111in"}

**玻尔兹曼机（Boltzmann
machine）**是[随机神经网络]{.underline}和[递归神经网络]{.underline}的一种，由Geoffrey
Hinton和Terry Sejnowski在1985年发明。

![图片包含 物体
描述已自动生成](/home/jimzeus/outputs/AANN/images/media/image458.png){width="2.420138888888889in"
height="2.296527777777778in"}

### 深度置信网络（DBN）

**深度置信网络（Deep Belief
Network，DBN）**，DBN是一个概率生成模型，与传统的判别模型的神经网络相对，生成模型是建立一个观察数据和标签之间的联合分布，对P(Observation\|Label)和
P(Label\|Observation)都做了评估，而判别模型仅仅而已评估了后者，也就是P(Label\|Observation)。

DBN由多个**限制玻尔兹曼机**层组成。这些网络被"限制"为一个可视层和一个隐层，层间存在连接，但层内的单元间不存在连接。隐层单元被训练去捕捉在可视层表现出来的高阶数据的相关性。

神经网络可视化
--------------

神经网络的可视化大体分为三种，即模型结构本身的可视化；训练的可视化；以及特征图/卷积核的可视化。

### 模型可视化

模型可视化可以让用户更加直观的了解模型的结构

#### Netron

将NN模型可视化的开源工具，**代码**[[https://github.com/lutzroeder/netron]{.underline}](https://github.com/lutzroeder/netron)，目前支持：

1.  **ONNX** (.onnx, .pb, .pbtxt)

2.  **Keras** (.h5, .keras)

3.  **Core ML** (.mlmodel)

4.  **Caffe** (.caffemodel, .prototxt)

5.  **Caffe2** (predict\_net.pb, predict\_net.pbtxt)

6.  **MXNet** (.model, -symbol.json)

7.  **TorchScript** (.pt, .pth)

8.  **NCNN** (.param)

9.  **TensorFlow Lite** (.tflite)

部分支持：

1.  **PyTorch** (.pt, .pth)

2.  **Torch** (.t7)

3.  **CNTK** (.model, .cntk)

4.  **Deeplearning4j** (.zip)

5.  **PaddlePaddle** (.zip, \_\_model\_\_)

6.  **Darknet** (.cfg)

7.  **scikit-learn** (.pkl)

8.  **ML.NET** (.zip)

9.  **TensorFlow.js** (model.json, .pb)

10. **TensorFlow** (.pb, .meta, .pbtxt)

#### Keras

-   **model.summary()**

通过**model.summary()**可以打印出Keras模型的基本结构及参数数量，结果如下：

-   **plot\_model()**

plot\_model()函数则以一种稍微图形化一点的方式显示模型的结构，可以生成一张结构图，代码如下：

**from keras.utils.vis\_utils import plot\_model**

**plot\_model(model, to\_file="xxx.png", show\_shapes=True)**

生成的结构图：

### 训练可视化

训练可视化的目的是为了帮助更好的了解训练的情况，以更快速的训练。**Tensorflow**以及使用tensorflow作为后端的**keras**可以使用**tensorboard**进行训练的可视化。

### 卷积核/特征图可视化

卷积核和特征图的可视化是图像处理NN中会用到的一个手段。

卷积核的可视化可以看出卷积核所关心的图案，但是卷积核的可视化仅限于之前的大卷积核，在3\*3卷积核兴起之后就没有意义了（3\*3的图片能看出什么？），下面是AlexNet论文中作者对AlexNet第一层卷积核的可视化图：

![](/home/jimzeus/outputs/AANN/images/media/image459.png){width="3.8041666666666667in"
height="3.7784722222222222in"}

虽然卷积核可视化的意义在现在已经不大，特征图可视化还是有意义的，可以让用户直观的了解"神经网络都在图像中寻找什么"

[https://github.com/keplr-io/quiver]{.underline}

[https://github.com/raghakot/keras-vis]{.underline}

[https://raghakot.github.io/keras-vis/]{.underline}

相关学习资料
------------

### 视频

网易云课堂：**台大李宏毅------《机器学习》2017版**

[[https://study.163.com/course/introduction/1208946807.htm]{.underline}](https://study.163.com/course/introduction/1208946807.htm)

哔哩哔哩：**台大李宏毅------《机器学习》2020版**

[[https://www.bilibili.com/video/av94534906/]{.underline}](https://www.bilibili.com/video/av94534906/)

网易云课堂：**吴恩达------《机器学习》**

[[https://study.163.com/course/courseMain.htm?courseId=1004570029]{.underline}](https://study.163.com/course/courseMain.htm?courseId=1004570029)

### 课程

[[威斯康辛大学《机器学习导论》2020秋季课程]{.underline}](https://zhuanlan.zhihu.com/p/339180553)

### 电子书

当当云阅读：**Francois Chollet------《Python深度学习》**（付费）

[[http://e.dangdang.com/pc/reader/index.html?id=1901100618]{.underline}](http://e.dangdang.com/pc/reader/index.html?id=1901100618)

当当云阅读：**斋滕康毅------《深度学习入门》**（付费）

[[http://e.dangdang.com/pc/reader/index.html?id=1901100563]{.underline}](http://e.dangdang.com/pc/reader/index.html?id=1901100563)

**代码Ian Goodfellow、Yoshua Bengio、Aaron Courville---《Deep
Learning》中文**

[[https://github.com/exacity/deeplearningbook-chinese]{.underline}](https://github.com/exacity/deeplearningbook-chinese)

著名人士 (TODO)
---------------

### Bayers（贝叶斯）

### 马尔可夫

### Michael Jordan

### Bengio

### Hinton

### Yann LeCun

### 李飞飞

### 吴恩达

### Ian Goodfellow

### 汤晓欧

香港中文大学教授，DeepID系列论文的一作，商汤科技创始人。
