# 文本表示


**文本表示：**将文本数据表示成计算机能够运算的数字或向量。

在自然语言处理（Natural Language Processing，NLP）领域，文本表示是处理流程的第一步，主要是将文本转换为计算机可以运算的数字。

这里多提一下，并不是所有人工智能（AI）模型都要做表示转换的，如计算机视觉（Compute Vision，CV）的图像识别，因为图片存储本身就是数字化的，所以直接用像素值处理就可以，这也是做CV的同学转NLP要注意的一点。

文本表示有两大类方法：

1. 离散表示
2. 分布表示

## 一、离散表示

### 1.1 独热编码（One-hot）

将语料库中所有的词拉成一个向量，给每个词一个下标，就得到对应的词典。每个分词的文本表示为该分词的比特位为1,其余位为0的矩阵表示。

```python
import pandas as pd

colors = pd.Series(["red", "green", "blue", "red"])
print(pd.get_dummies(colors,dtype="int"))
```

### 1.2 词袋模型(Bag of Words)

词袋模型把文本视为“装满单词的袋子”，忽略单词的顺序、语法和上下文，只关注每个词汇在文本中的出现频率。

例如，给定两个文档：“A rose is red, a violet is blue”和“My love is like a red, red rose”，模型会统计每个文档中单词（如“red”、“rose”、“violet”）的出现次数，并表示为向量（如(1,1,1)和(2,1,0)）

!!! tip

    [机器学习文本特征提取：CountVectorizer与TfidfVectorizer详解](https://blog.csdn.net/wyy202206174248/article/details/149070656?fromshare=blogdetail&sharetype=blogdetail&sharerId=149070656&sharerefer=PC&sharesource=weixin_41370128&sharefrom=from_link)


#### 词频统计
模型通过构建词汇表（所有文档中唯一词的集合）来实现文本到向量的转换。每个文本被表示为一个固定维度的向量，向量的每个维度对应词汇表中的一个词，值为该词在文本中的出现次数或权重。例如，使用Python的CountVectorizer工具可以实现这一过程：

CountVectorizer将文本集合转换为词频矩阵，统计每个文档中每个词的出现次数。它执行以下步骤： 

1.  分词（Tokenization）：将文本拆分为单词或n-gram  
2.  构建词汇表：收集所有文档中的所有唯一单词  
3.  生成词频矩阵：统计每个文档中每个单词的出现次数



```python
from sklearn.feature_extraction.text import CountVectorizer

# 文本数据
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

# 创建 CountVectorizer 实例
vectorizer = CountVectorizer()

# 拟合并转换文本数据
X = vectorizer.fit_transform(corpus)

# 获取特征名称
feature_names = vectorizer.get_feature_names_out()
print("Feature Names:", feature_names)

# 打印特征矩阵
print("\nFeature Matrix:\n", X.toarray())
```
!!! tip "输出结果"

    ```text
    特征名称: ['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']

    特征矩阵:
     [[0 1 1 1 0 0 1 0 1]
     [0 2 0 1 0 1 1 0 1]
     [1 0 0 1 1 0 1 1 1]
     [0 1 1 1 0 0 1 0 1]]

    ```

##### 参数详解


| 参数             | 类型          | 默认值         | 描述                                                                                           |
|------------------|---------------|----------------|------------------------------------------------------------------------------------------------|
| input            | str           | `'content'`    | 输入类型，可以是 `'filename'` 或 `'file'`。                                                      |
| encoding         | str           | `'utf-8'`      | 文件编码方式。                                                                                 |
| decode_error     | str           | `'strict'`     | 解码错误处理策略，可以设置为 `'ignore'` 或 `'replace'`。                                       |
| strip_accents    | str or None   | `None`         | 去除重音符号，可以设置为 `'ascii'` 或 `'unicode'`。                                             |
| lowercase        | bool          | `True`         | 是否转小写。                                                                                   |
| preprocessor     | callable or None | `None`       | 自定义预处理器函数。                                                                           |
| tokenizer        | callable or None | `None`       | 自定义分词器函数。                                                                             |
| analyzer         | str           | `'word'`       | 分析方法，可以设置为 `'char'` 或 `'char_wb'`（字符边界）。                                     |
| stop_words       | list, set, str or None | `None`  | 停用词列表或集合，可以设置为 `'english'` 来使用内置停用词列表。                              |
| token_pattern    | str           | `r"(?u)\b\w\w+\b"` | 正则表达式模式，用于匹配词汇。                                                                |
| ngram_range      | tuple         | `(1, 1)`       | n-grams 范围。                                                                                 |
| max_df           | float in range [0.0, 1.0] or int | `1.0`      | 最大文档频率阈值。                                                                         |
| min_df           | float in range [0.0, 1.0] or int | `1`          | 最小文档频率阈值。                                                                         |
| max_features     | int or None   | `None`         | 特征数量上限。                                                                                 |
| vocabulary       | Mapping or iterable | `None`     | 自定义词汇表。                                                                               |
| binary           | bool          | `False`        | 是否二值化。                                                                                   |
| dtype            | type          | `np.int64`     | 数据类型。                                                                                     |


### 1.3 N-Gram模型（统计语言模型）

!!! tip
    
    [N-gram 语言模型](https://blog.csdn.net/benzhujie1245com/article/details/124590836)


## 二、分布式表示

[分布式表示](https://github.com/shibing624/nlp-tutorial/blob/main/01_word_embedding/01_%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA.ipynb)