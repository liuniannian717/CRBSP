import os
import nltk
# nltk.download('punkt')
import multiprocessing
from gensim.models import Word2Vec
import sys

path = os.getcwd() + '/3mer-data'          # os.getcwd() 不需要传递参数，只用于返回当前工作目录

os.listdir(path)


files = []
# r=root, d=directories, f = files

for r, d, f in os.walk(path):        #  os.walk() 函数用于在目录树中遍历所有的文件及文件夹
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

sentences = []
dictionary = dict()            # dict()函数用于创建一个字典

for i, f in enumerate(files):
    CircRNA_RBP = open(f, "r", encoding='utf-8')
    fl = CircRNA_RBP.readlines()
    for x in fl:
        tokens = nltk.word_tokenize(x)       # 分词   nltk.word_tokenize(sentences)
        sentences.append(tokens)
        for word in tokens:
            if word not in dictionary:
                dictionary[word] = 1
            else:
                dictionary[word] += 1

print("样本中存在的句子数量 " + str(len(sentences)))     # 语料库中存在的句子数量
print("唯一词的数量 " + str(len(dictionary)))          # 唯一词的数量


num_features = [32]                          #Dimensionality of the resulting word vectors  生成的词向量的维度
min_word_count = 1                           #Minimum word count threshold    最小字数阈值
num_workers = multiprocessing.cpu_count()    #Number of threads to run in parallel   并行运行的线程数
context_size = 5                             #Context window length   上下文窗口长度
seed = 1                                     #Seed for the RNG, to make the result reproducible   RNG 的种子，使结果可重现

for p in num_features:
    word2vec_model = Word2Vec(
        sentences=sentences,
        sg=1,                               # Skip-gram
        seed=seed,
        workers=num_workers,
        size=p,
        min_count=min_word_count,
        window=context_size)
    print(word2vec_model)
    word2vec_model.save('save_w2v_model/LIN28B.model')          # 保存训练模型
    # word2vec_model.wv.save_word2vec_format('save_w2v_data/37(NP)合并(101nt)50D', binary=False)  # 保存训练好的向量

