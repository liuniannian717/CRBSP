import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib as plt
# 载入数据集
data = np.load('Extract_result(1)/ZC3H7B.npy')
Features = data[:, :-1]
print(Features.shape)

Labels = data[:, -1]
print(Labels.shape)

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(Features, Labels, test_size=0.2, random_state=42)

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 构建模型
# class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True,
# intercept_scaling=1, class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’,
# verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

lr = LogisticRegression(random_state=7, tol=1e-6, penalty='l2', max_iter=80, C=10.0)  # 逻辑回归模型
# lr = LogisticRegression(random_state=7)


def muti_score(model):
    warnings.filterwarnings('ignore')
    accuracy = cross_val_score(model, Features, Labels, scoring='accuracy', cv=5)
    precision = cross_val_score(model, Features, Labels, scoring='precision', cv=5)
    recall = cross_val_score(model, Features, Labels, scoring='recall', cv=5)
    f1_score = cross_val_score(model, Features, Labels, scoring='f1', cv=5)
    auc = cross_val_score(model, Features, Labels, scoring='roc_auc', cv=5)

    print("准确率ACC:", accuracy.mean())
    print("精确率Pre:", precision.mean())
    print("召回率Recall:", recall.mean())
    print("F1_score:", f1_score.mean())
    print("AUC:", auc.mean())

model_name = ["lr"]
for name in model_name:
    model = eval(name)
    print(name)
    muti_score(model)
