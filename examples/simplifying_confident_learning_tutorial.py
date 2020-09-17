
# coding: utf-8

# # simplified Confident Learning Tutorial
# *Author: Curtis G. Northcutt, cgn@mit.edu*

# In this tutorial, we show how to implement confident learning without using cleanlab (for the most part). 
# This tutorial is to confident learning what this tutorial
# https://pytorch.org/tutorials/beginner/examples_tensor/two_layer_net_numpy.html
# is to deep learning.

# The actual implementations in cleanlab are complex because they support parallel processing,
# numerous type and input checks, lots of hyper-parameter settings,
# lots of utilities to make things work smoothly for all types of inputs, and ancillary functions.

# I ignore all of that here and provide you a bare-bones implementation using mostly for-loops and some numpy.
# Here we'll do two simple things:
# 1. Compute the confident joint which fully characterizes all label noise.
'''上面这句话不理解。什么是 confident joint ？
confident joint 是一个变量，通常被写作为 confident_joint

Throughout these examples, you’ll see a variable called confident_joint.
The confident joint is an m x m matrix (m is the number of classes) that counts,
for every observed, noisy class, the number of examples that confidently belong to every latent, hidden class.
It counts the number of examples that we are confident are labeled correctly or incorrectly for every pair of obseved
and unobserved classes.The confident joint is an unnormalized estimate of the complete-information latent joint distribution, Ps,y.

Most of the methods in the cleanlab package start by first estimating the confident_joint.
'''
# 2. Find the indices of all label errors, ordered by likelihood of being an error.

# ## INPUT (stuff we need beforehand):
# 1. s - These are the noisy labels. This is an np.array of noisy labels, shape (n,1)
# 2. psx - These are the out-of-sample holdout predicted probabilities for every example in your dataset.
# This is an np.array (2d) of probabilities, shape (n, m)

# ## OUTPUT (what this returns):
# 1. confident_joint - an (m, m) np.array matrix characterizing all the label error counts for every pair of labels.
# 一个 numpy 矩阵表征所有的标签错误数针对每对labels。

# 2. label_errors_idx - a numpy array comprised of indices of every label error,
# ordered by likelihood of being a label error.

# In this tutorial we use the handwritten digits dataset as an example.

import cleanlab
import numpy as np
from sklearn.datasets import load_digits  # 使用sklearn 中的数据集
from sklearn.linear_model import LogisticRegression # 导入线性回归模型

# To silence convergence warnings caused by using a weak logistic regression classifier on image data
import warnings
warnings.simplefilter("ignore")
np.random.seed(477)

# STEP 0 - Get some real digits data. Add a bunch of label errors. Get probs.
# Get handwritten digits data
X = load_digits()['data']  # 拿到训练数据。 这个数据的形式是二维的，是个tuple，大小为 (1797,64)
# print(digits.data.shape) (1797, 64)
y = load_digits()['target']  # 拿到训练数据对应的标签
print('Handwritten digits datasets number of classes:', len(np.unique(y)))
print('Handwritten digits datasets number of examples:', len(y))

# Add lots of errors to labels => 这里是添加错误？
NUM_ERRORS = 100  # 设置100个错误的标签
s = np.array(y)  # 这步操作是干什么用？ => 不想修改y，从而搞了一份一模一样的 ndarray 吗？
error_indices = np.random.choice(len(s), NUM_ERRORS, replace=False)
for i in error_indices:
    # Switch to some wrong label thats a different class
    # s[i] 是个标签， np.delete(range(10),s[i]) 是<class 'numpy.ndarray'>
    # 去掉正确的标签，然后从剩余的标签中，随机选择一个标签作为错误标签。
    # np.random.choice() 随机选择一个数
    wrong_label = np.random.choice(np.delete(range(10), s[i]))
    s[i] = wrong_label  # 将标签改为错误的标签

# Confirm that we indeed added NUM_ERRORS label errors
assert (len(s) - sum(s == y) == NUM_ERRORS)
actual_label_errors = np.arange(len(y))[s != y]
print('\nIndices of actual label errors:\n', actual_label_errors)

# To keep the tutorial short, we use cleanlab to get the 
# out-of-sample predicted probabilities using cross-validation
# with a very simple, non-optimized logistic regression classifier
'''
01.使用 cleanlab 的包
02.使用一个没有优化的逻辑回归分类器 + 交叉验证，从而获得out-of-sample predicted probabilities
03.怎么理解这个 out-of-sample ？
04.latent estimate 是什么意思？
05.最后返回的就是一个概率矩阵 psx 
'''
psx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(X, s,
                                                                     clf=LogisticRegression(max_iter=1000,
                                                                                            multi_class='auto',
                                                                                            solver='lbfgs')
                                                                     )

# Now we have our noisy labels s and predicted probabilities psx.
# That's all we need for confident learning.
'''
下面就是关键，用于计算 confident_joint，也就是我不懂的地方
'''
# STEP 1 - Compute confident joint
# Verify inputs
# asarray() => convert an input to an array
s = np.asarray(s)
psx = np.asarray(psx)

# Find the number of unique classes if K is not given
# 很奇怪，为什么这里直接取len()？？  如果说某一类丢失了，怎么办？
K = len(np.unique(s))

# Estimate the probability thresholds for confident counting
# You can specify these thresholds yourself if you want
# as you may want to optimize them using a validation set.
# By default (and provably so) they are set to the average class prob.
'''
01.这里用到了列表生成表达式
02.
'''
thresholds = [np.mean(psx[:,k][s == k]) for k in range(K)] # P(s^=k|s=k)
thresholds = np.asarray(thresholds)


'''
1. 我觉得这个 confident joint 其实就是 计数矩阵
'''
# Compute confident joint
confident_joint = np.zeros((K, K), dtype = int)
for i, row in enumerate(psx):  # 这里的row 是一个item在每个标签下的概率值
    s_label = s[i] #  该item 的label
    # Find out how many classes each example is confidently labeled as
    # 找出每个example（也就是每个item）会被确切的标记为多少类
    # 为什么会叫 confident_bins 这个名字？  为什么会叫confident_joint 这个名字？
    confident_bins = row >= thresholds - 1e-6
    num_confident_bins = sum(confident_bins)
    # If more than one conf class, inc the count of the max prob class
    '''
    01.如果该item只属于一个类，则直接+=1；为什么是 np.argmax(confident_bins)???
    否则取其最大的一个+=1，也就是说相信概率大的那一个，这里又是 np.argmax(row)??    
    其实不管取哪一个，都是相同的作用，为的就是找出概率最大的那个。
    02. np.argmax(confident_bins) 方法的作用，见我git的学习笔记
    '''
    if num_confident_bins == 1:
        confident_joint[s_label][np.argmax(confident_bins)] += 1
    elif num_confident_bins > 1:
        confident_joint[s_label][np.argmax(row)] += 1

# 看一下它的size
print(confident_joint.size())
'''
01.可以看到下面这个才是 normalize，也就是做一个归一化操作
'''
# Normalize confident joint (use cleanlab, trust me on this)
confident_joint = cleanlab.latent_estimation.calibrate_confident_joint(
    confident_joint, s)

cleanlab.util.print_joint_matrix(confident_joint)


# STEP 2 - Find label errors

# We arbitrarily choose at least 5 examples left in every class.
# Regardless of whether some of them might be label errors.
MIN_NUM_PER_CLASS = 5
# Leave at least MIN_NUM_PER_CLASS examples per class.
# NOTE prune_count_matrix is transposed (relative to confident_joint)
# 可以看到上面for 循环中是先写s[label]，这个s[label]是人工标注的标签；而np.argmax(confident_bins)或者
# np.argmax(row) 则是预测标签，所以需要转置
prune_count_matrix = cleanlab.pruning.keep_at_least_n_per_class(
    prune_count_matrix=confident_joint.T,
    n=MIN_NUM_PER_CLASS,
)

'''下面这部分任务我还不是很清楚
'''
s_counts = np.bincount(s)  # 计算s中非负整数的个数
noise_masks_per_class = []
# For each row in the transposed confident joint
for k in range(K):
    noise_mask = np.zeros(len(psx), dtype=bool)  # 注意 mask 操作
    psx_k = psx[:, k]
    if s_counts[k] > MIN_NUM_PER_CLASS:  # Don't prune if not MIN_NUM_PER_CLASS
        for j in range(K):  # noisy label index (k is the true label index)
            if k != j:  # Only prune for noise rates, not diagonal entries
                num2prune = prune_count_matrix[k][j]
                if num2prune > 0:
                    # num2prune'th largest p(classk) - p(class j)
                    # for x with noisy label j
                    margin = psx_k - psx[:, j]
                    s_filter = s == j
                    threshold = -np.partition(
                        -margin[s_filter], num2prune - 1
                    )[num2prune - 1]
                    noise_mask = noise_mask | (s_filter & (margin >= threshold))
        noise_masks_per_class.append(noise_mask)
    else:
        noise_masks_per_class.append(np.zeros(len(s), dtype=bool))

# Boolean label error mask
label_errors_bool = np.stack(noise_masks_per_class).any(axis=0)

# Remove label errors if given label == model prediction
for i, pred_label in enumerate(psx.argmax(axis=1)):
    # np.all let's this work for multi_label and single label
    if label_errors_bool[i] and np.all(pred_label == s[i]):
        label_errors_bool[i] = False

# Convert boolean mask to an ordered list of indices for label errors
label_errors_idx = np.arange(len(s))[label_errors_bool]
# self confidence is the holdout probability that an example
# belongs to its given class label
self_confidence = np.array(
    [np.mean(psx[i][s[i]]) for i in label_errors_idx]
)
margin = self_confidence - psx[label_errors_bool].max(axis=1)
label_errors_idx = label_errors_idx[np.argsort(margin)]

print('Indices of label errors found by confident learning:')
print('Note label errors are sorted by likelihood of being an error')
print('but here we just sort them by index for comparison with above.')
print(np.array(sorted(label_errors_idx)))


# In[5]:
score = sum([e in label_errors_idx for e in actual_label_errors]) / NUM_ERRORS
print('% actual errors that confident learning found: {:.0%}'.format(score))
score = sum([e in actual_label_errors for e in label_errors_idx]) / len(label_errors_idx)
print('% confident learning errors that are actual errors: {:.0%}'.format(score))