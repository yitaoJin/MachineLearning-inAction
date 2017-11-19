#coding: utf-8
import os
import pickle
import numpy as np
import operator
import matplotlib.pyplot as plt

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    # numpy的shape[n,d,row,col]
    # reshape为(10000,3,32,32) ，transpose为(10000,32,32,3)
    # astype为float型
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b,))
    X, Y = load_CIFAR_batch(f)
    #把5个batch整合起来
    xs.append(X)
    ys.append(Y)
  #最终的Xtr为（50000，32，32，3）
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

# Load the raw CIFAR-10 data.
cifar10_dir = '/Users/jinyitao/Desktop/机器学习相关/机器学习测试数据集/cifar10_py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

'''
# 显示一部分图片
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, className in enumerate(classes):
    idxs = np.flatnonzero(y_train == y) #返回符合要求的序号
    idxs = np.random.choice(idxs, samples_per_class, replace=False) #在符合要求的label中随机选择序号
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
           plt.title(className)
plt.savefig("/Users/jinyitao/Desktop/机器学习相关/《机器学习实战》/myProject/1_knn/cifar10.jpg")
plt.show()
'''


from KNearestNeighbor import KNearestNeighbor


#缩小数据规模

num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 1000
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]
'''
# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
dists = classifier.compute_distances_no_loops(X_test)
y_test_pred = classifier.predict_labels(dists, k=3)
num_correct = np.sum(y_test_pred == y_test)
num_test = X_test.shape[0]
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
'''

# Let's compare how fast the implementations are
def time_function(f, *args):
  """
  Call a function f with args and return the time (in seconds) that it took to execute.
  """
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic

'''
two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print 'Two loop version took %f seconds' % two_loop_time

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print 'One loop version took %f seconds' % one_loop_time

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print 'No loop version took %f seconds' % no_loop_time
'''
classifier = KNearestNeighbor()
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)


################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in k_choices:
    k_to_accuracies[k] = np.zeros(num_folds)
    for i in range(num_folds):
        Xtr = np.array(X_train_folds[:i] + X_train_folds[i+1:])
        ytr = np.array(y_train_folds[:i] + y_train_folds[i+1:])
        Xte = np.array(X_train_folds[i])
        yte = np.array(y_train_folds[i])

        Xtr = np.reshape(Xtr, (X_train.shape[0] * 4 / 5, -1))
        ytr = np.reshape(ytr, (y_train.shape[0] * 4 / 5, -1))
        Xte = np.reshape(Xte, (X_train.shape[0] / 5, -1))
        yte = np.reshape(yte, (y_train.shape[0] / 5, -1))

        classifier.train(Xtr, ytr)
        yte_pred = classifier.predict(Xte, k)
        yte_pred = np.reshape(yte_pred, (yte_pred.shape[0], -1))
        num_correct = np.sum(yte_pred == yte)
        accuracy = float(num_correct) / len(yte)
        k_to_accuracies[k][i] = accuracy
        print 'k = %d, accuracy = %f' % (k, accuracy)

#图形化显示结果
# plot the raw observations
for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.savefig('/Users/jinyitao/Desktop/机器学习相关/《机器学习实战》/myProject/1_knn/knn_cifar10交叉验证图.jpg')
plt.show()
