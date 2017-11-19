# -- coding: utf-8 --
import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def knnClassify(inX, dataSet, labels, k):
    #(1) 计算已知数据和测试点数据的距离（采用欧式距离计算）
    #采用numpy矩阵计算的方式快速计算测试点数据和每个已知数据的欧式距离
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    #（2）按照距离大小排序
    sortedDistIndicies = distances.argsort() #获得从小到大排序的索引

    #（3）选取距离最小的前k个点，统计他们的label和次数
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1

    #（4）返回字典classCount中频率最高的label作为预测结果
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

'''
group,labels = createDataSet()
print group
print labels
print knnClassify([0,0],group,labels,3)
'''

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines) #读取文本中的行数

    #建立一个存放特征数据的numpy矩阵和label数据的列表
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #删除每行前面的空白符
        listFromLine = line.split('\t') #把字符串按照指定分隔符进行切片，并把结果返回为字符串列表
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index+=1
    return returnMat,classLabelVector

datingDataMat,datingLabels = \
    file2matrix('/Users/jinyitao/Desktop/机器学习相关/《机器学习实战》/machinelearninginaction/Ch02/datingTestSet2.txt')
#print datingDataMat
#print datingLabels

'''
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.show()
'''

#归一化特征到[0,1]
#根据公式：normValue = (value-min)/(max-min)
#用np.tile化为矩阵进行计算
def autoNorm(dataSet):
    #0表示选取每一列的最大最小值
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    ranges = maxValue-minValue
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minValue,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minValue

def datingClassTest():
    testRatio = 0.10 #测试数据的比例
    datingDataMat,datingLabels = \
        file2matrix('/Users/jinyitao/Desktop/机器学习相关/《机器学习实战》/machinelearninginaction/Ch02/datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0] #m为样本总数
    numTestVecs = int(m*testRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = knnClassify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %s,the real answer is: %s"%(classifierResult,datingLabels[i])
        if (classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print "the total error rate is: %f"% (errorCount/float(numTestVecs))

datingClassTest()
