# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:38:22 2019

@author: AndrewYq

@Email:hfyqstar@163.com
"""

import numpy as np
import time
import pandas as pd

#加载数据
def loadData(filePath):
    '''
    加载Mnist数据集
    :param filePath:加载的数据集路径
    :return: 输入数据集及输出数据集
    '''
    print('Start to load data')
    # 读入CSV数据
    # pandas.read_csv默认会将第一行数据作为列名，其中header用于指定某一行作为列名。
    # header=None将列的序列作为列名。
    csv_data=pd.read_csv(filePath, header = None)
    
    # CSV数据集中第一列为输出数据集，第一列以外均为输入数据集
    # dataFrame中若定义了index，那么loc是根据index来索引，iloc根据行号来索引，行号从0开始。
    # loc的索引值为index,为字符串；iloc的索引值为行号，为整数。
    # 切片表达式：dataFrame[row,column]
    XArr=csv_data.iloc[:,1::]
    yArr=csv_data.iloc[:,0]
    
    # 将所有输入数据除255归一化
    XArr=XArr/256
    # Mnsit有0-9是个标记，由于是二分类任务，所以将>=5的作为1，<5为-1
    yArr=yArr.replace([0,1,2,3,4,5,6,7,8,9],[-1,-1,-1,-1,-1,1,1,1,1,1])
    
    # 返回输入数据集和输出数据集
    return XArr,yArr

# 感知机预测
def perceptron_train(XArr,yArr,iter=100):
    
    # 计算输入矩阵的维度，其中 m 为横列数， n 为纵列数。
    m,n=np.shape(XArr)
    
    #将数据转换成矩阵形式（在机器学习中通常都是向量的运算，转换称矩阵形式方便运算）
    #转换后的数据中每一个样本的向量都是横向的
    # np.shape(dataMat)的返回值为m，n -> np.shape(dataMat)[1])的值即为n，与样本长度保持一致
    XArrMat=np.mat(XArr)
    #对于只有1xN的label可以不转换成矩阵，直接dataMat[i]即可，这里转换是为了格式上的统一
    yArrMat=np.mat(yArr).T
    
    # 初始化 w 为 0, w 为 weight vector, 长度等于每一个样本的特征值数，为 n。
    w=np.zeros(n)
    # 初始化b=0，b 为 bias
    b=0
    # 初始化步长，也即是梯度下降过程中的下降速率,控制梯度下降速率。
    h = 0.0001
    
    # 进行iter次迭代计算
    for j in range(iter):
        #对于每一个样本进行梯度下降
        #李航书中在2.3.1开头部分使用的梯度下降，是全部样本都算一遍以后，统一
        #进行一次梯度下降
        #在2.3.1的后半部分可以看到（例如公式2.6 2.7），求和符号没有了，此时用
        #的是随机梯度下降，即计算一个样本就针对该样本进行一次梯度下降。
        #两者的差异各有千秋，但较为常用的是随机梯度下降。
        for i in range(m):
            #获取当前样本的向量
            xi = XArrMat[i]
            #获取当前样本所对应的标签
            yi = yArrMat[i]
            #判断是否是误分类样本
            #误分类样本特诊为： -yi(w*xi+b)>=0，详细可参考书中2.2.2小节
            #在书的公式中写的是>0，实际上如果=0，说明改点在超平面上，也是不正确的
            if -1*yi*(w*xi.T+b)>=0:
                #对于误分类样本，进行梯度下降，更新w和b
                w = w + h * yi * xi
                b = b + h *yi
                
        #打印训练进度
        print('Round %d:%d training' % (j, iter))
    
    #返回训练完的w、b  
    return w,b  

def perceptron_test(XArr, yArr, w, b):
    '''
    测试准确率
    :param XArr:测试集
    :param yArr: 测试集标签
    :param w: 训练获得的权重w
    :param b: 训练获得的偏置b
    :return: 正确率
    '''
    
    #获取测试数据集矩阵的大小
    m,n=np.shape(XArr)
    #将数据集转换为矩阵形式方便运算
    XArrMat=np.mat(XArr)
    yArrMat=np.mat(yArr).T
    #错误样本数计数
    errorCount = 0
    #遍历所有测试样本
    for i in range (m):
        #获得单个样本向量
        xi = XArrMat[i]
        #获得该样本标记 
        yi = yArrMat[i]
        #获得运算结果   
        result = -1*yi*(w*xi.T+b)
        #如果-yi(w*xi+b)>=0，说明该样本被误分类，错误样本数加一
        if result>=0:
            errorCount +=1
            
    #正确率 = 1 - （样本分类错误数 / 样本总数）         
    accruRate = 1 - (errorCount / m) 
    #返回正确率
    return accruRate
                
if __name__ == '__main__':
    #获取当前时间
    #在文末同样获取当前时间，两时间差即为程序运行时间
    start = time.time()

    #获取训练集及标签
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    #获取测试集及标签
    testData, testLabel = loadData('../Mnist/mnist_test.csv')

    #训练获得权重
    w, b = perceptron_train(trainData, trainLabel, iter = 30)
    #进行测试，获得正确率
    accruRate = perceptron_test(testData, testLabel, w, b)

    #获取当前时间，作为结束时间
    end = time.time()
    #显示正确率
    print('accuracy rate is:', accruRate)
    #显示用时时长
    print('time span:', end - start)
