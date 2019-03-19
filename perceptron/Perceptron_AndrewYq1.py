# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:46:42 2019

@author: s66jmn
"""

import numpy as np
import time
import pandas as pd

#加载数据
def loadData(filePath):
    '''
    加载Mnist数据集
    :param filePath:加载的数据集路径
    :return: 返回输入数据集及输出数据集
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
    
    # 计算Gram Matrix矩阵
    
    print("初始化Gram_Matrix")
    # 初始化Gram_Matrix,初始值均为0
    gram_matrix=np.zeros((m,m),dtype='float16')
    
    print("计算Gram_Matrix")
    # 计算Gram_Matrix矩阵
    for i in range(m):
        gram_matrix[i]
    
    for i in range(m):
        for j in range(m):
            gram_matrix[i,j]=XArrMat[i,:]*XArrMat[j,:].T
            
    print("初始化，开始计算")    
    # 初始化 alpha 为 0
    alpha = np.zeros(m)
    
    # 初始化 b 为 0
    b = 0
    
    # 初始化步长 h, 即是梯度下降过程中的下降速率,控制梯度下降速率。
    h = 0.0001
    
    # 进行iter次迭代更新
    for j in range(iter):
        #对于每一个样本进行梯度下降
        for i in range(m):
            #获取当前样本所对应的标签
            yi = yArrMat[i]
            
            # 获取当前样本的向量
            xi = XArrMat[i]
            
            # 计算误分条件，计算yi(alpha*y*Gram+b),用来判断是否是误分类点
            if yi*(alpha*yi*gram_matrix[i,j]):
                
                alpha = alpha + h
                
                b = b + h*yi
        
        #打印训练进度
        print('Round %d:%d training' % (j, iter))
    
    #返回训练完的w、b  
    return alpha,b  
            

def perceptron_test(XArr, yArr, alpha, b):
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
    
    # 计算Gram Matrix矩阵
    
    # 初始化Gram_Matrix,初始值均为0
    gram_matrix=np.zeros((m,m),dtype='float16')
    
    # 计算Gram_Matrix矩阵
    
    
    for i in range(m):
        for j in range(m):
            gram_matrix[i,j]=XArrMat[i,:]*XArrMat[j,:].T
            
            
    #遍历所有测试样本
    for i in range (m):
        #获得单个样本向量
        xi = XArrMat[i]
        #获得该样本标记 
        yi = yArrMat[i]
        #获得运算结果   
        result = yi*(alpha*yi*gram_matrix[i,j])
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

    print("开始训练")
    #训练获得权重
    alpha, b = perceptron_train(trainData, trainLabel, iter = 30)
    
    print("开始测试")
    #进行测试，获得正确率
    accruRate = perceptron_test(testData, testLabel, alpha, b)

    #获取当前时间，作为结束时间
    end = time.time()
    #显示正确率
    print('accuracy rate is:', accruRate)
    #显示用时时长
    print('time span:', end - start)
