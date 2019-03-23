import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import random


'''
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
'''


#加载数据
#预处理，生成原始评分矩阵
def load_data():
    r_cols= ['user_id', 'movie_id', 'rating', 'timestamp']
    R_ori= pd.read_csv('./dataset/u1.base', sep='\t', names=r_cols)
    R_train= R_ori.pivot(index ='user_id', columns ='movie_id', values ='rating').fillna(0)
    R_ori= pd.read_csv('./dataset/u1.test',sep='\t',names=r_cols)
    R_test= R_ori.pivot(index ='user_id', columns ='movie_id', values ='rating').fillna(0)
    
    R_train=R_train.values
    R_test=R_test.values
    #print(type(R_train))
    print(R_train)
    return R_train,R_test

#def Calculate_Loss(R,P,Q,K,L,beta=0.02):

def SGD(R,R_test,P,Q,K,steps=400,beta=0.02,alpha=0.0002):
    L_train,L_test=[],[]
    Q = Q.T
    #for i in range(len(R)):
    for step in range(steps):
        print("step : ",step)
        for ii in range(0,128):
            i=random.randint(0,len(R)-1)
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        e = 0
        count=0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    count+=1
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        e/=count
        L_train.append(e)
        print("train : ",e)
        count=0
        e=0
        for i in range(len(R_test)):
            for j in range(len(R_test[0])):
                if R_test[i][j]>0:
                    count+=1
                    e=e+pow(R_test[i][j]-np.dot(P[i,:],Q[:,j]), 2)
        e/=count
        L_test.append(e)
        print("test",e)
    return L_train,L_test


def matrix_factorization(R,R_test):
    N = len(R)
    M = len(R[0])
    K = 20
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    L_train,L_test=SGD(R,R_test,P,Q,K)   
    Draw_pic(L_train,L_test)

def Draw_pic(L_train,L_test):
    n = len(L_train)
    x = range(n)
    plt.plot(x, L_train, color='r',linewidth=3,label='Train_Loss')
    plt.plot(x, L_test, color='b',linewidth=3,label='Test_Loss')
    plt.title('Convergence curve')
    plt.xlabel('generation')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    R_train,R_test=load_data()
    matrix_factorization(R_train,R_test)
