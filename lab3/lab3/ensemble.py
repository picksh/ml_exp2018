import pickle
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier=weak_classifier
        self.n_weakers_limit=n_weakers_limit
        self.maxEpoch=20
        self.train_acc=[]
        self.vali_acc=[]
        self.vali_label=[]
        self.validation_prediction=[]
        


    def get_accuracy(self,pred, y):
        #return sum((pred == y) / float(len(y)))
        return np.sum(pred==y)/len(y)*1.0

    def is_good_enough(self):
        '''Optional'''
        if self.train_acc[-1] == 1:
            return 1

    def fit(self,X,y):
        vali_feature=self.load('validation_feature')
        self.vali_label=self.load('validation_label')
        alp_m=[]
        
        train_prediction = np.zeros(len(X),dtype=np.int32)
        self.validation_prediction = np.zeros(len(vali_feature),dtype=np.int32)
    
        #weight= np.mat(np.ones((len(X),1))/len(X))
        weight=np.ones(len(X))/len(X)
        train_hypothesis,validation_hypothesis=[],[]
        

        for i in range(0,self.maxEpoch):
            #weight=weight[0]
            print(type(weight[0]))
            if(isinstance(weight[0], np.ndarray)):
                weight=weight[0]
            self.weak_classifier.fit(X,y,sample_weight=weight)
            train_hypothesis.append(self.weak_classifier.predict(X))
            validation_hypothesis.append(self.weak_classifier.predict(vali_feature))

            errArray =  np.mat(np.ones((len(X),1)))
            errArray[train_hypothesis[i] == y] = 0
            err_m = weight * errArray
            if err_m>0.5 :
                break
            alp_m.append(0.5*np.log((1-err_m)/float(err_m)))
            expon=np.exp(-1*err_m*(y*train_hypothesis[i].T))
            weight=weight*expon.getA()
            weight=weight/weight.sum()

            
            train_prediction = train_prediction+ alp_m[i].getA() * train_hypothesis[i]
            self.validation_prediction= self.validation_prediction + alp_m[i].getA() * validation_hypothesis[i]
            train_pp=np.sign(train_prediction)
            self.train_acc.append( self.get_accuracy(train_pp,y) )
            self.vali_acc.append( self.get_accuracy(np.sign(self.validation_prediction),self.vali_label) )
            print("Train Accuracy:", self.train_acc[-1])
            print("Validation Accuracy:", self.vali_acc[-1])
            if self.is_good_enough()==1 :
                break

    def drawPic(self):
        plt.xlabel("Number of Decision Trees")
        plt.ylabel("Accuracy")
        plt.plot(self.train_acc, label ="train")
        plt.plot(self.vali_acc, label="validation")
        plt.legend(loc="lower right")
        plt.savefig('result.png', format='png')

        f = open("./report.txt", 'w+')  
        print(classification_report(self.vali_label,np.sign(self.validation_prediction[0])),file=f)
        f.close()


    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
