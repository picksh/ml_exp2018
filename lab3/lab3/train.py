import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import pickle
from scipy import misc
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from feature import NPDFeature
from ensemble import AdaBoostClassifier


label,train_label,validation_label=[],[],[]
feature,train_feature,validation_feature=[],[],[]
'''
class image:
    def __init__(self,feature,label):
        self.feature=feature
        self.label=label

def convert_gray(f):
    rgb=io.imread(f)
    gray=color.rgb2gray(rgb)  
    dst=transform.resize(gray,(24,24))
    return dst
'''
def rgb2gray(rgb):
    rgbimage=np.dot(rgb[...,:3],[0.299,0.587,0.114])
    return misc.imresize(rgbimage,(24,24))

#加载图片
#预处理数据
def readimg(): 
    currentpath1='./datasets/original/face/face_'
    currentpath2='./datasets/original/nonface/nonface_'
    for i in range(0,500):
        img_face = mpimg.imread(currentpath1+"{:0>3d}".format(i)+".jpg")
        img_face_=rgb2gray(img_face)
        f=NPDFeature(img_face_)
        feature_=f.extract()
        feature.append(feature_)
        label.append(1)

    for i in range(0,500):
        img_nonface=mpimg.imread(currentpath2+"{:0>3d}".format(i)+".jpg")
        img_nonface_=rgb2gray(img_nonface)
        f=NPDFeature(img_nonface_)
        feature_=f.extract()
        feature.append(feature_)
        label.append(-1)

if __name__ == "__main__":
    # write your code here  
    
    readimg()
    train_feature,validation_feature,train_label,validation_label=train_test_split(feature,label,test_size=0.3)
    #adaboost
    adaboostClassifier=AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1, random_state = 1),20)
    
    adaboostClassifier.save(train_feature,'train_feature')
    adaboostClassifier.save(train_label,'train_label')
    adaboostClassifier.save(validation_feature,'validation_feature')
    adaboostClassifier.save(validation_label,'validation_label')

    adaboostClassifier.fit(train_feature,train_label)
    adaboostClassifier.drawPic()
    

    '''
    #debug
    adaboostClassifier=AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1, random_state = 1),20)
    train_feature=adaboostClassifier.load("train_feature")
    train_label=adaboostClassifier.load("train_label")
    adaboostClassifier.fit(train_feature,train_label)
    adaboostClassifier.drawPic()
    '''
    



