# 分类器训练(分类vec，结构规整->独热编码)
# 
import numpy as np
import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time

from sklearn.preprocessing import StandardScaler


def get_trained_model(x_train, y_train, trainmode, display=True):
    '''
    调用不同的分类器
    '''
    if display:
        print(colorstr('='*50, 'red'))
        print(colorstr('Training...'))

    if trainmode =='SVC':
        trained_model = model_SVC()
        trained_model.train(x_train, y_train)
    else:
        print('666')
    return trained_model


#========================================分类器======================

class model_SVC():
    def __init__(self):
        from sklearn.svm import SVC
        self.scaler = StandardScaler()
        self.svc = SVC()
    #
    def train(self, x_train, y_train):
        #
        #数据归一化
        self.scaler.fit(x_train)
        x_train = self.scaler.transform(x_train)
        
        #分类训练
        self.svc.fit(x_train, y_train)
    #
    def predict(self, x_test):
        #归一化
        x_test = self.scaler.transform(x_test)
        #预测
        y_test_pred = self.svc.predict(x_test)
        return y_test_pred

'''
class classify_model():
    def __init__(self):
        pass
    def train(self, x_train, y_train):
        scaler = StandardScaler().fit(x_train)                     #标准化
        #scaler = MinMaxScaler().fit(x_train)                       #归一化
        x_train = scaler.transform(x_train)
        pass
    def predict(self, x_test):
        pass
'''