# 分类器训练(分类vec，结构规整->独热编码)
# 
import numpy as np
import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time



def get_trained_model(x_train, y_train, trainmode, display=True):
    '''
    调用不同的分类器
    '''
    if display:
        print(colorstr('='*50, 'red'))
        print(colorstr('Training...'))

    if trainmode =='PCA_SVC':
        temp = min(len(x_train[0]), 20)
        trained_model = model_PCA_SVC(pca_components = temp)
        trained_model.train(x_train, y_train)
    else:
        print('666')
    return trained_model


#========================================分类器======================

class model_PCA_SVC():
    '''
    如果比20大，自动PCA到20
    '''
    def __init__(self, pca_components):
        from sklearn.svm import SVC
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.svc = SVC()
        self.pca = PCA(n_components=pca_components)
    #
    def train(self, x_train, y_train):
        #
        #数据归一化
        self.scaler.fit(x_train)
        x_train = self.scaler.transform(x_train)
        #PCA降维
        self.pca.fit(x_train)
        x_train = self.pca.transform(x_train)
        
        #分类训练
        self.svc.fit(x_train, y_train)
    #
    def predict(self, x_test):
        #归一化
        x_test = self.scaler.transform(x_test)
        #PCA
        x_test = self.pca.transform(x_test)
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