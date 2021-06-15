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
        #trained_model = model_PCA_simply_classifier('SVC', pca_components = temp)
        #trained_model = model_PCA_simply_classifier('DT', pca_components = temp)
        trained_model = model_PCA_simply_classifier('RFC', pca_components = temp)
        #trained_model = model_PCA_simply_classifier('NB', pca_components = temp)
        #trained_model = model_PCA_simply_classifier('KNN', pca_components = temp)
        #trained_model = model_PCA_simply_classifier('GBDT', pca_components = temp)
        #
        #trained_model = model_PCA_simply_classifier('SVR', pca_components = temp)
        #trained_model = model_PCA_simply_classifier('RFR', pca_components = temp)
        #trained_model = model_PCA_simply_classifier('LR', pca_components = temp)
        
        trained_model.train(x_train, y_train)
    else:
        print('666')
    return trained_model


#========================================分类器======================


class model_PCA_simply_classifier():
    '''
    直接分类器。先PCA，然后塞入分类器中
    如果比20大，自动PCA到20
    '''
    def __init__(self,classifier, pca_components):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components = pca_components)
        self.classifier = None
        #
        if classifier =='SVC':
            #支持向量机分类
            from sklearn.svm import SVC
            self.classifier = SVC(C=1, kernel='rbf', gamma='scale', probability=True, verbose=0)
        elif classifier == 'SVR':
            #支持向量机回归
            from sklearn.svm import SVR
            self.classifier = SVR(kernel='rbf', verbose=0)
        elif classifier == 'DT':
            #决策树
            from sklearn.tree import DecisionTreeClassifier
            self.classifier = DecisionTreeClassifier(    criterion="gini",
                                                    splitter="best",
                                                    max_depth=None)
        elif classifier == 'RFC':
            #随机森林
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(    n_estimators=100,
                                                    criterion="gini", 
                                                    max_depth=None)
        elif classifier == 'RFR':
            from sklearn.ensemble import RandomForestRegressor 
            self.classifier = RandomForestRegressor(    n_estimators=100,
                                                    criterion="gini", 
                                                    max_depth=None)
        elif classifier == 'NB':
            #朴素贝叶斯多项式
            from sklearn.naive_bayes import GaussianNB
            self.classifier = GaussianNB()
        elif classifier == 'KNN':
            #K最近邻分类器
            from sklearn.neighbors import KNeighborsClassifier
            self.classifier = KNeighborsClassifier()
        elif classifier == 'LR':
            #逻辑斯蒂回归
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(penalty='l2')
        elif classifier == 'GBDT':
            from sklearn.ensemble import GradientBoostingClassifier
            self.classifier = GradientBoostingClassifier(n_estimators=200)
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
        self.classifier.fit(x_train, y_train)
    #
    def predict(self, x_test):
        #归一化
        x_test = self.scaler.transform(x_test)
        #PCA
        x_test = self.pca.transform(x_test)
        #预测
        y_test_pred = self.classifier.predict(x_test)
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