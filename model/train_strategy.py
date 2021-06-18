# 分类器训练(分类vec，结构规整->独热编码)
# 
import numpy as np

import torch
import torch.nn.functional as  F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets,transforms



import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time



def get_trained_model(x_train, y_train, trainmode, weights, display=True):
    '''
    调用不同的分类器
    '''
    if display:
        print(colorstr('='*50, 'red'))
        print(colorstr('Training...'))

    if trainmode in ['PCA_SVC', 'PCA_RFC', 'PCA_DT', 'PCA_NB', 'PCA_KNN', 'PCA_GBDT']:
    #机器学习
        temp = min(len(x_train[0]), 20) #PCA维度
        temp2 = trainmode[4:]
        trained_model = model_PCA_simply_classifier(temp2, pca_components = temp)
        #
        trained_model.train(x_train, y_train, weights)
    
    #深度学习
    elif trainmode in ['CNN1', 'CNN2', 'alexnet', 'VGG16', 'shufflenet', 'pyramidnet', 'efficientnet',\
                                                                                'wideresnet', 'DenseNet', 'ResNeXt', 'SENet', 'ResNet']:
        print(colorstr('pytorch mode.',"magenta"))
        #
        network_model = None
        #初始化网络
        if trainmode == 'SENet':
            from model.net.SEResNet import Se_ResNet18
            network_model = Se_ResNet18()
        elif trainmode == 'ResNet':
            from model.net.ResNet import ResNet
            pass
        #移入GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        network_model.to(device)
        #
        #
        #trained_model = model_PCA_simply_classifier('SVR', pca_components = temp)
        #trained_model = model_PCA_simply_classifier('RFR', pca_components = temp)
        #trained_model = model_PCA_simply_classifier('LR', pca_components = temp)
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
        #self.scaler_afterPCA = StandardScaler()
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components = pca_components)
        self.classifier = None
        self.weights = []
        #
        if classifier =='SVC': #支持向量机分类
            from sklearn.svm import SVC
            self.classifier = SVC(C=1, kernel='rbf', gamma='scale', probability=True, verbose=0)
        elif classifier == 'SVR': #支持向量机回归
            from sklearn.svm import SVR
            self.classifier = SVR(kernel='rbf', verbose=0)
        elif classifier == 'DT':  #决策树
            from sklearn.tree import DecisionTreeClassifier
            self.classifier = DecisionTreeClassifier(    criterion="gini",
                                                        splitter="best",
                                                        max_depth=None)
        elif classifier == 'RFC':  #随机森林
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(    n_estimators=100,
                                                    criterion="gini", 
                                                    max_depth=None)
        elif classifier == 'RFR': 
            from sklearn.ensemble import RandomForestRegressor 
            self.classifier = RandomForestRegressor(    n_estimators=100,
                                                    criterion="gini",
                                                    max_depth=None)
        elif classifier == 'NB': #朴素贝叶斯多项式
            from sklearn.naive_bayes import GaussianNB
            self.classifier = GaussianNB()
        elif classifier == 'KNN': #K最近邻分类器
            from sklearn.neighbors import KNeighborsClassifier
            self.classifier = KNeighborsClassifier()
        elif classifier == 'LR': #逻辑斯蒂回归
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(penalty='l2')
        elif classifier == 'GBDT':
            from sklearn.ensemble import GradientBoostingClassifier
            self.classifier = GradientBoostingClassifier(n_estimators=100,
                                                            subsample=0.2,
                                                            max_depth=7,
                                                            min_samples_split=30,
                                                            max_features=5,
                                                            max_leaf_nodes = 10
                                                            )
    #
    def train(self, x_train, y_train, weights):
        #
        #数据归一化
        self.scaler.fit(x_train)
        x_train = self.scaler.transform(x_train)

        #权重处理
        self.weights = [x/sum(weights)*len(weights) for x in weights]
        assert(len(self.weights) == len(x_train[0]))
        for idx, weight in enumerate(self.weights):
            x_train[:,idx] = x_train[:,idx] * weight
        #print(f"self.weights={[round(x,4) for x in self.weights]}")

        #PCA降维
        self.pca.fit(x_train)
        x_train = self.pca.transform(x_train)
        print(f"PCA at classifier:{sum(self.pca.explained_variance_ratio_)}")

        #self.scaler_afterPCA.fit(x_train)
        #x_train = self.scaler_afterPCA.transform(x_train)
        #分类训练
        self.classifier.fit(x_train, y_train)
        #
        '''
        from sklearn.model_selection import GridSearchCV
        #param_test = {'n_estimators': list(range(1, 501, 50))} = 100
        #param_test = {'max_depth': list(range(3, 14, 2)), 'min_samples_split': list(range(2, 30, 5))}
        #param_test = {'max_features': list(range(1, 6, 1))}
        #param_test = {'max_features': list(range(5, 200, 30))}
        #param_test = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
        gsearch2 = GridSearchCV(self.classifier,
                                    param_grid=param_test,
                                    scoring='accuracy', 
                                    cv=5,
                                    verbose = 2
                                    )
        grid_result  = gsearch2.fit(x_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_,gsearch2.best_params_))
        
        means = grid_result.cv_results_['mean_test_score']
        params = grid_result.cv_results_['params']
        for mean,param in zip(means,params):
            print("%f  with:   %r" % (mean,param))
        return 1
        '''
    #
    def predict(self, x_test):
        #归一化
        x_test = self.scaler.transform(x_test)

        #权重处理
        assert(len(self.weights) == len(x_test[0]))
        for idx, weight in enumerate(self.weights):
            x_test[:,idx] = x_test[:,idx] * weight

        #PCA
        x_test = self.pca.transform(x_test)
        #
        #x_test = self.scaler_afterPCA.transform(x_test)
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