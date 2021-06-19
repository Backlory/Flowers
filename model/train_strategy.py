# 分类器训练(分类vec，结构规整->独热编码)
# 
import cv2
import numpy as np
import time

import torch
import torch.nn.functional as  F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

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
    #*=*=*机器学习
        temp = min(len(x_train[0]), 20) #PCA维度
        temp2 = trainmode[4:]
        trained_model = model_PCA_simply_classifier(temp2, pca_components = temp)
        #
        trained_model.train(x_train, y_train, weights)
        #trained_model = model_PCA_simply_classifier('SVR', pca_components = temp)
        #trained_model = model_PCA_simply_classifier('RFR', pca_components = temp)
        #trained_model = model_PCA_simply_classifier('LR', pca_components = temp)
    
    #*=*=*深度学习
    elif trainmode in ['CNN1', 'CNN2', 'alexnet', 'VGG16', 'shufflenet', 'pyramidnet', 'efficientnet',\
                                                                                'wideresnet', 'DenseNet', 'ResNeXt', 'SENet', 'ResNet']:
        print(colorstr('pytorch mode.',"magenta"))
        #
        trained_model = Network_model(trainmode)
        #
        trained_model.train(x_train, y_train, 100, 64)
        #
    return trained_model


#========================================分类器======================
class Network_model():
    def __init__(self, trainmode):

        #初始化网络
        self.network_model = None
        self.trainmode = trainmode
        self.criterion = nn.CrossEntropyLoss()
        if trainmode == 'SENet':
            from model.net.SEResNet import Se_ResNet18
            self.network_model = Se_ResNet18()
            self.inputshape = 224
        elif trainmode == 'ResNet':
            from model.net.ResNet import ResNet
            self.network_model = ResNet()
            self.inputshape = 224
        
        #转移到GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())])
        self.network_model.to(device)
        
        #配置优化器
        self.optimizer = optim.Adam(self.network_model.parameters())
    #
    def train(self, x_dataset, y_dataset, epochs=50, batch_size=32, validation_split = 0.3):
        assert(x_dataset.dtype==np.uint8)
        #模型转移到GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network_model.to(device)
        #数据准备
        x_dataset_resized = []
        for idx,img in enumerate(x_dataset):
            x_dataset_resized.append(cv2.resize(img, (self.inputshape,self.inputshape)))
        x_dataset =  np.array(x_dataset_resized, np.float) / 255.0
        x_dataset = np.transpose(x_dataset, (0,3,1,2)) #500,128,128,3 -> 500,3,128,128
        #ohe = OneHotEncoder()
        #temp = y_dataset[:, np.newaxis]
        #ohe.fit(temp)
        #y_dataset_ohe = ohe.transform(temp).toarray()
        assert(len(y_dataset.shape) == 1)
        #划分数据集
        valid_split = StratifiedShuffleSplit(   n_splits=1, 
                                                test_size = validation_split, 
                                                train_size = 1-validation_split) #分成5组，测试比例为0.25，训练比例是0.75
        for train_index, valid_index in valid_split.split(x_dataset, y_dataset):
            x_train, x_valid = x_dataset[train_index], x_dataset[valid_index]
            y_train, y_valid = y_dataset[train_index], y_dataset[valid_index]
        #
        x_train = torch.from_numpy(x_train).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)
        train_dataset=TensorDataset(x_train,y_train)
        train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
        #
        x_valid = torch.from_numpy(x_valid).float().to(device)
        y_valid = torch.from_numpy(y_valid).float().to(device)
        valid_dataset=TensorDataset(x_valid,y_valid)
        valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size, shuffle=True)
        #训练
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        best_acc = 0
        for epoch in range(epochs):
            starttime = time.time()         
            #数据准备
            train_losses_epoch = []
            valid_losses_epoch = []
            train_acc = 0
            valid_acc = 0
            #拟合阶段
            self.network_model.train()
            correct = 0
            for batch_idx,(data,target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.network_model(data)
                loss = self.criterion(output,target.long())
                loss.backward()
                self.optimizer.step()
                #
                train_losses_epoch.append(loss.item())
                if (batch_idx)%1 == 0:   #int(len(train_loader)/8)
                    predicted = torch.max(output.data,1)[1] #第一个1代表取每行的最大值，第二个1代表只取最大值的索引
                    correct += (predicted == target).sum()
                    train_acc = float(float(correct)/(batch_size)/(batch_idx+1))
                    #train_acc = train_acc.numpy()
                    temp = f'epoch {epoch}/{epochs}=> {batch_idx*len(data)}/{len(train_loader.dataset)}, loss : {loss.item():.5f} acc = {train_acc:.3f}'
                    print("\r"+temp,end="",flush=True)
            #验证阶段
            self.network_model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    output = self.network_model(data)
                    #val_loss
                    loss = self.criterion(output,target.long())
                    valid_losses_epoch.append(loss.item())
                    #val_acc
                    predicted = torch.max(output.data,1)[1] #第一个1代表取每行的最大值，第二个1代表只取最大值的索引
                    correct += (predicted == target).sum()
                    valid_acc = float(float(correct)/(batch_size)/(batch_idx+1))
                    #valid_acc = valid_acc.numpy()
            #评估本epoch
            train_loss = np.average(train_losses_epoch)
            valid_loss = np.average(valid_losses_epoch)
            print(f'\repoch {epoch}/{epochs}=>{(time.time()-starttime):.2f}s, loss={train_loss:.5f}, acc={train_acc:.3f}, val_loss={valid_loss:.5f}, val_acc={valid_acc:.3f}')
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)
            #earlystop
            if valid_acc >best_acc:
                best_acc = valid_acc
                earlystop_point = 0
                path = str(f'weights\\network\\model_{self.trainmode}_weight.pt')
                torch.save(self.network_model.state_dict(), path)
            else:
                earlystop_point +=1
                if earlystop_point > 4:
                    break
        del x_train, x_valid, y_train, y_valid, train_dataset, valid_dataset
        torch.cuda.empty_cache()

    #
    def predict(self, x_test):
        assert(x_test.dtype==np.uint8)
        #转移到GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network_model.to(device)
        #读取权重
        state_dict = torch.load(f'weights\\network\\model_{self.trainmode}_weight.pt')
        self.network_model.load_state_dict(state_dict)
        #数据移入GPU
        x_test_resized = []
        for idx,img in enumerate(x_test):
            x_test_resized.append(cv2.resize(img, (self.inputshape,self.inputshape)))
        x_test = np.array(x_test_resized, np.float) / 255.0
        x_test = np.transpose(x_test, (0,3,1,2)) #500,128,128,3 -> 500,3,128,128
        x_test = torch.from_numpy(x_test).float().to(device)
        #验证
        self.network_model.eval()
        with torch.no_grad():
            output = self.network_model(x_test)
            predicted = torch.max(output.data,1)[1] #第一个1代表取每行的最大值，第二个1代表只取最大值的索引
        predicted = predicted.cpu().numpy()
        return predicted

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