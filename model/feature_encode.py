#特征编码（特征vec->分类vec，结构规整)
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
        
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time

def data_augment(Dataset_imgs, Dataset_labels, mode_fet):
    Dataset_imgs_amt = []
    Dataset_labels_amt = []
    if mode_fet not in  ['CNN1', 'CNN2', 'alexnet', 'VGG16', 'shufflenet', 
                        'pyramidnet', 'efficientnet','wideresnet', 'DenseNet', 'ResNeXt', 
                        'SENet', 'ResNet']:
        Dataset_imgs_amt = Dataset_imgs
        Dataset_labels_amt = Dataset_labels
    else:
        #原图
        for idx, img in enumerate(Dataset_imgs):
            Dataset_imgs_amt.append(img)
            Dataset_labels_amt.append(Dataset_labels[idx])
        #翻转
        #旋转
        #拉伸
        #cutmix
        #mixup
        #Cutout
        #randaugment
        from model._randaugment import RandomAugment
        randaug = RandomAugment(N=2, M=10)
        for idx, img in enumerate(Dataset_imgs):
            #
            img_amt = randaug(img)
            Dataset_imgs_amt.append(img_amt)
            Dataset_labels_amt.append(Dataset_labels[idx])
            #
            img_amt = randaug(img)
            Dataset_imgs_amt.append(img_amt)
            Dataset_labels_amt.append(Dataset_labels[idx])

        
    Dataset_imgs_amt = np.array(Dataset_imgs_amt)
    Dataset_labels_amt = np.array(Dataset_labels_amt)
    return Dataset_imgs_amt, Dataset_labels_amt


@fun_run_time
def Featurencoder(datas_list, labels, onehot=False, display=True):
    '''
    输入：
    datas_list=N个元素的特征列表
    labels=N个元素的标签矩阵,numpy，1维
    mode= normal
    输出：X_dataset,  Y_dataset，代表训练集向量，N个*m维特征矩阵，N个*K类的二维独热编码
    '''
    if display:
        print(colorstr('='*50, 'red'))
        print(colorstr('Feature encoding...'))
    #X_dataset
    X_dataset = np.array(datas_list)
    if len(X_dataset.shape) == 1:
        X_dataset = X_dataset[:, np.newaxis]

    #Y_dataset
    if onehot:
        ohe = OneHotEncoder()
        labels = labels[:, np.newaxis]
        ohe.fit(labels)
        Y_dataset = ohe.transform(labels).toarray()
    else:
        Y_dataset = labels

    #处理结束，得到二维特征矩阵，每行一个图，每列一个特征
    #assert(len(X_dataset.shape) == 2)
    return X_dataset,  Y_dataset
