from sys import path
import numpy as np
import cv2
import random
from datetime import datetime
from data.data_loading import load_data

from sklearn.metrics import classification_report                         
from sklearn.metrics import confusion_matrix

import model.feature_extract as m_fet
import model.feature_encode as m_fed
import model.train_strategy as m_ts

from weights.weightio import load_obj
from utils.img_display import prepare_path


if __name__ =='__main__':
    #变量准备ff
    #
    experiment_type = 'test'   #, train_ori, train_expend
    timenow = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    experiment_dir = 'experiment/'+ timenow +'/'
    #prepare_path(experiment_dir)
    
    #数据加载
    Dataset_imgs, Dataset_labels = load_data('data\\data_'+experiment_type, datatype=experiment_type)
    print('size of imgs and labels in testset:')
    print(Dataset_imgs.shape)
    print(Dataset_labels.shape)
    if_ROI, mode_fet, trained_model = load_obj('weights\\trained_model.joblib')
    print(f"if_ROI={if_ROI}, mode_fet={mode_fet}")
    Fea_extractor = load_obj(f'weights\\Fea_extractor_{if_ROI}{mode_fet}.joblib')
    
    #数据采样
    readlist = list(range(len(Dataset_imgs)))
    Dataset_imgs = Dataset_imgs[readlist]
    Dataset_labels = Dataset_labels[readlist]

    #ROI
    if if_ROI == 'R_':
        Dataset_imgs = m_fet.ROI(Dataset_imgs)

    # 特征列表提取
    if Fea_extractor == 0:
        Dataset_fea_list, _ = m_fet.Featurextractor(Dataset_imgs,
                                                    mode_fet,
                                                    True)
    else:
        Dataset_fea_list = Fea_extractor.extract(Dataset_imgs)

    # 特征编码
    X_dataset,  Y_dataset= m_fed.Featurencoder( Dataset_fea_list,
                                                Dataset_labels
                                                )
    
    #获取测试集数据
    x_test, y_test_gt = X_dataset, Y_dataset
    
    #测试
    y_test_pred = trained_model.predict(x_test)

    print(confusion_matrix(y_test_gt, y_test_pred))
    print(classification_report(y_test_gt, y_test_pred, zero_division=1, digits=4, output_dict=False))