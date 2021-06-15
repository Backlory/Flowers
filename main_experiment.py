from sys import path
import numpy as np
import cv2
import random
from datetime import datetime
from data.data_loading import load_data

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report                         
from sklearn.metrics import confusion_matrix

import model.feature_extract as m_fet
import model.feature_encode as m_fed
import model.train_strategy as m_ts

from weights.weightio import save_obj, load_obj
import utils.img_display as u_idsip
from utils.img_display import prepare_path, save_pic


if __name__ =='__main__':
    #变量准备
    mode_fet = 'SIFT'                 #Hu, Colorm, SIFT, greycomatrix, HOG, LBP, DAISY
    #mode_encode = 'bagofword'          #bagofword, normal
    mode_train = 'PCA_SVC'              #PCA_SVC
    #
    experiment_type = 'train_ori'   #test, train_ori, train_expend
    timenow = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    experiment_dir = 'experiment/'+ timenow +'/'
    prepare_path(experiment_dir)
    
    #数据加载
    Dataset_imgs, Dataset_labels = load_data('data\\data_'+experiment_type, datatype=experiment_type)
    print('size of imgs and labels in dataset:')
    print(Dataset_imgs.shape)
    print(Dataset_labels.shape)
    
    #数据采样
    readlist = list(range(len(Dataset_imgs)))
    Dataset_imgs = Dataset_imgs[readlist]
    Dataset_labels = Dataset_labels[readlist]

    # 显示样本列表
    np.random.seed(777)
    try:
        disp_sample_list = random.sample(range(len(readlist)), 64) #9,16,64
    except:
        disp_sample_list = random.sample(range(len(readlist)), 16) #9,16,64
    #u_idsip.show_pic(Dataset_imgs[disp_sample_list])
    

    # 特征列表提取
    try:
        Dataset_fea_list = load_obj(f'data\\{experiment_type}_{mode_fet}_{mode_train}_fea.joblib')
    except:
        Dataset_fea_list = m_fet.Featurextractor(   Dataset_imgs,
                                                    mode_fet,
                                                    True)
        save_obj(Dataset_fea_list, f'data\\{experiment_type}_{mode_fet}_{mode_train}_fea.joblib')
    

    # 特征编码
    if len(Dataset_fea_list[0].shape)==1:
        mode_encode = 'normal'
    else:
        mode_encode = 'bagofword'
    try:
        X_dataset,  Y_dataset = load_obj(f'data\\{experiment_type}_{mode_fet}_{mode_train}_encode.joblib')
    except:
        X_dataset,  Y_dataset= m_fed.Featurencoder(     Dataset_fea_list,
                                                        Dataset_labels,
                                                        mode_encode
                                                        )
        save_obj((X_dataset,  Y_dataset), f'data\\{experiment_type}_{mode_fet}_{mode_train}_encode.joblib')
    
    #K折交叉验证
    for K_fold_size in [5]:
        skf = StratifiedKFold(n_splits=K_fold_size, shuffle = True,random_state=999) #交叉验证，分层抽样
        
        y_test_gt_list, y_test_pred_list = [], []
        for idx, (train_index, test_index) in enumerate(skf.split(X_dataset, Y_dataset)):
            print(f'K = {idx+1} / {skf.n_splits}')
            
            #获取数据
            x_train, y_train = X_dataset[train_index], Y_dataset[train_index]
            x_test, y_test_gt = X_dataset[test_index], Y_dataset[test_index]
            
            #训练
            trained_model = m_ts.get_trained_model(x_train, y_train, mode_train, display=True)
            
            #测试
            y_test_pred = trained_model.predict(x_test)

            #统计测试结果
            y_test_gt_list.append(y_test_gt)
            y_test_pred_list.append(y_test_pred)
        
        #总数据统计
        print('total:')
        conf_mat = confusion_matrix(y_test_gt, y_test_pred)
        report_str = classification_report(y_test_gt, y_test_pred, zero_division=1, digits=4, output_dict=False)
        print(conf_mat)
        print(report_str)

        #评估报告打印 
        performence_report = f'K_fold_size={K_fold_size}'
        performence_report += '\n' + str(timenow)
        performence_report += '\n' + f'mode_fet={mode_fet}, mode_encode={mode_encode}, mode_train={mode_train}'
        performence_report += '\n' + '='*50
        #
        performence_report += '\n' + str(conf_mat)
        performence_report += '\n' + '-'*50
        performence_report += '\n' + str(report_str)

        path = experiment_dir+f'performence_{experiment_type}_{mode_fet}_{mode_train}_K={K_fold_size}.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(performence_report)
            f.close()

    save_obj(trained_model, 'weights\\trained_model.joblib')
