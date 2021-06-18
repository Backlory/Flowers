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
    if_ROI = 'R_'                         #R_, N_
    mode_fet = 'SENet'                     #Hu, Colorm,  greycomatrix, HOG, LBP, DAISY, glgcm 
                                            #SIFT, BRISK, Colorm_SIFT, Colorm_HOG_DAISY
                                            
                                            # 'CNN1', 'CNN2', 'alexnet', 'VGG16', 'shufflenet', 
                                            # 'pyramidnet', 'efficientnet','wideresnet', 'DenseNet', 'ResNeXt', 
                                            # 'SENet', 'ResNet'

    mode_train = 'SENet'              #'PCA_SVC', 'PCA_RFC', 'PCA_DT', 'PCA_NB', 'PCA_KNN', 'PCA_GBDT'    #其中RFC、KNN都挺好
    experiment_type = 'train_expend'   #test, train_ori, train_expend
    #
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
    try:
        disp_sample_list = random.sample(range(len(readlist)), 64) #9,16,64
    except:
        disp_sample_list = random.sample(range(len(readlist)), 16) #9,16,64
    #u_idsip.show_pic(Dataset_imgs[disp_sample_list])
    
    #======================================================================================
    #ROI
    if if_ROI == 'R_':
        try:
            Dataset_imgs,  Dataset_labels = load_obj(f'data\\{experiment_type}_{if_ROI}.joblib')
        except:
            Dataset_imgs = m_fet.ROI(Dataset_imgs)
            save_obj((Dataset_imgs,  Dataset_labels), f'data\\{experiment_type}_{if_ROI}.joblib')
    else:
        pass
    #======================================================================================
    # 特征列表提取与编码(测试时需Fea_extractor)
    try:
        X_dataset,  Y_dataset = load_obj(f'data\\{experiment_type}_{if_ROI}{mode_fet}_encode.joblib')
        Fea_extractor = load_obj( f'weights\\Fea_extractor_{if_ROI}{mode_fet}.joblib')   #因为训练阶段训练集测试集一起训练，所以不用load
    except:
        # 特征提取
        Dataset_feas, Fea_extractor = m_fet.Featurextractor(Dataset_imgs,
                                                            mode_fet,
                                                            True)
        # 特征编码
        X_dataset,  Y_dataset= m_fed.Featurencoder(     Dataset_feas,
                                                        Dataset_labels,
                                                        onehot=False
                                                        )
        # 数据增强
        X_dataset,  Y_dataset = m_fed.data_augment(X_dataset,  Y_dataset, mode_fet)
        save_obj((X_dataset,  Y_dataset), f'data\\{experiment_type}_{if_ROI}{mode_fet}_encode.joblib')
        save_obj(Fea_extractor, f'weights\\Fea_extractor_{if_ROI}{mode_fet}.joblib')
    #======================================================================================
    #K折交叉验证
    for K_fold_size in [5]:
        if K_fold_size != 0:
            seed = random.randint(0, 9999)
            skf = StratifiedKFold(n_splits=K_fold_size, shuffle = True,random_state=seed) #交叉验证，分层抽样
            
            y_test_gt_list, y_test_pred_list = [], []
            for idx, (train_index, test_index) in enumerate(skf.split(Y_dataset, Y_dataset)):
                print(f'K = {idx+1} / {skf.n_splits}')
                
                #获取数据
                x_train, y_train = X_dataset[train_index], Y_dataset[train_index]
                x_test, y_test_gt = X_dataset[test_index], Y_dataset[test_index]
                
                #训练
                if mode_fet == 'Colorm_SIFT':
                    weights=[200]*9+[9]*200
                else:
                    weights=[1]*len(x_train[0])
                trained_model = m_ts.get_trained_model(x_train, y_train, mode_train, weights, display=True)
                
                #测试
                y_test_pred = trained_model.predict(x_test)

                #统计测试结果
                y_test_gt_list.append(y_test_gt)
                y_test_pred_list.append(y_test_pred)
        else:
            
                y_test_gt_list, y_test_pred_list = [], []
                x_train, y_train = X_dataset, Y_dataset
                x_test, y_test_gt = X_dataset, Y_dataset
                #训练
                if mode_fet == 'Colorm_SIFT':
                    weights=[200]*9+[9]*200
                else:
                    weights=[1]*len(x_train[0])
                trained_model = m_ts.get_trained_model(x_train, y_train, mode_train, weights, display=True)
                
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
        performence_report += '\n' + f'mode_fet={mode_fet}, mode_train={mode_train}'
        performence_report += '\n' + '='*50
        #
        performence_report += '\n' + str(conf_mat)
        performence_report += '\n' + '-'*50
        performence_report += '\n' + str(report_str)

        path = experiment_dir+f'performence_{experiment_type}_{mode_train}_{mode_fet}_K={K_fold_size}.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(performence_report)
            f.close()
    trained_model = (if_ROI, mode_fet, trained_model)
    save_obj(trained_model, 'weights\\trained_model.joblib')
