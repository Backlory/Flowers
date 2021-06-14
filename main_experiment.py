import numpy as np
import cv2
import random
from datetime import datetime
from data.data_loading import load_data

import utils.img_display as u_idsip
from utils.img_display import prepare_path, save_pic


if __name__ =='__main__':
    #变量准备
    
    experiment_type = 'train_expend'   #test, train_ori, train_expend
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
    
    u_idsip.show_pic(Dataset_imgs[disp_sample_list])
    #


