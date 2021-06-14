import numpy as np
import cv2
import data.data_loading
import time
from datetime import datetime

from data.data_loading import load_data
import utils.img_display as u_idsip
from utils.img_display import prepare_path, save_pic, img_square


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
    
    #


