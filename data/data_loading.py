import os
import numpy as np
import cv2

def load_data(dataset_path, datatype='train_ori'):
    '''
    输入路径，数据类型，读取数据。
    datatype = ['train_expend', 'test', 'train_ori']
    '''
    assert(datatype in ['train_expend', 'test', 'train_ori'])
    labels = []
    filedirlist = []
    if datatype=='train_expend' or datatype=='train_ori':
        for folders in os.listdir(dataset_path):
            folder_path = dataset_path+'\\'+folders
            for filename in os.listdir(folder_path):
                path = folder_path +'\\'+ filename
                label = int(folders)
                #
                filedirlist.append(path)
                labels.append(label)
    elif datatype=='test':
        for filename in os.listdir(dataset_path):
            path = dataset_path +'\\'+ filename
            label = int(filename[0:2])
            #
            filedirlist.append(path)
            labels.append(label)
    #
    imgs = []
    for filedir in filedirlist:
        img = cv2.imread(filedir, cv2.IMREAD_COLOR)
        imgs.append(img)
    #
    imgs = np.array(imgs, dtype=np.uint8)
    labels = np.array(labels)
    return imgs, labels
    