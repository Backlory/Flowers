import numpy as np
import cv2
import data.data_loading
import time
from datetime import datetime
from torch.utils.data import DataLoader
import utils.img_display as u_idsip

if __name__ =='__main__':
    #数据加载
    PSR_Dataset = data.data_loading.PSR_Dataset('data')
    
    #数据预处理
    PSR_Dataset_img = []
    PSR_Dataset_label = []
    for idx, item in enumerate(PSR_Dataset):
        img, label = item
        img = u_idsip.cv2numpy(img)    #channal, height, width
        #
        PSR_Dataset_img.append(img)
        PSR_Dataset_label.append(label)
    
    #转化为numpy格式
    PSR_Dataset_img = PSR_Dataset_img[0:120] + PSR_Dataset_img[840:960] + PSR_Dataset_img[1680:1800]
    PSR_Dataset_img = np.array(PSR_Dataset_img)
    PSR_Dataset_label = PSR_Dataset_label[0:120] + PSR_Dataset_label[840:960] + PSR_Dataset_label[1680:1800]
    PSR_Dataset_label = np.array(PSR_Dataset_label)

    temp = u_idsip.img_square(PSR_Dataset_img)
    #u_idsip.show_pic(temp,'temp','freedom')
    u_idsip.save_pic(temp, 'temp', 'experiment/'+datetime.now().strftime('%Y%m%d-%H_%M_%S')+'/')

#TODO: 下边这段代码是视频形式显示删了，不需要了
    for i in range(120*3):
        #
        img = PSR_Dataset_img[i]
        label = PSR_Dataset_label[i]
        #
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        #img = cv2.Canny(img, 60,200)

        #contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        #img = cv2.drawContours(img, contours,-1,(255,255,255),thickness=-1)  #边缘框

        cv2.imshow('a', img)
        cv2.waitKey(10)