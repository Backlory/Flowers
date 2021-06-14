# 特征提取。（pic1->vec_list)


# ROI区域提取。（pic3->pic1黑白)
# CV图片通道在第四位，平时numpy都放在第二位的
# 预处理部分。（pic3->pic3)
import math
from os import replace
from sys import api_version
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time


@fun_run_time
def Featurextractor(PSR_Dataset_img, mode = '', display=True):
    '''
    输入：4d图片集，(num, h, w,  c),BGR\n

    输出：特征列表，列表内每个元素都是矩阵。\n
    '''
    if display:
        print(colorstr('='*50, 'red'))
        print(colorstr('Feature extracting...'))
    #
    num, c, h, w = PSR_Dataset_img.shape

    #特征获取
    if mode == 'Hu':
        #Hu不变矩
        PSR_Dataset_Vectors = get_Vectors(PSR_Dataset_img, fea_hu_moments)
    elif mode == 'Colorm':
        PSR_Dataset_Vectors = get_Vectors(PSR_Dataset_img, fea_color_moments)
        pass

    #处理结束

    return PSR_Dataset_Vectors
#====================================================================
def fea_color_moments(img_cv):
    
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    color_feature = []
    # The first central moment - average 
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

    return color_feature


#Hu不变矩
def fea_hu_moments(img_cv):
    '''
    计算Hu不变矩的负对数。输入cv图片，RGB
    '''
    #
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    #
    moments = cv2.moments(img_cv)   #支持自动转换，非零像素默认为1，计算图像的三阶以内的矩
    humoments = cv2.HuMoments(moments) #计算Hu不变矩
    humoments = humoments[:,0]
    humoments = -np.log10(np.abs(humoments))
    return humoments




#====================================================================
def get_Vectors(imgs, func, **kwargs):
    '''
    输入图片组和函数、参数字典，输出函数结果
    '''
    t=tic()
    u_st._check_cvimgs(imgs)
    img_num = len(imgs)
    #
    result = []
    for idx, img in enumerate(imgs):
        temp = func(img, **kwargs)
        temp = np.array(temp)
        result.append(temp)
        if img_num>10:
            if idx % int(img_num/10) == int(img_num/10)-1:
                print(f'\t----{idx+1}/{img_num} has been extracted...')
    #
    #result = np.array(result) #对于同长度向量而言可以转化为array，对于sift等视觉词特征则不行
    toc(t, func.__name__, img_num)
    return result