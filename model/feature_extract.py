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
    Dataset_fea_list = []
    if mode == 'Hu':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, fea_hu_moments)
    elif mode == 'Colorm':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, fea_color_moments)
    elif mode == 'SIFT':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, feas_SIFT)
    elif mode == 'greycomatrix':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, fea_greycomatrix)
    elif mode == 'HOG':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, fea_HOG)
    elif mode == 'LBP':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, fea_LBP)
    elif mode == 'DAISY':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, fea_daisy)
    #
    elif mode == 'Colorm_HOG_DAISY':
        fea1 = get_Vectors(PSR_Dataset_img, fea_color_moments)
        fea2 = get_Vectors(PSR_Dataset_img, fea_HOG)
        fea3 = get_Vectors(PSR_Dataset_img, fea_daisy)
        for item in zip(fea1,fea2, fea3):
            f1, f2, f3 = item
            temp = np.concatenate((f1, f2, f3), axis=0)
            Dataset_fea_list.append(temp)
        
        

    #处理结束

    return Dataset_fea_list
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

#SIFT特征
def feas_SIFT(img_cv):
    sift = cv2.xfeatures2d.SIFT_create()
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img_cv,None)
    des1 = np.array(des1, dtype=np.uint8)
    return des1

#灰度共生矩阵导出量
def fea_greycomatrix(img_cv):
    from skimage import feature as ft
    img_GRAY = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    img_GRAY = img_GRAY.astype(np.uint8, )
    #
    grmt = ft.greycomatrix(img_GRAY, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    feas = ft.greycoprops(grmt, 'contrast').flatten()
    feas = np.concatenate((feas, ft.greycoprops(grmt, 'dissimilarity').flatten()),axis=0)
    feas = np.concatenate((feas, ft.greycoprops(grmt, 'homogeneity').flatten()),axis=0)
    feas = np.concatenate((feas, ft.greycoprops(grmt, 'ASM').flatten()),axis=0)
    feas = np.concatenate((feas, ft.greycoprops(grmt, 'energy').flatten()),axis=0)
    feas = np.concatenate((feas, ft.greycoprops(grmt, 'correlation').flatten()),axis=0)
    return feas

#HOG特征
def fea_HOG(img_cv):
    from skimage import feature as ft
    img_GRAY = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    img_GRAY = img_GRAY.astype(np.uint8)
    #
    feas = ft.hog(img_GRAY)
    return feas

#LBP特征
def fea_LBP(img_cv):
    from skimage import feature as ft
    img_GRAY = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    img_GRAY = img_GRAY.astype(np.uint8)
    #
    lbp_map = ft.local_binary_pattern(img_GRAY, 8, 1)
    max_bins = int(np.max(lbp_map) + 1)
    feas, _ = np.histogram(lbp_map, normed=True, bins=max_bins, range=(0, max_bins))
    return feas

#DAISY特征
def fea_daisy(img_cv):
    from skimage import feature as ft
    img_GRAY = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    img_GRAY = img_GRAY.astype(np.uint8)
    #
    feas = ft.daisy(img_GRAY)
    feas = feas.flatten()
    return feas

#灰度梯度共生矩阵
from numba import jit

def fea_glgcm(img_gray, ngrad=16, ngray=16):
    '''
    Gray Level-Gradient Co-occurrence Matrix,取归一化后的灰度值、梯度值分别为16、16
    glgcm_features = glgcm(img_gray, 15, 15)
    '''
    # 利用sobel算子分别计算x-y方向上的梯度值
    gsx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gsy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    height, width = img_gray.shape
    grad = (gsx ** 2 + gsy ** 2) ** 0.5 # 计算梯度值
    grad = np.asarray(1.0 * grad * (ngrad-1) / grad.max(), dtype=np.int16)
    gray = np.asarray(1.0 * img_gray * (ngray-1) / img_gray.max(), dtype=np.int16) # 0-255变换为0-15
    gray_grad = np.zeros([ngray, ngrad]) # 灰度梯度共生矩阵
    for i in range(height):
        for j in range(width):
            gray_value = gray[i][j]
            grad_value = grad[i][j]
            gray_grad[gray_value][grad_value] += 1
    gray_grad = 1.0 * gray_grad / (height * width) # 归一化灰度梯度矩阵，减少计算量
    glgcm_features = get_glgcm_features(gray_grad)
    return glgcm_features
@jit
def get_glgcm_features(mat):
    '''根据灰度梯度共生矩阵计算纹理特征量，包括小梯度优势，大梯度优势，灰度分布不均匀性，梯度分布不均匀性，能量，灰度平均，梯度平均，
    灰度方差，梯度方差，相关，灰度熵，梯度熵，混合熵，惯性，逆差矩'''
    sum_mat = mat.sum()
    small_grads_dominance = big_grads_dominance = gray_asymmetry = grads_asymmetry = energy = gray_mean = grads_mean = 0
    gray_variance = grads_variance = corelation = gray_entropy = grads_entropy = entropy = inertia = differ_moment = 0
    for i in range(mat.shape[0]):
        gray_variance_temp = 0
        for j in range(mat.shape[1]):
            small_grads_dominance += mat[i][j] / ((j + 1) ** 2)
            big_grads_dominance += mat[i][j] * j ** 2
            energy += mat[i][j] ** 2
            if mat[i].sum() != 0:
                gray_entropy -= mat[i][j] * np.log(mat[i].sum())
            if mat[:, j].sum() != 0:
                grads_entropy -= mat[i][j] * np.log(mat[:, j].sum())
            if mat[i][j] != 0:
                entropy -= mat[i][j] * np.log(mat[i][j])
                inertia += (i - j) ** 2 * np.log(mat[i][j])
            differ_moment += mat[i][j] / (1 + (i - j) ** 2)
            gray_variance_temp += mat[i][j] ** 0.5

        gray_asymmetry += mat[i].sum() ** 2
        gray_mean += i * mat[i].sum() ** 2
        gray_variance += (i - gray_mean) ** 2 * gray_variance_temp
    for j in range(mat.shape[1]):
        grads_variance_temp = 0
        for i in range(mat.shape[0]):
            grads_variance_temp += mat[i][j] ** 0.5
        grads_asymmetry += mat[:, j].sum() ** 2
        grads_mean += j * mat[:, j].sum() ** 2
        grads_variance += (j - grads_mean) ** 2 * grads_variance_temp
    small_grads_dominance /= sum_mat
    big_grads_dominance /= sum_mat
    gray_asymmetry /= sum_mat
    grads_asymmetry /= sum_mat
    gray_variance = gray_variance ** 0.5
    grads_variance = grads_variance ** 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            corelation += (i - gray_mean) * (j - grads_mean) * mat[i][j]
    glgcm_features = [small_grads_dominance, big_grads_dominance, gray_asymmetry, grads_asymmetry, energy, gray_mean, grads_mean,
        gray_variance, grads_variance, corelation, gray_entropy, grads_entropy, entropy, inertia, differ_moment]
    return np.round(glgcm_features, 4)




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