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
from numba import jit
from matplotlib import pyplot as plt

import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time

def ROI(PSR_Dataset_img):
    PSR_Dataset_img = get_Vectors(PSR_Dataset_img, get_flower_area)
    PSR_Dataset_img = np.array(PSR_Dataset_img)
    return PSR_Dataset_img


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
    Fea_extractor = None

    #特征获取
    Dataset_fea_list = []
    if mode == 'Hu':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, fea_hu_moments)
    elif mode == 'Colorm':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, fea_color_moments)
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
    elif mode=='glgcm':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, fea_glgcm)
    elif mode=='BRISK':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, fea_BRISK)
    elif mode == 'SIFT':
        Dataset_fea_list = get_Vectors(PSR_Dataset_img, feas_SIFT)


    return Dataset_fea_list, Fea_extractor
        
def bagofword(feas_list):
    '''
    归一化处理器，词袋
    '''
    #词袋模型
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    #生成词袋
    word_num = 500
    word_bag = feas_list[0]   #视觉词袋，m*36
    for data in feas_list[1:]:
        word_bag = np.concatenate((word_bag, data), axis=0) 
    scaler = StandardScaler()
    scaler.fit(word_bag)
    word_bag = scaler.transform(word_bag)
    #训练词典
    #
    word_dict = KMeans(n_clusters=word_num,verbose=1) #视觉词典，容量500
    word_dict.fit(word_bag)
    #编码转化，视觉词统计
    _dtype = word_bag.dtype
    feas = np.zeros((len(feas_list), word_num), dtype=_dtype)
    for idx, data in enumerate(feas_list):
        words = word_dict.predict(data)
        for word in words:
            feas[idx, word] += 1

    #处理结束
    feas = np.array(feas)
    return feas, scaler, word_dict
#====================================================================

#BRISK特征
def fea_BRISK(img_cv):
    from skimage import feature as ft
    img_GRAY = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    img_GRAY = (img_GRAY - np.min(img_GRAY))/(np.max(img_GRAY)-np.min(img_GRAY))*255
    img_GRAY = img_GRAY.astype(np.uint8)
    #
    detector = cv2.BRISK_create()       #BRISK_create, AKAZE_create
    kp = detector.detect(img_GRAY,None)  
    kp, feas = detector.compute(img_GRAY, kp)

    
    return fea


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
@jit
def fea_glgcm(img_cv, ngrad=16, ngray=16):
    '''
    Gray Level-Gradient Co-occurrence Matrix,取归一化后的灰度值、梯度值分别为16、16
    glgcm_features = glgcm(img_gray, 15, 15)
    '''
    img_GRAY = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gsx = cv2.Sobel(img_GRAY, cv2.CV_64F, 1, 0, ksize=3)
    gsy = cv2.Sobel(img_GRAY, cv2.CV_64F, 0, 1, ksize=3)
    height, width = img_GRAY.shape
    grad = (gsx ** 2 + gsy ** 2) ** 0.5 # 计算梯度值
    grad = np.asarray(1.0 * grad * (ngrad-1) / grad.max(), dtype=np.int16)
    gray = np.asarray(1.0 * img_GRAY * (ngray-1) / img_GRAY.max(), dtype=np.int16) # 0-255变换为0-15
    gray_grad = np.zeros([ngray, ngrad]) # 灰度梯度共生矩阵
    for i in range(height):
        for j in range(width):
            gray_value = gray[i][j]
            grad_value = grad[i][j]
            gray_grad[gray_value][grad_value] += 1
    gray_grad = 1.0 * gray_grad / (height * width) # 归一化灰度梯度矩阵，减少计算量
    feas = get_glgcm_features(gray_grad)
    return feas
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
    glgcm_features = [  small_grads_dominance,
                        big_grads_dominance, 
                        gray_asymmetry, 
                        grads_asymmetry, 
                        energy, gray_mean, 
                        grads_mean,
                        gray_variance, 
                        grads_variance, 
                        corelation, 
                        gray_entropy, 
                        grads_entropy, 
                        entropy, 
                        inertia, 
                        differ_moment]
    return np.round(glgcm_features, 8)

#区域
def get_flower_area(img_cv):
    #1、图像中的像素分为花和背景两类#
    #花的颜色比背景更鲜艳。给鲜艳程度分等级？#
    #像素的空间位置比像素值本身更重要。
    #中心的像素大多都属于花。
    #相邻的像素大多都是好的。
    # 目标：判断出超像素。花=1，非花=0
    # 用超像素的属性均值作为特征，对
    def gaussian2D_mask(width, height, R_factor):
        R = np.sqrt(width**2 + height**2)/4 * R_factor #高斯蒙版半径
        distance_map = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                dis = np.sqrt((i-height/2)**2+(j-width/2)**2)
                distance_map[i, j] = np.exp(-0.5*dis/R)*255
        distance_map = distance_map.astype(np.uint8)
        return distance_map
    def graylevel_down(img, obj=16):
        '''
        输入图像，输出灰度级降低过的图像。
        目前是fac=16.
        256级灰度降低为256/fac级灰度。
        '''
        fac = 256/obj
        temp = img*1.0/fac
        temp = temp.astype(np.uint8)
        temp = temp * fac
        temp = temp.astype(np.uint8)
        return temp

    #
    #转换
    img_cv = graylevel_down(img_cv, 64)
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    i_h, i_s, i_v = cv2.split(img_hsv)
    i_d = gaussian2D_mask(128, 128, 0.3)
    
    #滤波
    temp = i_h.copy()
    temp[i_h>35]=1
    temp[i_h>77]=0
    i_h = np.where(temp == 1, 56, i_h)
    i_v = np.where(temp == 1, i_v*0.6, i_v)
    
    #统计特征
    from skimage.segmentation import slic,mark_boundaries
    img_segments_out = slic(img_cv, n_segments=100, compactness=50, start_label=1).astype(np.uint8)
    superpixels_fea = []
    for label in range(np.max(img_segments_out)):
        sp_h = np.where(img_segments_out==label, i_h, 0).astype(np.uint8)
        sp_s = np.where(img_segments_out==label, i_s, 0).astype(np.uint8)
        sp_v = np.where(img_segments_out==label, i_v, 0).astype(np.uint8)
        sp_d = np.where(img_segments_out==label, i_d*2, 0).astype(np.uint8)   #权重不同
        fea_vector = np.zeros((7))
        for idx, item in enumerate([sp_h, sp_s, sp_v, sp_d]):   #sp_r, sp_g, sp_b, 
            temp = np.sum(item>0)
            if temp ==0: temp = 1
            temp = np.sum(item) /temp
            fea_vector[idx] = temp
        superpixels_fea.append(fea_vector)
    #聚类
    from sklearn.cluster import KMeans
    kmeans=KMeans(n_clusters=2)
    kmeans.fit(superpixels_fea)
    y_pred = kmeans.predict(superpixels_fea)    #超像素标签

    #根据聚类结果转换标签
    img_segment_mask = np.ones_like(img_segments_out, dtype=np.uint8)
    for label in range(np.max(img_segments_out)):
        img_segment_mask[img_segments_out == label] = y_pred[label]
    
    # 图像最外围检测
    fac1 = np.average(i_d[img_segment_mask==0])
    fac2 = np.average(i_d[img_segment_mask==1])
    if fac1 > fac2: 
        img_segment_mask = 1 - img_segment_mask
    
    #中心区域保护
    h, w = img_segment_mask.shape
    nlabels, labelsmap, stats, centroids = cv2.connectedComponentsWithStats(1-img_segment_mask)
    for i in range(1, nlabels):
        regions_size = stats[i,4]
        centerx = centroids[i,0]
        centery = centroids[i,1]
        dis = ((centerx-64)**2+(centery-64)**2)**0.5
        if  regions_size < h * w * 0.1 and dis < 9999: #300
            x0 = stats[i,0]
            y0 = stats[i,1]
            x1 = stats[i,0]+stats[i,2]
            y1 = stats[i,1]+stats[i,3]
            area = np.zeros_like(labelsmap)
            area[y0:y1, x0:x1] = 1
            img_segment_mask = np.where(area>0, 1, img_segment_mask)
    #
    i_b, i_g, i_r = cv2.split(img_cv)
    sp_b = np.where(img_segment_mask>0, i_b, 0).astype(np.uint8)
    sp_g = np.where(img_segment_mask>0, i_g, 0).astype(np.uint8)
    sp_r = np.where(img_segment_mask>0, i_r, 0).astype(np.uint8)
    out2 = cv2.merge((sp_b, sp_g, sp_r))
    return out2



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