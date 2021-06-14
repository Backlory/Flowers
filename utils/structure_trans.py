import numpy as np
import cv2

def gray2ggg(img):
    '''
    灰度转换为3d-numpy
    
    input:单通道灰度图
    output:三通道灰度图
    '''
    if len(img.shape) == 2:
        return np.concatenate((img, img, img), axis = 0)

def cv2numpy(img):
    '''
    将CV图片转换为numpy矩阵,RGB。
    输入（32，32，3），输出（3，32，32）
    输入（1500, 32，32，3），输出（1500, 3，32，32）
    '''
    if len(img.shape) == 3:
        assert(img.shape[2] == 1 or img.shape[2] == 3)
        return np.transpose(img,(2,0,1))
    else:
        assert(img.shape[3] == 1 or img.shape[3] == 3)
        return np.transpose(img,(0,3,1,2))
 
def numpy2cv(img):
    '''
    将numpy矩阵转换为CV图片。不转换通道
    输入（3，32，32），输出（32，32，3）
    输入（1500, 3，32，32），输出（1500, 32，32，3）
    '''
    if len(img.shape) == 3:
        assert(img.shape[0] == 1 or img.shape[0] == 3)
        return np.transpose(img,(1,2,0))
    else:
        assert(img.shape[1] == 1 or img.shape[1] == 3)
        return np.transpose(img,(0,2,3,1))

def _check_imgs(imgs):
    '''
    检查图片组。
    '''
    assert(len(imgs.shape) == 4)
    assert (imgs.shape[1] == 1 or imgs.shape[1] == 3) #号*channal*高*宽
    assert(np.max(imgs) > 1 or np.max(imgs) == 0)#255.
    return 1

def img2GaussianPyramid(img, level=3):
    '''
    输入图像，输出高斯金字塔
    imgs channal, h, w

    '''
    assert(len(img.shape) == 3)
    _, h, w = img.shape
    assert(2**(level-1) <= min(h, w))
    #
    img_pyramid = [img]
    temp = img.copy()
    for _ in range(level):
        dst = cv2.pyrDown(temp)
        img_pyramid.append(dst)
        temp = dst.copy()
    return img_pyramid
 
 
#拉普拉斯金字塔
def laplian_image(image):
    pyramid_images = img2GaussianPyramid(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if(i-1) < 0 :
            expand = cv2.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv2.subtract(image, expand)
            cv2.imshow("拉普拉斯"+str(i), lpls)
        else:
            expand = cv2.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv2.subtract(pyramid_images[i-1], expand)
            cv2.imshow("拉普拉斯"+str(i), lpls)

'''
src = cv2.imread("C://01.jpg")
cv2.imshow("原来", src)
laplian_image(src)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''