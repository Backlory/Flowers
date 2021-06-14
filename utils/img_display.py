import numpy as np
import cv2
import os,sys

from .structure_trans import gray2ggg
from .tools import fun_run_time
from .structure_trans import cv2numpy

def prepare_path(name_dir):
    '''
    生成路径。
    '''
    name_dir = name_dir.replace('/','\\')
    if os.path.exists(name_dir):
        pass
    elif sys.platform=='win32':
        os.system('mkdir ' + name_dir)
    else:
        os.system('mkdir -p ' +name_dir)
    return 1

#@fun_run_time
def save_pic(data,filename,filedir = ''):
    '''
    输入二维图片，三维、四维图片都可。
    save_pic(temp,'1.png','test/test1')，
    '''
    path = filedir+'\\'+filename
    
    if np.max(data)<=1 and np.max(data)>0: 
        data=data*255.
    data=np.uint8(data)

    try: 
        prepare_path(filedir)
        #
        if len(data.shape)==2:
            cv2.imwrite(path, data)

        elif len(data.shape)==3:
            if (data.shape[0] == 1) or (data.shape[0] == 3):
                data = np.transpose(data,(1,2,0))
            cv2.imwrite(path, data)

        elif len(data.shape)==4:
            if (data.shape[1] == 1) or (data.shape[1] == 3):\
                data = np.transpose(data,(0,2,3,1))
            img = img_square(data)
            cv2.imwrite(path, img)
        #
        print(f'\t----image saved in {path}')
    except:
        print('\t----file dir error')



def show_pic(data,windowname = 'default',showtype='freeze'):
    '''
    展示CV图片。
    show_pic(pic,"asd","freeze") 冻结型显示
    show_pic(pic,"asd","freedom")自由型显示
    '''
    if np.max(data)<=1 and np.max(data)>0: 
        data=data*255.
    data=np.uint8(data)

    if len(data.shape)==2:
        data = data
    elif len(data.shape)==3:
        if (data.shape[0] == 1) or (data.shape[0] == 3):
            data = np.transpose(data,(1,2,0))

    elif len(data.shape)==4:
        if (data.shape[1] == 1) or (data.shape[1] == 3):\
            data = np.transpose(data,(0,2,3,1))
        data = img_square(data)

    cv2.namedWindow(windowname,0)
    cv2.resizeWindow(windowname, 640, 480)
    cv2.imshow(windowname, data)

    if showtype=='freeze':
        cv2.waitKey(0)
    else:
        cv2.waitKey(3000)
    return 1

def _check_cvimages(imgs):
    temp = imgs.shape
    #
    try:
        assert(len(temp) == 4)
    except:
        raise Exception('图片格式不对，应为[num, channal, h, width]') 
    #
    try:
        assert(temp[3] == 1 or temp[3] == 3)
    except:
        raise Exception('通道数应为1或3！') 
    #
    return 1

def img_square(imgs):
    '''
    将所给图片组整合为最合适的正方形。4d->3d

    img = img_square(patches_pred)
    '''
    _check_cvimages(imgs)
    if np.max(imgs)<=1 and np.max(imgs)>0: 
        data=imgs*255.
    imgs=np.uint8(imgs)
    #
    num, height, width, chan = imgs.shape
    temp = int(num**0.5)
    img_out_mat = np.zeros((temp * height, temp * width, chan),dtype=np.uint8)
    #
    for m in range(temp): #m行n列，m*height+n+1
        for n in range(temp):#拼接	
            img_out_mat[m*height:(m+1)*height, n*width:(n+1)*width, :] = imgs[m*temp+n,:,:,:]
    return img_out_mat

