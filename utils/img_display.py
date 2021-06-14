import numpy as np
from PIL import Image
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
    将三维RGB data保存成png图片。
    save_pic(temp,'1','test/test1')，
    '''
    assert (len(data.shape)==3) #channels, height, width
    assert (data.shape[0] ==1 or data.shape[0] == 3)
    
    if np.max(data)<=1: data=data*255.
    data=np.uint8(data)
    
    if data.shape[0]==1:
        img = Image.fromarray(data[0,:])  #黑白图片
    else:
        img1= Image.fromarray(data[0,:])
        img2= Image.fromarray(data[1,:])
        img3= Image.fromarray(data[2,:])
        img = Image.merge('RGB', (img1, img2, img3))
    
    try:  
        prepare_path(filedir)
        temp = './'+ filedir + filename + '.png'
        img.save(temp)  #【】【】【】】【】】【】】【】】【】】【】】【】】【】】【】】【】】【】】【】
        print(f'\t----image saved in {temp}')
    except:
        print('\t----file dir error')


def show_pic(data,windowname = 'default',showtype='freeze'):
    '''
    展示三维RGB矩阵图片，格式为numpy格式
    show_pic(pic,"asd","freeze") 冻结型显示
    show_pic(pic,"asd","freedom")自由型显示
    '''
    try:
        assert (len(data.shape)==3) 
    except:
        data = data[np.newaxis, :, :]

    try:
        assert (data.shape[0] ==1 or data.shape[0] == 3)    #通道数，高，宽
    except:
        assert (data.shape[2] ==1 or data.shape[2] == 3)    #高，宽，通道数
        data = cv2numpy(data)
    if np.max(data)<=1: data=data*255.
    data=np.uint8(data)
    #
    if data.shape[0]==1:
        img = Image.fromarray(data[0,:])                  #黑白图片,直接展示
    else:
        img1= Image.fromarray(data[0,:])
        img2= Image.fromarray(data[1,:])
        img3= Image.fromarray(data[2,:])
        img = Image.merge('RGB', (img1, img2, img3))

    img=np.asarray(img)
    if data.shape[0]==3: img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)#如果是三通道的话，需要RGB转BGR
    
    cv2.namedWindow(windowname,0)
    cv2.resizeWindow(windowname, 640, 480)
    cv2.imshow(windowname, img)

    if showtype=='freeze':
        cv2.waitKey(0)
    else:
        cv2.waitKey(3000)


def _check_image(imgs):
    temp = imgs.shape
    #
    try:
        assert(len(temp) == 4)
    except:
        raise Exception('图片格式不对，应为[num, channal, h, width]') 
    #
    try:
        assert(temp[1] == 1 or temp[1] == 3)
    except:
        raise Exception('通道数应为1或3！') 
    #
    return 1


def img_hstack(imgs):
    '''
    水平摆放图片。4d->3d
    [图][图][图][图]
    input: imgs, [num, channal, h, width]
    output: img, [channal, h, width*num]

    e.g.
    img_newline = img_hstack([patches_ori[i], patches_pre[i], patches_gt[i]])

    Warning:如果图片中有单通道也有三通道，三通道一定要放在第一个
    '''
    _check_image(imgs)

    img_out = imgs[0]
    for img in imgs[1:]:
        if img.shape[0] == 1 and img_out.shape[0] == 3:#通道扩增
            img = gray2ggg(img)
        img_out = np.concatenate((img_out, img), axis = 2)
    return img_out



def img_vstack(imgs):
    '''
    竖直摆放图片。4d->3d
    input: imgs, [num, chan, height, width]
    output: img, [chan, height*num, width]

    e.g.
    img = img_vstack([img, img_newline])

    Warning:如果图片中有单通道也有三通道，三通道一定要放在第一个
    '''
    _check_image(imgs)
    img_out = imgs[0]
    for img in imgs[1:]:
        if img.shape[0] == 1 and img_out.shape[0] == 3:#通道扩增
            img = gray2ggg(img)
        img_out = np.concatenate((img_out, img), axis = 1)
    return img_out




def img_square(imgs):
    '''
    将所给图片组整合为最合适的正方形。4d->3d
    input: imgs, [num, chan, height, width]
    output: img, [chan, int((height*width)**0.5), int((height*width)**0.5)]

    img = img_square(patches_pred)
    '''
    _check_image(imgs)
    num, chan, height, width = imgs.shape
    temp = int(num**0.5)
    img_out_mat = np.zeros((chan, temp * height, temp * width))

    for m in range(temp): #m行n列，m*height+n+1
        for n in range(temp):#拼接	
            img_out_mat[:, m*height:(m+1)*height, n*width:(n+1)*width] = imgs[m*temp+n,:,:,:]
    return img_out_mat

