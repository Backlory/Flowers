import os
import numpy as np
import cv2

if __name__ == '__main__':
    '''
    将原始数据的文件名转换为对应的标签号
    '''
    
    '''
    dir1 = 'data\\data_test\\data_train_expend'
    dir2 = 'data\\data_test\\data_train_ori'
    hashlist1 = []
    filename_list = []
    for filename in os.listdir(dir1):
        print(filename)
        temp = cv2.imread(dir1+'\\'+filename)
        hashlist1.append(hash(str(temp)))
        filename_list.append(str(dir1+'\\'+filename))
    
    hashlist2 = []
    for filename in os.listdir(dir2):
        print(filename)
        temp = cv2.imread(dir2+'\\'+filename)
        hashlist2.append(hash(str(temp)))

    for i in range(len(hashlist1)):
        if hashlist1[i] not in hashlist2:
            print(filename_list[i])
            os.rename(filename_list[i], filename_list[i]+'66')
    '''
    
    dir1 = 'data\\data_test'
    for filename in os.listdir(dir1):
        os.rename(dir1+'\\'+filename, dir1+'\\'+filename[:-2])