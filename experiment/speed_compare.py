# coding=utf-8

import cv2
import time
"""
INTER_NEAREST | 最近邻插值
INTER_LINEAR | 双线性插值（默认设置）
INTER_AREA |  使用像素区域关系进行重采样
INTER_CUBIC  | 4x4像素邻域的双三次插值
INTER_LANCZOS4 |  8x8像素邻域的Lanczos插值

(480, 296) - INTER_NEAREST ==>  0.2214069366455078 ms 【】
(480, 296) - INTER_LINEAR ==>  0.467266321182251 ms   【】
(480, 296) - INTER_AREA ==>  1.6743276119232178 ms
(480, 296) - INTER_CUBIC ==>  0.8347911834716797 ms
(480, 296) - INTER_LANCZOS4 ==>  2.8498942852020264 ms
(720, 508) - INTER_NEAREST ==>  0.6176402568817139 ms  【】
(720, 508) - INTER_LINEAR ==>  1.0324881076812744 ms
(720, 508) - INTER_AREA ==>  1.0438525676727295 ms
(720, 508) - INTER_CUBIC ==>  1.4810755252838135 ms
(720, 508) - INTER_LANCZOS4 ==>  5.357002258300781 ms
"""
def tic():
    return time.time()

def toc(time_start):
    return time.time() - time_start

def check(img, size, way = 'INTER_NEAREST'):
    t = tic()
    for i in range(1000):
        temp = cv2.resize(img, size, interpolation = eval('cv2.'+way))
        #cv2.imshow(way, temp)
        #cv2.waitKey(5000)
    print(size,"-",str(way) ,"==> \t",toc(t),'ms')


if __name__ == '__main__':
    img = cv2.imread("girl.jpg")
    height, width = img.shape[:2]

    # 缩小图像
    size = (int(width*0.8), int(height*0.7))
    check(img, size, "INTER_NEAREST")
    check(img, size, "INTER_LINEAR")
    check(img, size, "INTER_AREA")
    check(img, size, "INTER_CUBIC")
    check(img, size, "INTER_LANCZOS4")
    #放大图像
    size = (int(width*1.2), int(height*1.2))
    check(img, size, "INTER_NEAREST")
    check(img, size, "INTER_LINEAR")
    check(img, size, "INTER_AREA")
    check(img, size, "INTER_CUBIC")
    check(img, size, "INTER_LANCZOS4")


