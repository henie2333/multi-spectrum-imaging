import cv2
import numpy as np
import sys
import math

def RGBtoHIS(RGBimg): #定义转换函数
    rows = int(RGBimg.shape[0])
    cols = int(RGBimg.shape[1])
    B, G, R = cv2.split(RGBimg)
    # 归一化到[0,1]
    B = B / 255.0
    G = G / 255.0
    R = R / 255.0
    HISimg = RGBimg.copy()
    H, S, I = cv2.split(HISimg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((R[i, j] - G[i, j]) + (R[i, j] - B[i, j]))
            den = np.sqrt((R[i, j] - G[i, j]) ** 2 + (R[i, j] - B[i, j]) * (G[i, j] - B[i, j]))
            theta = float(np.arccos(num / den))
            if den == 0:
                H = 0
            elif B[i, j] <= G[i, j]:
                H = theta
            else:
                H = 2 * 3.14169265 - theta
            min_RGB = min(min(B[i, j], G[i, j]), R[i, j])
            summ = B[i, j] + G[i, j] + R[i, j]
            if summ == 0:
                S = 0
            else:
                S = 1 - 3 * min_RGB / summ
            H = H / (2 * 3.14159265)
            I = summ / 3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            HISimg[i, j, 0] = H * 255
            HISimg[i, j, 1] = S * 255
            HISimg[i, j, 2] = I * 255
    return HISimg

if __name__ == "__main__": 
    filepath='./img/raw1.jpg' #文件路径
    RGBimg=cv2.imread(filepath,1) #使用opencv读取RGB图像

    cv2.namedWindow('MSS',0)
    cv2.resizeWindow('MSS',500,500)
    cv2.imshow('MSS',RGBimg)

    HISimg=RGBtoHIS(RGBimg) #调用函数RGB to HIS
    cv2.namedWindow('HISimg',1)
    cv2.resizeWindow('HISimg',500,500)
    cv2.imshow('HISimg', HISimg)

    cv2.imwrite('HISimg.tif',HISimg, [cv2.IMWRITE_JPEG_QUALITY, 2]) #保存转换结果
    print(HISimg.shape)

    cv2.waitKey(0)
    cv2.destroyAllWindows()