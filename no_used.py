import cv2;
import numpy as np
import matplotlib.pyplot as plt
import os;
import sys;
import cv2
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import TrainDatasetFromFolder
from pylab import *
from util.get_perspective import four_point_transform
mpl.rcParams['font.sans-serif'] = ['SimHei']

def rad(x):     # 角度转换
    return x * np.pi / 180

def transform(img):  #转换鸟瞰图
    # 扩展图像，保证内容不超出可视范围
    img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, 0)
    w, h = img.shape[0:2]

    anglex = 16
    angley = 3
    anglez = 380 #是旋转
    fov = 142

    # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))


    r =  np.array([[ 0.9986295,   0          ,0.05233596  ,0        ],
    [ 0.01442574  ,0.9612617  ,-0.2752596   ,0        ],
    [-0.05030855 ,-0.27563736  ,0.9599443   ,0        ],
    [ 0          ,0          ,0          ,1        ]]
    )
    # 四对点的生成
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
            [w, 0],
            [0, h],
            [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)
    result = cv2.warpPerspective(img, warpR, (h, w))[130:620,257:700]
    return result

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


class MLP(nn.Module):
    def __init__(self, input_size=3, common_size=1):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(3,3),stride=1,padding=1),
            nn.Linear(200, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 200)
        )
    
    def forward(self, x):
        return self.layer(x)

def train():
    loss_func = nn.MSELoss()
    net = MLP()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    train_set = TrainDatasetFromFolder('./data/train/', crop_size=200, upscale_factor=1) #训练集导入
    train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=1, shuffle=True) #训练集制作
    for epoch in range(70):
        optimizer = torch.optim.Adam(net.parameters(), lr=0.005*(0.98**epoch))
        for target, ir, img in train_loader:
            target = cv2.resize(target.detach().numpy(), (img.shape[2], img.shape[3]))
            target = torch.tensor(target, dtype=torch.float32).reshape((img.shape[1], img.shape[0],1))
            predict = net(img)
            optimizer.zero_grad()
            loss = loss_func(predict, target)
            loss.backward()
            optimizer.step()
    torch.save(net.state_dict(), './net.pkl')

if __name__ == "__main__":
    train()
    img = cv2.imread('./img/origin.jpg')
    #target_img = cv2.imread('./img/target2.bmp')
    #img = cv2.resize(img, (target_img.shape[1], target_img.shape[0]))
    #cv2.imwrite('./img/raw3.bmp',img)
    rows,cols,channel=img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    net = MLP()
    net.load_state_dict(torch.load('./net.pkl'))

    state = torch.tensor(img, dtype=torch.float32)

    c = net(state)
    cv2.imshow('gray', gray_img)
    c = np.rint(c.detach().numpy()).astype('uint8')
    cv2.imshow('combine', c)
    while 1:    
        if cv2.waitKey(0) == 0: break