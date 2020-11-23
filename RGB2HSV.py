import cv2
import numpy as np


img = cv2.imread('./img/raw.jpg')

while(1):
    cv2.imshow("capture",img)     #显示读取出来的图像
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      #把RGB模型转换成HSV模型
    cv2.imshow("HSV",hsv)    #显示转换后的图像
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destoryAllwindows()