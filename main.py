import numpy as np
import cv2


def calculate(contours):
    area = []
    for c in contours:
        area.append(cv2.contourArea(c))
    return np.array(area)


def darker(gray_img,M,img,binary):
    processed_img = gray_img[:]
    for i in range(row):
        for j in range(col):
            light = binary[i][j]
            if light==255:
                processed_img[i][j] = (255+gray_img[i][j])//2
            elif light==0:
                processed_img[i][j] = gray_img[i][j]//2
    return processed_img

def transform(processed_img,M,img,binary):
    point_map = [[] for i in range(len(M))]
    for j in range(col):
        for i in range(row):   
            for id,m in enumerate(M):
                    if cv2.pointPolygonTest(m, (j,i) ,False) > 0:
                        point_map[id].append((i,j))
    light_list = []
    f_list = []
    for i,each in enumerate(point_map[:-1]):
        R = img[:,:,2].take([i*j for i,j in each])
        G = img[:,:,1].take([i*j for i,j in each])
        B = img[:,:,0].take([i*j for i,j in each])
        color_list = [np.mean(R),np.mean(G),np.mean(B)]
        junzhi = np.mean(color_list)
        fangcha = np.var(color_list)
        f_list.append(fangcha)
        light = np.mean(binary.take([i*j for i,j in each]))
        light_list.append(light)
        if light > 20 and fangcha < 50 and junzhi <220:
            for i,j in each:
                processed_img[i,j] = (processed_img[i,j]*2-255)//2
    R = img[:,:,2]
    G = img[:,:,1]
    B = img[:,:,0]
    for j in range(col):
        for i in range(row):   
            if np.var([R[i,j], G[i,j],B[i,j]]) < 23 and 134<np.mean([R[i,j], G[i,j],B[i,j]])<197 and processed_img[i,j]>100:
                processed_img[i,j] = (processed_img[i,j]*2-255)//2
            if i<153 and j>259 and processed_img[i,j] > 120:
                processed_img[i,j] = (processed_img[i,j]*2-255)//2
    return processed_img

def getposlight(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print('y-x:', [y, x]) 
        print('light:',process_img[y, x]) 
        print('bgr:',img[y, x])     #   B,G,R   

if __name__ == "__main__":
    img = cv2.imread('./img/origin.jpg')    # B,G,R
    img = cv2.resize(img, (318,239))
    row,col,channel = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _,binary = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)
    for i in range(row):
        for j in range(col):
            if binary[i][j]>0 and i <=row/3:
                binary[i][j] = 0
    #ret, process_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    process_img = cv2.Canny(img,30,90)
    process_img = cv2.GaussianBlur(process_img, (5,5), 0)
    contours,hierarchy = cv2.findContours(process_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(img,contours,-1,(0,255,0))
    area = calculate(contours)
    index=area.argsort()[::-1]
    contours_max = [contours[i] for i in index]
    #M = [cv2.convexHull(c) for c in contours_max]
    process_img = darker(gray_img, contours_max,img,binary)
    process_img = transform(gray_img, contours_max,img,binary)
    process_img = cv2.GaussianBlur(process_img, (7,7), 0)
    cv2.imwrite('./gray.jpg',process_img)
    cv2.imshow('./origin.jpg', img)
    cv2.setMouseCallback('./gray.jpg', getposlight)
    while 1:
        if cv2.waitKey() == 25:
            cv2.destroyAllWindows()


