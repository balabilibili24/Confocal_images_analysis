import os
import cv2
import numpy as np


data_path = r'C:\Users\Huang Lab\Desktop\Confocal_Image\ct26\47'  #存放位置
Images = os.listdir(data_path)

# for Image in Images:

img = cv2.imread(data_path + "/" + 'predict_1.1.png')
# # img[np.where(img >= 64)] = 255
# # img[np.where(img < 64)] = 0
#
#cv2.imwrite(data_path + '/test.png', img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

img1 = np.zeros(img.shape,np.uint8) + 255

img1 = cv2.drawContours(img1,contours,-1,(0,0,255),2)

result = cv2.cvtColor(img1,cv2.COLOR_BGR2BGRA)

for i in range(0,img1.shape[0]):
    for j in range(0,img1.shape[1]):
        if img1[i,j,0] > 200 and img1[i,j,1]>200 and img1[i,j,2]>200:
            result[i,j,3] = 0
cv2.imwrite(data_path + "/FTn.png",result,[int(cv2.IMWRITE_PNG_COMPRESSION),0])


