import os
import cv2
import numpy as np


data_path = r'C:\Users\Huang Lab\Desktop\Confocal_Image\ct26\47'   #存放位置
Images = os.listdir(data_path)

# for Image in Images:

img1 = cv2.imread(data_path + "/" + 'FTn.png')
img2 = cv2.imread(data_path + "/" + 'Vessel.png')

img3 = cv2.imread(data_path + "/" + 'Image 47_c1.png')
img4 = cv2.imread(data_path + "/" + 'Image 47_c2.png')


dst1 = cv2.bitwise_and(img1,img2)
dst2 = cv2.addWeighted(img3,1,img4,1,0)



cv2.imwrite(data_path + "/" +"原图.png",dst2)


result = cv2.cvtColor(dst1,cv2.COLOR_BGR2BGRA)

for i in range(0,dst1.shape[0]):
    for j in range(0,dst1.shape[1]):
        if dst1[i,j,0] > 200 and dst1[i,j,1]>200 and dst1[i,j,2]>200:
            result[i,j,3] = 0
cv2.imwrite(data_path + "/" + "/勾画.png",result)
#cv2.imwrite(data_path + "/勾画.png",result,[int(cv2.IMWRITE_PNG_COMPRESSION),0])

