# import os
# import cv2
# import numpy as np
#
# path = '/mnt/data10t/bakuphome20210617/lz/sgd/image/'
#
# pictures = os.listdir(path)
#
# img = cv2.imread(path + '2381630480492_.pic.jpg')
#
# img[np.where(img > 127)] = 255
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
#
# for k in contours[0:]:
#     # contour = contours[k].squeeze()
#     # gray = cv2.fillPoly(gray, contour, 255)
#
#     # res4 = cv2.drawContours(draw_img4, k, -1, (0, 1, 1), cv2.FILLED)
#     gray = cv2.fillConvexPoly(gray, k, (255, 255, 255))
#
#
# cv2.imwrite('/home/lizhe/data/sgd/1.jpg', gray)
#
#
# # element = set()
# # for i in range(image.shape[0]):
# #     for j in range(image.shape[1]):
# #
# #         if image[i, j] == 255:
# #             pass
# #         else:
# #             image[i, j] = 0
# #
#
#
#
#
# # import os
# # import cv2
# # import numpy as np
# # img_path = '/mnt/data10t/bakuphome20210617/lz/sgd/33_json/img.png'
# # mask_path = '/mnt/data10t/bakuphome20210617/lz/sgd/33_json/label.png'
# #
# # np.set_printoptions(threshold=np.inf)
# #
# # img = cv2.imread(img_path)
# # mask = cv2.imread(mask_path)
# # print(img.shape)
# # print(mask.shape)
# #
# # num = set()
# #
# # # print(mask)
# #
# # for i in range(1024):
# #     for j in range(1024):
# #         # print(mask[i, j, :])
# #         num.add(tuple(mask[i, j, :]))
# #
# # print(num)

