import os
import re

import cv2
import numpy as np
import pandas as pd

import copy


def mkdir(path):
    '''make dir'''
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def calculate():

    path = 'C:/Users/Huang Lab/Desktop/CSV/'

    test_path = 'C:/Users/Huang Lab/Desktop/Confocal/test/'
    test_paths = [os.path.join(test_path, test_img) for test_img in os.listdir(test_path) if '.DS_Store' not in test_img]

    red_masks = []
    green_masks = []

    pre_red_masks = []
    pre_green_masks = []

    img_res = 0.69*0.69       #输入Confocal图像分辨率，从ZEN软件里查看。这个与放大倍数/视野大小有关，每次计算必须确保此值正确！！！！
    pix_res = 1.024*1.024     #确保每次拍得的图像分辨率是1024*1024.否则计算结果将有误！！！！！！！



    for test_path in test_paths:
        for test_names in os.listdir(test_path):
            if '.DS_Store' not in test_names:
                #red_masks.append(test_path + '/' + test_names + '/' + '1.1.png')
                #green_masks.append(test_path + '/' + test_names + '/' + '2.1.png')

                pre_red_masks.append(test_path + '/' + test_names + '/' + 'predict_1.1.png')
                pre_green_masks.append(test_path + '/' + test_names + '/' + 'predict_2.1.png')


   # print(pre_green_masks)
   # print(pre_red_masks)
   # print(len(pre_red_masks))
   # print(len(pre_green_masks))

    i = 1
    for ( pre_red_mask, pre_green_mask) in zip( pre_red_masks, pre_green_masks):

        image_red = cv2.imread(pre_red_mask.split('/predict_1.1.png')[0] + '/' + 'Image {}.png'.format(pre_red_mask.split('/')[-2]))
        # print(image_red.shape)
        # print(len(image_red))

        # image_green = cv2.imread(red_mask.split('/1.1.png')[0] + '/' + 'Image {}_c2png'.format(red_mask.split('/')[-2]))
        if image_red is None:
            image_red = cv2.imread(
                pre_red_mask.split('/predict_1.1.png')[0] + '/' + 'Image {}_c1.png'.format(pre_red_mask.split('/')[-2]))
            if image_red is None:
                image_red = cv2.imread(
                    pre_red_mask.split('/predict_1.1.png')[0] + '/' + 'Snap-{}_c1.png'.format(
                        pre_red_mask.split('/')[-2]))

            # if image_red is None:
            #     image_red = cv2.imread(
            #         pre_red_mask.split('/predict_1.1.png')[0] + '/' + 'Snap-{}_c1.png'.format(
            #             pre_red_mask.split('/')[-2]))

        #red_img = cv2.imread(red_mask)
        pre_red_img = cv2.imread(pre_red_mask)

        #green_img = cv2.imread(green_mask)
        pre_green_img = cv2.imread(pre_green_mask)


        # 二值化
        # _, red_img = cv2.threshold(
        #     cv2.cvtColor(red_img.copy(), cv2.COLOR_BGR2GRAY),  # 转换为灰度图像,
        #     127, 255,  # 大于130的改为255  否则改为0
        #     cv2.THRESH_BINARY)  # 黑白二值化

        _, pre_red_img = cv2.threshold(
            cv2.cvtColor(pre_red_img.copy(), cv2.COLOR_BGR2GRAY),  # 转换为灰度图像,
            127, 255,  # 大于130的改为255  否则改为0
            cv2.THRESH_BINARY)  # 黑白二值化

        # _, green_img = cv2.threshold(
        #     cv2.cvtColor(green_img.copy(), cv2.COLOR_BGR2GRAY),  # 转换为灰度图像,
        #     127, 255,  # 大于130的改为255  否则改为0
        #     cv2.THRESH_BINARY)  # 黑白二值化

        _, pre_green_img = cv2.threshold(
            cv2.cvtColor(pre_green_img.copy(), cv2.COLOR_BGR2GRAY),  # 转换为灰度图像,
            127, 255,  # 大于130的改为255  否则改为0
            cv2.THRESH_BINARY)  # 黑白二值化

        # contours_red, _ = cv2.findContours(red_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours_green, _ = cv2.findContours(green_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        pre_contours_red, _ = cv2.findContours(pre_red_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pre_contours_green, _ = cv2.findContours(pre_green_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        # 1. 总血管个数
        #N_green = len(contours_green)
        N_green_pre = len(pre_contours_green)

        # 2. 血管密度
        #p_green = N_green / 1
        p_green_pre = N_green_pre / 1

        # 3. 平均血管面积
        #avg_green = np.sum(green_img / 255.) / N_green
        avg_green_pre = (np.sum(pre_green_img / 255.) / N_green_pre)*img_res/pix_res


        # 4. 总渗透面积
        #area_red = np.sum(red_img / 255.)
        area_red_pre = (np.sum(pre_red_img / 255.))*img_res/pix_res

        # 5. 平均渗透面积
        #avg_area = area_red / N_green
        avg_area_pre = (area_red_pre / N_green_pre)*img_res/pix_res

        # 6. 平均渗透面积比
        #avg_area_red = area_red / 1
        avg_area_red_pre = area_red_pre / 1

        # 7.01 单个血管面积，单个渗透面积
        area_pre_green = []
        area_pre_red = []

        # 7.02 单个血管周长
        perimeter_pre_green = []

        #7.03 单个血管周长面积比
        perimeter_ratio_green = []

        #7.04 单个血管横纵比

        aspect_ratio = []


        # 7. 单个血管渗透面积比
        #area_ratio = []
        area_ratio_pre = []

        # 8. 单个血管渗透面积差
        #diff_val = []
        diff_val_pre = []

        # 9. 单个血管渗透量
        #num_vess = []
        num_vess_pre = []

        # 10. 单个血管渗透率
        #per_vess = []
        per_vess_pre = []


        #zeros_image = np.zeros((pre_red_img.shape[0], pre_red_img.shape[1]))

        # # for c in contours_red:
        # #     x, y, w, h = cv2.boundingRect(c)
        # #
        # #     c = copy.copy(zeros_image)
        # #     c[y:(y + h), x:(x + w)] = 1
        # #     a = copy.copy(red_img) * c
        # #
        # #     green_img_k = copy.copy(zeros_image)
        #
        #     for c1 in contours_green:
        #         x1, y1, w1, h1 = cv2.boundingRect(c1)
        #
        #         d = copy.copy(zeros_image)
        #         d[y1:(y1 + h1), x1:(x1 + w1)] = 1
        #         b = copy.copy(green_img) * d
        #
        #         sum = np.sum(a * b)
        #
        #         if sum > 0:
        #             green_img_k = green_img_k + b
        #
        #     green_img_k = np.clip(green_img_k, 0, 255)
        #
        #     if np.sum(green_img_k) > 0:
        #         area_ratio.append(np.sum(a / 255) / np.sum(green_img_k / 255))
        #         diff_val.append(np.sum(a / 255) - np.sum(green_img_k / 255))
        #
        #         M_red = np.sum(image_red * np.dstack([a] * 3) / 255) / (np.sum(a / 255))
        #         M_green = np.sum(image_red * np.dstack([green_img_k] * 3) / 255) / np.sum(green_img_k / 255)
        #
        #         num_vess.append(np.sum(a / 255) * M_red - np.sum(green_img_k / 255) * M_green)
        #         per_vess.append(1 - ((np.sum(green_img_k / 255) * M_green) / (np.sum(a / 255) * M_red)))

        pre_zeros_image = np.zeros((pre_red_img.shape[0], pre_red_img.shape[1]))

        for c in pre_contours_red:
            zhouchang = 0
            peri_ratio = 0
            x, y, w, h = cv2.boundingRect(c)

            c = copy.copy(pre_zeros_image)
            c[y:(y + h), x:(x + w)] = 1
            a = copy.copy(pre_red_img) * c

            green_img_k = copy.copy(pre_zeros_image)

            for c1 in pre_contours_green:
                x1, y1, w1, h1 = cv2.boundingRect(c1)

                d = copy.copy(pre_zeros_image)
                d[y1:(y1 + h1), x1:(x1 + w1)] = 1
                b = copy.copy(pre_green_img) * d


                sum = np.sum(a * b)

                if sum > 0:
                    green_img_k = green_img_k + b
                    zhouchang = zhouchang + cv2.arcLength(c1,True)
                    peri_ratio = peri_ratio + float(w1) / h1


            green_img_k = np.clip(green_img_k, 0, 255)




            if np.sum(green_img_k) > 0:

                M_red = np.sum(image_red * np.dstack([a] * 3) / 255) / (np.sum(a / 255))
                M_green = np.sum(image_red * np.dstack([green_img_k] * 3) / 255) / np.sum(green_img_k / 255)
                vess = (1 - ((np.sum(green_img_k / 255) * M_green) / (np.sum(a / 255) * M_red)))
                if -0.2 <= vess <= 1 :

                    num_vess_pre.append(np.sum(a / 255) * M_red - np.sum(green_img_k / 255) * M_green)
                    area_ratio_pre.append(np.sum(a / 255) / np.sum(green_img_k / 255))
                    diff_val_pre.append((np.sum(a / 255) - np.sum(green_img_k / 255))*img_res/pix_res)
                    per_vess_pre.append(vess)
                    area_pre_green.append((np.sum(green_img_k / 255))*img_res/pix_res)
                    area_pre_red.append((np.sum(a / 255))*img_res/pix_res)
                    perimeter_pre_green.append(zhouchang)
                    perimeter_ratio_green.append(zhouchang/((np.sum(green_img_k / 255))*img_res/pix_res))
                    aspect_ratio.append(peri_ratio)

        mkdir(path + 'calculate')

        # 1
        # dataDF = pd.DataFrame()
        # dataDF['name'] = [str(red_mask.split('1.')[0])]
        #
        # dataDF['N_green'] = [N_green]
        # dataDF['p_green'] = [p_green]
        # dataDF['avg_green'] = [avg_green]
        # dataDF['area_red'] = [area_red]
        # dataDF['avg_area'] = [avg_area]
        # dataDF['avg_area_red'] = [avg_area_red]
        #
        # dataDF['area_ratio'] = area_ratio
        #
        # dataDF['diff_val'] = diff_val
        # dataDF['num_vess'] = num_vess
        # dataDF['per_vess'] = per_vess


        # dataDF = pd.concat([pd.DataFrame({'name': [str(red_mask.split('1.')[0])]}),
        #                     pd.DataFrame({'N_green': [N_green]}),
        #                     pd.DataFrame({'p_green': [p_green]}),
        #                     pd.DataFrame({'avg_green': [avg_green]}),
        #                     pd.DataFrame({'area_red': [area_red]}),
        #                     pd.DataFrame({'avg_area': [avg_area]}),
        #                     pd.DataFrame({'avg_area_red': [avg_area_red]}),
        #
        #                     pd.DataFrame({'area_ratio': area_ratio}),
        #                     pd.DataFrame({'diff_val': diff_val}),
        #                     pd.DataFrame({'num_vess': num_vess}),
        #                     pd.DataFrame({'per_vess': per_vess}),
        #                    ],
        #                    axis=1)

       # dataDF.to_csv(path + "calculate/act_{}.csv".format(i), mode='a', index=False)

        # # 2
        # dataDF_pre = pd.DataFrame()
        # dataDF_pre['name'] = [str(red_mask.split('1.')[0])]
        #
        # dataDF_pre['N_green'] = [N_green_pre]
        # dataDF_pre['p_green'] = [p_green_pre]
        # dataDF_pre['avg_green'] = [avg_green_pre]
        # dataDF_pre['area_red'] = [area_red_pre]
        # dataDF_pre['avg_area'] = [avg_area_pre]
        # dataDF_pre['avg_area_red'] = [avg_area_red_pre]
        #
        # dataDF_pre['area_ratio'] = area_ratio_pre
        #
        # dataDF_pre['diff_val'] = diff_val_pre
        # dataDF_pre['num_vess'] = num_vess_pre
        # dataDF_pre['per_vess'] = per_vess_pre

        # 改过的算法 #


        # 1.总血管个数
        num_vess = len(per_vess_pre)
        # 2.血管密度
        p_num_vess = num_vess/img_res
        # 3.平均血管面积
        avg_vess_area = np.sum(area_pre_green)/num_vess
        # 4.平均血管周长
        avg_vess_prei = np.sum(perimeter_pre_green)/num_vess
        # 5.平均血管周长面积比
        avg_prei_ratio = np.sum(perimeter_ratio_green)/num_vess
        # 6.平均血管横纵比
        avg_aspect_ratio = np.sum(aspect_ratio)/num_vess
        # 7.总血管面积
        tol_vess_area = np.sum(area_pre_green)
        # 8.总渗透面积
        tol_pen_area = np.sum(area_pre_red)
        # 9.平均渗透面积
        avg_pen_area = tol_pen_area/num_vess
        # 10.平均渗透面积比
        avg_area_ratio = np.sum(area_ratio_pre)/num_vess
        # 11.平均渗透面积差
        avg_area_diff = np.sum(diff_val_pre)/num_vess
        # 12.平均血管渗透量
        avg_num_vess = np.sum(num_vess_pre)/num_vess
        # 13.平均血管渗透率
        avg_per_vess = np.sum(per_vess_pre) / num_vess



        dataDF_pre = pd.concat([pd.DataFrame({'文件': [str(pre_red_mask.split('1.')[0])]}),
                                pd.DataFrame({'1.总血管个数': [num_vess]}),
                                pd.DataFrame({'2.血管密度': [p_num_vess]}),
                                pd.DataFrame({'3.平均血管面积': [avg_vess_area]}),
                                pd.DataFrame({'4.平均血管周长': [avg_vess_prei]}),
                                pd.DataFrame({'5.平均血管周长面积比': [avg_prei_ratio]}),
                                pd.DataFrame({'6.平均血管横纵比': [avg_aspect_ratio]}),
                                pd.DataFrame({'7.总血管面积': [tol_vess_area]}),
                                pd.DataFrame({'8.平均渗透面积': [avg_pen_area]}),
                                pd.DataFrame({'9.总渗透面积': [tol_pen_area]}),
                                pd.DataFrame({'10.平均渗透面积比': [avg_area_ratio]}),
                                pd.DataFrame({'11.平均渗透面积差': [avg_area_diff]}),
                                pd.DataFrame({'12.平均血管渗透量': [avg_num_vess]}),
                                pd.DataFrame({'13.平均血管渗透率': [avg_per_vess]}),


                                pd.DataFrame({'14.单个血管面积': area_pre_green}),
                                pd.DataFrame({'15.单个血管周长': perimeter_pre_green}),
                                pd.DataFrame({'16.单个血管周长面积比': perimeter_ratio_green}),
                                pd.DataFrame({'17.单个血管横纵比': aspect_ratio}),
                                pd.DataFrame({'18.单个血管渗透面积': area_pre_red}),
                                pd.DataFrame({'19.单个血管渗透面积比': area_ratio_pre}),
                                pd.DataFrame({'20.单个血管渗透面积差': diff_val_pre}),
                                pd.DataFrame({'21.单个血管渗透量': num_vess_pre}),
                                pd.DataFrame({'22.单个血管渗透率': per_vess_pre}),
                               ],
                               axis=1)

        dataDF_pre.to_csv(pre_red_mask.split('/predict_1.1.png')[0] + "/pre_{}.csv".format(i), mode='a', index=False,encoding='utf_8_sig')
        i = i + 1





if __name__ == "__main__":
    calculate()
