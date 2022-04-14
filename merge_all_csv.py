import os
import pandas as pd


base = r'C:\Users\Huang Lab\Desktop\231 材料\HH'        # 所有数据的存放位置，建议不变


tumor_names = os.listdir(base)                  #打开base文件夹，提取所有肿瘤类型子文件夹

for tumor in tumor_names:  # 遍历所有肿瘤文件夹
    csv_list = []
    tumor_path = base + "/" + tumor  # 确定绝对路径
    Image_names = os.listdir(tumor_path)  # 打开肿瘤文件夹，提取所有图像文件夹
    for Image in Image_names:  # 遍历所有图像文件夹
        Image_path = tumor_path + "/" + Image  # 确定图像文件夹绝对路径
        file_names = os.listdir(Image_path)  # 打开图像文件夹，提取所有图片，包括预测图，原始图和“.csv文件”
        for file in file_names:  # 遍历所有文件
            file_name = Image_path + "/" + file
            if file.split(".")[1] == "csv":  # 判断是否为“.csv文件”
                print(file_name)
                df = pd.read_csv(file_name)  # 读取CSV
                csv_list.append(df)
    result = pd.concat(csv_list)  # 合并CSV
    result.to_csv("C:/Users/Huang Lab/Desktop/231" + "/" + f"{tumor}.csv", mode='a', index=False,
                  encoding='utf_8_sig')  # 输出合并后的csv






