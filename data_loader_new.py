import os
# import random
# import cv2
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt

import torch
import torch.utils.data as data

from data_utils import get_transform, make_one_hot
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset_self(data.Dataset):
    def __init__(self, data_dir):
        super(Dataset_self, self).__init__()

        self.root_path = data_dir + r'\t1'
        self.filenames = [x for x in sorted(os.listdir(self.root_path))]  # 获取该路径下的文件并进行了排序

        # self.transform = get_transform(convert=True, normalize= True)
        self.transform = get_transform(convert=True, normalize=False)  # 对图像和标签数据进行预处理转换，第一个参数：将图像转换成张量，第二个参数表示是否对图像进项标准化处理
        self.label_transform = get_transform()
        self.file_name = ""

    def __getitem__(self, index):
        # index 根据给定index索引获取对应位置的文件路径，并通过一系列的字符串替换操作来得到不同类型的文件路径。
        # filepath 表示原始图片路径
        # filepath2表示转换后的图片路径
        # filepath_label 表示标签图片的路径
        # print(self.filenames[index])
        filepath = os.path.join(self.root_path, self.filenames[index])
        # filepath2 = filepath.replace('images', 't2').replace('t1', 't2')
        filepath2 = filepath.replace('t1', 't2')
        # print("filepath2="+filepath2)
        # filepath_label = filepath.replace('images', 'label').replace('t1', 'label')
        filepath_label = filepath.replace('images', 'label').replace(r'\t2', '').replace(r'\t1', '')
        # print("filepath_label="+filepath_label)


        ## without resize
        img = self.transform(Image.open(filepath).convert('RGB'))
        img2 = self.transform(Image.open(filepath2).convert('RGB')) # RGB形式
        label_img = Image.open(filepath_label).convert('1')
        label = self.label_transform(label_img)  # 二值化形式的张量
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)   # 独热编码处理
        # labels = label.to(device, dtype=torch.float)
        # labels = torch.argmax(labels, 1).unsqueeze(1).float().cuda()
        return img, img2, label, self.filenames[index]

    def __len__(self):
        # 返回数据集的长度，即文件列表的长度作为数据集的样本总数
        return len(self.filenames)

