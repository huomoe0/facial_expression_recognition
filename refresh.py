import os
import random
import shutil

import splitfolders
import torch
import torchvision
from torchvision import datasets

dir = "RAF-DB"

# try:
#     shutil.rmtree("data/fane_data")
# except Exception as e:
#     print("文件夹不存在")
# splitfolders.ratio('data/fane_photos', output="data/fane_data", seed=random.randint(1, 1337), ratio=(.8, 0.2))  #  划分数据集

data_root = os.path.abspath(os.getcwd())
image_path = os.path.join(data_root, "data",  dir)
# 加载你的数据集
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=torchvision.transforms.ToTensor())

# 计算平均值
mean = 0.
for images, _ in train_dataset:
    mean += images.mean([1,2])

mean = mean / len(train_dataset)

# 计算标准差
std = 0.
for images, _ in train_dataset:
    std += ((images - mean.unsqueeze(1).unsqueeze(2))**2).mean([1,2])

std = torch.sqrt(std / len(train_dataset))

print('mean:', mean)
print('std:', std)


