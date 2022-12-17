import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
from torchvision import transforms
import random
from torchvision import models
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
from torch import nn

from model import classifer
from model import Mynn


data_path = Path("BitmojiDataset_Sample")
meta_file = data_path / 'train.csv'
df_meta = pd.read_csv(meta_file)
# print(df_meta.head())
df = pd.DataFrame()
df_meta['relative_path'] = '/' + 'trainimages' + '/' + df_meta['image_id'].astype(str)
df_meta['classID'] = df_meta['is_male'].astype(str)
df = df_meta[['relative_path', 'classID']]


# print(df.head())
class myDataSet(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)

    def __len__(self):
        return len(df)

    def __getitem__(self, idx):
        tensor_trans = transforms.ToTensor()
        img_path = self.data_path + self.df.loc[idx, 'relative_path']
        classID = self.df.loc[idx, 'classID']
        classID = 0 if classID == '-1' else 1
        img_PIL = Image.open(img_path)
        img_PIL = img_PIL.resize((224, 224))
        img_tensor = tensor_trans(img_PIL)
        return img_tensor, classID


myds = myDataSet(df, data_path)
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])
batch_size = 16
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

train_data_size = len(train_dl)
test_data_size = len(val_dl)
# random_batch = next(iter(train_dl))
# batch_data, label = random_batch
# print("Shape of datas: \n", batch_data.shape)
# print(batch_data)
# show = torchvision.transforms.ToPILImage()
# show(batch_data[0]).show()
# print("Shape of labels: \n", label.shape)
# print(label)


# 模型创建
mynn = classifer()
if torch.cuda.is_available():
    print(1)
    mynn = mynn.cuda()
# loss
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# optim
learning_rate = 0.005
optimizer = torch.optim.SGD(mynn.parameters(), lr=learning_rate)
# 设置参数
total_train_step = 0  # 训练的次数
total_test_step = 0  # 测试次数
epoch = 50
# tensorboard
writer = SummaryWriter("./regnet")
for i in range(epoch):
    print("----------------第{}轮训练开始----------------".format(i+1))
    #  训练开始
    for data in train_dl:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = mynn(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        writer.add_scalar("train_loss", loss.item(), total_train_step)
        if total_train_step % 100 == 0 or total_train_step == 0:
            print("训练次数:{}, loss:{}".format(total_train_step, loss.item()))
    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0.0
    with torch.no_grad():
        for data in val_dl:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = mynn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    total_test_step += 1
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/(test_data_size*batch_size)))
    writer.add_scalar("total_test_loss", total_test_loss, total_test_step)
    writer.add_scalar("total_acc", total_accuracy/(test_data_size*batch_size), total_test_step)
    torch.save(mynn, "reg_{}.pth".format(i))
    print("模型已保存")

writer.close()
