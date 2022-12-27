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
import win32api, win32con
import random

mymodel = torch.load('resnet_49.pth')
print(mymodel)
rannum = random.randint(0, 2999)
strnum = str(rannum)
while len(strnum) < 4:
    strnum = '0' + strnum
img_path = r'E:\face_classfication\BitmojiDataset_Sample\trainimages'
img_path = img_path + '/' + strnum + '.jpg'
img_pil = Image.open(img_path)
img_pil = img_pil.resize((224, 224))
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img_pil)
img_tensor=img_tensor.resize(1,3,224,224)
print(img_tensor.shape)
img_tensor = img_tensor.cuda()
mymodel.eval()
output = mymodel(img_tensor)
print(output)
evl = 'female' if output[0][0] > output[0][1] else 'male'
print(evl)
win32api.MessageBox(0, evl, "结果",win32con.MB_OK)
img_pil.show()
