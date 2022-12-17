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

mymodel = torch.load('reg_49.pth')
print(mymodel)
img_path = r'E:\face_classfication\BitmojiDataset_Sample\trainimages\0006.jpg'
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
