import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader


#Fetching dataframe of file names for tran and val

##train
t = "Semantic segmentation"
data = pd.read_csv('/media/keyurparalkar/310230E03A1B6D12/keras_tutorial/'+t+'/VOCdevkit/VOC2012/ImageSets/Main/train.txt',header=None)
##Train annotations:
