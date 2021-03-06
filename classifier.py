import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time

from load_data_classification import *

class resnetRcnn(nn.Module):
    def __init__(self,model):
        super(resnetRcnn, self).__init__()

        #copying feature extraction part
        self.features = nn.Sequential(*list(model.children())[:-2])

        #Classification:
        self.classify_conv_1 = nn.Conv2d(in_channels=512,out_channels=2,kernel_size=3,stride=2,padding=1) #one for car other for background
        
        self.conv = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=2,padding=1)
        self.classify_conv_2 = nn.Conv2d(in_channels=256,out_channels=2,kernel_size=3,stride=2,padding=1)

        #bounding_box regression:
        self.bnd_conv_1 = nn.Conv2d(in_channels=512,out_channels=4,kernel_size=3,stride=2,padding=1)

        self.bnd_conv_2 = nn.Conv2d(in_channels=256,out_channels=4,kernel_size=3,stride=2,padding=1)

    def forward(self,x):
        t = self.features(x)
        # t = t.view(t.size(0),-1)
        # x = F.relu(self.classify_fc_1(t))
        # x = F.relu(self.classify_fc_2(x))

        # y = self.bb_fc_1(t)
        # y = self.bb_fc_2(y)

        # size=(4x4)
        m = self.classify_conv_1(t)   

        # size=(2x2)
        p = self.conv(t)
        p = self.classify_conv_2(p)

        # size=(4x4)
        n = self.bnd_conv_1(t)

        # size=(2x2)
        q = self.conv(t)
        q = self.bnd_conv_2(q)

        #at last we can concat results from different sizes that is from 4x4 and 2x2    
        print(p.size(),q.size())
        return p,q


device = torch.device('cuda:0')

def train_model(model,  optimizer, scheduler, criterion_classification=None, criterion_bndbox=None,mode='train', num_epochs=10):
    since = time.time()

    epoch_loss = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)

        running_loss = 0.0
        running_corrects = 0

        if(mode=='train'):
            scheduler.step()
            model.train()
        else:
            model.eval()

        for i,batch in enumerate(dataset):
            inputs = batch['image'].to(device)
            labels = batch['ground_truth'].to(device)
            bndbox_xmin = batch['bndbox']['xmin'].to(device)
            bndbox_ymin = batch['bndbox']['ymin'].to(device)
            bndbox_xmax = batch['bndbox']['xmax'].to(device)
            bndbox_ymax = batch['bndbox']['ymax'].to(device)
            bndbox = {'xmin':bndbox_xmin,'ymin':bndbox_ymin,'xmax':bndbox_xmax,'ymax':bndbox_ymax}

            # print(bndbox)
            optimizer.zero_grad()

            if(mode == 'train'):
                pred_label,pred_box = model(inputs)
                print(pred_label,pred_box)
                # print("PREDICTION ==== \n:",pred_label,pred_box)
                # print("LABELS ====== \n:",labels)
                # _, preds = torch.max(outputs)
                # loss_classification = criterion_classification(pred_label,labels)
                # loss_bndbox = criterion_bndbox(pred_box,bndbox)

            # running_loss += loss*inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)

        # epoch_loss = running_loss / len(dataset)
        # epoch_accuracy = running_corrects.double() / len(dataset)

        # print('LOSS: {.4f} ACC: {.4f}',epoch_loss,epoch_accuracy)
        # print("Time elpased: ",time.time()-since)


    return model

model = models.resnet18(pretrained=True)
model_res = resnetRcnn(model)

model_res = model_res.to(device)

criterion_classify = nn.CrossEntropyLoss()
# criterion_bndbox = nn.L2
optimizer_res = optim.SGD(model_res.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_res,step_size=7,gamma=0.1)

train_model(model=model_res,criterion_classification=criterion_classify, optimizer=optimizer_res,scheduler=exp_lr_scheduler)