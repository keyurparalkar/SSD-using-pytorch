import torch
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import xml.etree.ElementTree as ET


#Let's create Dataset class for car classifier.

#/media/keyurparalkar/310230E03A1B6D12/keras_tutorial/Semantic segmentation/VOCdevkit/VOC2012

class PascalVocDataset(Dataset):
    def __init__(self,annotation_path,txt_file_name,root_dir,transform=None):
        """
            annotation_path = path to annotation xml files
            txt_file_name = path to .txt for images.
            root_dir = dataset's root_dir path.
            transform = transformations to be applied to each image
        """
        self.image_names = pd.read_csv(txt_file_name,delim_whitespace=True,header=None)
        self.annotation_path = annotation_path
        self.parent_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self,idx):
        #parsing xml files and obtaining ground-truth bounding boxes
        anno_file_path = os.path.join(self.annotation_path,self.image_names[0][idx]+'.xml')
        anno_obj = ET.parse(anno_file_path)
        root = anno_obj.getroot()
        obj_first = root.find('object')
        bndbox = obj_first.find('bndbox')
        bndbox_dict = {'xmin':int(bndbox[0].text),'ymin':int(bndbox[1].text),'xmax':int(bndbox[2].text),'ymax':int(bndbox[3].text)}

        image_path = os.path.join(self.parent_dir,self.image_names[0][idx]+'.jpg')
        img = Image.open(image_path)

        #For obejct with ground_truth value as 0 mark it as -1 while filing in sample dict. below.
        sample = {'image':img,'ground_truth':self.image_names[1][idx],'bndbox':bndbox_dict}
        
        if(self.transform):
            sample['image'] = self.transform(sample['image'])

        return sample

def show_image_grid(sample_batch):
    image_batches, ground_truths,bounding_boxes = sample_batch['image'],sample_batch['ground_truth'],sample_batch['bndbox']
    print(i,image_batches.size(),ground_truths.size())
    print('Ground-truths:',ground_truths)
    print('Bounding box: ',bounding_boxes)
    grid = utils.make_grid(image_batches)
    plt.imshow(grid.numpy().transpose(1,2,0))
    plt.show()

data = PascalVocDataset(annotation_path='/media/keyurparalkar/310230E03A1B6D12/Datasets/pascal-voc/VOC2012/Annotations',txt_file_name='/media/keyurparalkar/310230E03A1B6D12/Datasets/pascal-voc/VOC2012/ImageSets/Main/car_train.txt'
            ,root_dir='/media/keyurparalkar/310230E03A1B6D12/Datasets/pascal-voc/VOC2012/JPEGImages',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))

dataset = DataLoader(data,batch_size=4,shuffle=True,num_workers=4)


if(len(sys.argv)>=2 and sys.argv[1] == "visualize"):
    i, batch = next(iter(enumerate(dataset)))
    show_image_grid(batch)

