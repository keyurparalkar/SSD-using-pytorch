import torch
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn import preprocessing
import sys
import xml.etree.ElementTree as ET


#Let's create Dataset class for car classifier.

#/media/keyurparalkar/310230E03A1B6D12/keras_tutorial/Semantic segmentation/VOCdevkit/VOC2012

# class PascalVocDataset(Dataset):
#     def __init__(self,annotation_path,txt_file_name,root_dir,transform_ips=None,transform_y=None):
#         """
#             annotation_path = path to annotation xml files
#             txt_file_name = path to .txt for images.
#             root_dir = dataset's root_dir path.
#             transform = transformations to be applied to each image
#         """
#         self.image_names = pd.read_csv(txt_file_name,delim_whitespace=True,header=None)
#         self.annotation_path = annotation_path
#         self.parent_dir = root_dir
#         self.transform_ips = transform_ips
#         self.transform_y = transform_y

#     def __len__(self):
#         return len(self.image_names)

#     def __getitem__(self,idx):
#         #parsing xml files and obtaining ground-truth bounding boxes
#         anno_file_path = os.path.join(self.annotation_path,self.image_names[0][idx]+'.xml')
#         anno_obj = ET.parse(anno_file_path)
#         root = anno_obj.getroot()
#         obj_first = root.find('object')
#         bndbox = obj_first.find('bndbox')
#         bndbox_list = [int(bndbox[0].text), int(bndbox[1].text), int(bndbox[2].text), int(bndbox[3].text)]
#         bndbox_list = np.array(bndbox_list,dtype=np.float64)


#         image_path = os.path.join(self.parent_dir,self.image_names[0][idx]+'.jpg')
#         img = Image.open(image_path)

#         #For obejct with ground_truth value as 0 mark it as -1 while filing in sample dict. below.
#         sample = {'image':img,'ground_truth':self.image_names[1][idx],'bndbox':bndbox_list}
                
#         if(self.transform_ips):
#             sample['image'] = self.transform_ips(sample['image'])

#         if(self.transform_y):
#             sample['bndbox'] = self.transform_y(sample['bndbox'])
        
#         return sample

# data = PascalVocDataset(annotation_path='/media/keyurparalkar/310230E03A1B6D12/Datasets/pascal-voc/VOC2012/Annotations',txt_file_name='/media/keyurparalkar/310230E03A1B6D12/Datasets/pascal-voc/VOC2012/ImageSets/Main/car_train.txt'
#             ,root_dir='/media/keyurparalkar/310230E03A1B6D12/Datasets/pascal-voc/VOC2012/JPEGImages',
#             transform_ips=transforms.Compose([
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#             ,transform_y = transforms.Lambda(lambda x:norm_range(x))
#             )


def norm_range(x):
    '''
    For scaling the bounding box according to the size change we need to 's' scaling factor which is the ratio of original dim and changed dim.
    therefore, 
    s = original dim / changed dim
    Get original dim for each input and divide it by 224. Then multiple s with all the bndbox coordinates
    original_dim = x['size']['width'] and x['size']['height']
    changed dim = 224

    HYPOTHESIS 1 => use s = changed dim / orig_dim  = BETTER results(90%)
    HYPOTHESIS 2 => use s = abs(orig_dim - changed_dim) / orignal  = GOOD results(70%)
    '''
    orig_dim_width, orig_dim_height = float(x['annotation']['size']['width']), float(x['annotation']['size']['height'])
    # s_width = np.abs(orig_dim_width - 224)/orig_dim_width
    # s_height = np.abs(orig_dim_height-224)/orig_dim_height
    s_width = 224/orig_dim_width 
    s_height = 224/orig_dim_height
    # s_width = 0.55
    # s_height = 0.55

    for elem in x['annotation']['object']:
        elem['bndbox']['xmin'] = float(elem['bndbox']['xmin'])*s_width
        elem['bndbox']['ymin'] = float(elem['bndbox']['ymin'])*s_height
        elem['bndbox']['xmax'] = float(elem['bndbox']['xmax'])*s_width
        elem['bndbox']['ymax'] = float(elem['bndbox']['ymax'])*s_height
    return x

def show_original_bndboxes(batch):
    objects_len = len(batch[1]['annotation']['object'])
    total_objects = []

    for i in range(objects_len):
        objects = {'name':[],'bndbox':[]}
        curr_objs = batch[1]['annotation']['object'][i]

        for j in range(len(curr_objs['name'])):
            objects['name'].append(curr_objs['name'][j])
            objects['bndbox'].extend([[curr_objs['bndbox']['xmin'][j].item(),curr_objs['bndbox']['ymin'][j].item(),curr_objs['bndbox']['xmax'][j].item(),curr_objs['bndbox']['ymax'][j].item()]])

        total_objects.append(objects)

    image_batches= batch[0]

    #generating random index between 0 to 3 for cycling through images
    index_seed = np.random.randint(low=0,high=4)

    fig, ax = plt.subplots(1)
    ax.set_title('Ground-truth image visualization')
    ax.imshow((image_batches[index_seed].permute(1,2,0))*torch.tensor([0.229, 0.224, 0.225])+torch.tensor([0.485, 0.456, 0.406]))

    #Create no. of rectangles which is equivalent to no. of objects in the ground-truth objects_len
    for x in range(objects_len):
        rect = patches.Rectangle((total_objects[x]['bndbox'][index_seed][0],total_objects[x]['bndbox'][index_seed][1]), 
                total_objects[x]['bndbox'][index_seed][2], total_objects[x]['bndbox'][index_seed][3], linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.text(total_objects[x]['bndbox'][index_seed][0],total_objects[x]['bndbox'][index_seed][1],total_objects[x]['name'][index_seed],backgroundcolor='r')

    plt.show()


data = torchvision.datasets.VOCDetection(root='/media/keyurparalkar/310230E03A1B6D12/Datasets/',year='2012',image_set='train',
                download=False,
                transform=transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),            
            target_transform=transforms.Lambda(lambda x: norm_range(x)))

dataset = DataLoader(data,batch_size=4,shuffle=False,num_workers=4)

if(len(sys.argv)>=2 and sys.argv[1] == "bndbox"):
    '''
    Objects print and display = Image, bounding box coordinates and ground-truth
    batch is an array. Therefore, for given indices:
    0 = Image,
    1 = Annotations

    For:
    Image index = 0
    bndbox co = batch[1]['annotation']['object'][0]['bndbox']
    ground-truth labels = batch[1]['annotation']['object'][0]['name']
    '''

    i, batch = next(iter(enumerate(dataset)))
    show_original_bndboxes(batch)

#for printing avg and std for all the images:
# mean = 0
# std = 0
# mean_sum = []
# std_sum = []

# for i, batch in enumerate(dataset):
    # mean_sum.append(batch['bndbox'].mean())
    # std_sum.append(batch['bndbox'].std())
    # print(batch['bndbox'])

# mean = np.array(mean_sum).mean()
# std = np.array(std_sum).mean()

# print("AVG  = ",mean)
# print("STD = ",std)
