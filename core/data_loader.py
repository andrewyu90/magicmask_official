"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random
import pdb
from munch import Munch
from PIL import Image
import numpy as np
import glob
import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import copy
import matplotlib.pyplot as plt
import cv2
#from magic_mask_utils import *


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class Div(object):
    def __init__(self, denominator):
        assert isinstance(denominator, float)
        self.denominator = denominator
    def __call__(self, sample):

        return torch.div(torch.FloatTensor(sample),self.denominator)


class Subtract(object):
    def __init__(self, num):
        assert isinstance(num, int)
        self.sub = num
    def __call__(self, sample):
        return torch.sub(sample,self.sub)
        
        

class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

class LPFFDataSet(data.Dataset):
    def __init__(self, data_path_list,img_size):
        self.datasets = []
        #pdb.set_trace()
        self.num_per_folder =[]
        self.lm_image_path = data_path_list[0][:data_path_list[0].rfind('/')+1] \
                             + data_path_list[0][data_path_list[0].rfind('/')+1:] + '_lm_images/'
        self.mask_image_path = data_path_list[0][:data_path_list[0].rfind('/')+1] \
                             + data_path_list[0][data_path_list[0].rfind('/')+1:] + '_mask_images/'
        self.depth_image_path = data_path_list[0][:data_path_list[0].rfind('/')+1] \
                             + data_path_list[0][data_path_list[0].rfind('/')+1:] + '_depth_images/'
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            self.datasets.append(image_list)
            self.num_per_folder.append(len(image_list))

        self.img_size = img_size

    def __getitem__(self, item):
        idx = 0
        while item >= self.num_per_folder[idx]:
            item -= self.num_per_folder[idx]
            idx += 1
        image_path = self.datasets[idx][item]
        print(image_path)
        souce_lm_image_path = self.lm_image_path  + image_path.split('/')[-1]
        souce_depth_image_path = self.depth_image_path  + image_path.split('/')[-1]
        souce_mask_image_path = self.mask_image_path  + image_path.split('/')[-1]

        print(souce_mask_image_path)
        
        #pdb.set_trace()
        source_org_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        source_id_image = source_org_image
        
        source_org_image = cv2.resize(source_org_image,(self.img_size,self.img_size))
        source_image = torch.Tensor(source_org_image/255.0).permute(2, 0, 1)
        
        source_id_image = cv2.resize(source_id_image,(112,112))
        source_id_image = torch.Tensor(source_id_image/127.5-1.0).permute(2, 0, 1)
        
        source_lm_image = cv2.cvtColor(cv2.imread(souce_lm_image_path), cv2.COLOR_BGR2RGB)
        source_depth_image = cv2.cvtColor(cv2.imread(souce_depth_image_path), cv2.COLOR_BGR2RGB)
        source_mask_image =  cv2.cvtColor(cv2.imread(souce_mask_image_path), cv2.COLOR_BGR2RGB)
        
        source_lm_image = cv2.resize(source_lm_image,(self.img_size,self.img_size))
        source_lm_image = torch.Tensor(source_lm_image/255.0).permute(2, 0, 1)
        
        source_depth_image = cv2.resize(source_depth_image,(self.img_size,self.img_size))
        source_depth_image = torch.Tensor(source_depth_image/255.0).permute(2, 0, 1)
        source_mask_image = cv2.resize(source_mask_image,(self.img_size,self.img_size))
        source_mask_image = torch.Tensor(source_mask_image[:,:,0]/255).unsqueeze(0)
        
        #print('After transform')
        #print(source_image)
        #choose ref from the same folder image
        
        temp = copy.deepcopy(self.datasets[idx]) 
        temp.pop(item)
        
        
        reference_image_path = temp[random.randint(0, len(temp)-1)]
        #print(reference_image_path)
        reference_lm_image_path = self.lm_image_path + reference_image_path.split('/')[-1]
        reference_depth_image_path = self.depth_image_path  + reference_image_path.split('/')[-1]
        reference_mask_image_path = self.mask_image_path  + reference_image_path.split('/')[-1]
        
        
        reference_org_image = cv2.cvtColor(cv2.imread(reference_image_path), cv2.COLOR_BGR2RGB)
        reference_id_image = reference_org_image
        #pdb.set_trace()
        #print('just loaded')
        #print(reference_org_image)
        
        reference_org_image = cv2.resize(reference_org_image,(self.img_size,self.img_size))
        reference_image = torch.Tensor(reference_org_image/255.0).permute(2, 0, 1)
        
        reference_id_image = cv2.resize(reference_id_image,(112,112))
        reference_id_image = torch.Tensor(reference_id_image/127.5-1.0).permute(2, 0, 1)
        
        reference_lm_image = cv2.cvtColor(cv2.imread(reference_lm_image_path), cv2.COLOR_BGR2RGB)
        reference_depth_image = cv2.cvtColor(cv2.imread(reference_depth_image_path), cv2.COLOR_BGR2RGB)
        reference_mask_image = cv2.cvtColor(cv2.imread(reference_mask_image_path), cv2.COLOR_BGR2RGB)
        
        reference_lm_image = cv2.resize(reference_lm_image,(self.img_size,self.img_size))
        reference_lm_image = torch.Tensor(reference_lm_image/255.0).permute(2, 0, 1)
        
        reference_depth_image = cv2.resize(reference_depth_image,(self.img_size,self.img_size))
        reference_depth_image = torch.Tensor(reference_depth_image/255.0).permute(2, 0, 1)
        
        reference_mask_image = cv2.resize(reference_mask_image,(self.img_size,self.img_size))
        reference_mask_image = torch.Tensor(reference_mask_image[:,:,0]/255).unsqueeze(0)
        
        
        outputs=dict(src=source_image, 
                     ref=reference_image,
                     src_id=source_id_image,
                     ref_id=reference_id_image,
                     src_lm=source_lm_image,
                     ref_lm=reference_lm_image,
                     src_depth=source_depth_image,
                     ref_depth=reference_depth_image,
                     src_mask=(1-source_mask_image), 
                     ref_mask=(1-reference_mask_image),
                     src_img_file = image_path,
                     ref_img_file = reference_image_path
        )
        return outputs
    def __len__(self):
        return sum(self.num_per_folder)


class TrainFaceDataSet(data.Dataset):
    def __init__(self, data_path_list, transform_img=None, transform_id=None, transform_seg=None):
        self.datasets = []
        #pdb.set_trace()
        self.num_per_folder =[]
        self.lm_image_path = data_path_list[0][:data_path_list[0].rfind('/')+1] \
                             + data_path_list[0][data_path_list[0].rfind('/')+1:] + '_lm_images/'
        self.mask_image_path = data_path_list[0][:data_path_list[0].rfind('/')+1] \
                             + data_path_list[0][data_path_list[0].rfind('/')+1:] + '_mask_images/'
        self.depth_image_path = data_path_list[0][:data_path_list[0].rfind('/')+1] \
                             + data_path_list[0][data_path_list[0].rfind('/')+1:] + '_depth_images/'
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            self.datasets.append(image_list)
            self.num_per_folder.append(len(image_list))
        self.transform = transform_img
        self.transform_id = transform_id
        self.transform_seg = transform_seg

    def __getitem__(self, item):
        idx = 0
        while item >= self.num_per_folder[idx]:
            item -= self.num_per_folder[idx]
            idx += 1
        image_path = self.datasets[idx][item]
        souce_lm_image_path = self.lm_image_path  + image_path.split('/')[-1]
        souce_depth_image_path = self.depth_image_path  + image_path.split('/')[-1]
        souce_mask_image_path = self.mask_image_path  + image_path.split('/')[-1]

        #pdb.set_trace()
        source_org_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        source_id_image = source_org_image

        source_org_image = cv2.resize(source_org_image,(128,128))
        source_image = torch.Tensor(source_org_image/255.0).permute(2, 0, 1)

        source_id_image = cv2.resize(source_id_image,(112,112))
        source_id_image = torch.Tensor(source_id_image/127.5-1.0).permute(2, 0, 1)


        
        source_lm_image = cv2.cvtColor(cv2.imread(souce_lm_image_path), cv2.COLOR_BGR2RGB)
        source_depth_image = cv2.cvtColor(cv2.imread(souce_depth_image_path), cv2.COLOR_BGR2RGB)
        source_mask_image =  cv2.cvtColor(cv2.imread(souce_mask_image_path), cv2.COLOR_BGR2RGB)

        source_lm_image = cv2.resize(source_lm_image,(128,128))
        source_lm_image = torch.Tensor(source_lm_image/255.0).permute(2, 0, 1)

        source_depth_image = cv2.resize(source_depth_image,(128,128))
        source_depth_image = torch.Tensor(source_depth_image/255.0).permute(2, 0, 1)
        source_mask_image = cv2.resize(source_mask_image,(128,128))
        source_mask_image = torch.Tensor(source_mask_image[:,:,0]/255).unsqueeze(0)

        #print('After transform')
        #print(source_image)
        #choose ref from the same folder image
        
        temp = copy.deepcopy(self.datasets[idx]) 
        temp.pop(item)
        
        reference_image_path = temp[random.randint(0, len(temp)-1)]
        reference_lm_image_path = self.lm_image_path + reference_image_path.split('/')[-1]
        reference_depth_image_path = self.depth_image_path  + reference_image_path.split('/')[-1]
        reference_mask_image_path = self.mask_image_path  + reference_image_path.split('/')[-1]

        
        reference_org_image = cv2.cvtColor(cv2.imread(reference_image_path), cv2.COLOR_BGR2RGB)
        reference_id_image = reference_org_image
        #print('just loaded')
        #print(reference_org_image)

        reference_org_image = cv2.resize(reference_org_image,(128,128))
        reference_image = torch.Tensor(reference_org_image/255.0).permute(2, 0, 1)

        reference_id_image = cv2.resize(reference_id_image,(112,112))
        reference_id_image = torch.Tensor(reference_id_image/127.5-1.0).permute(2, 0, 1)


        
        reference_lm_image = cv2.cvtColor(cv2.imread(reference_lm_image_path), cv2.COLOR_BGR2RGB)
        reference_depth_image = cv2.cvtColor(cv2.imread(reference_depth_image_path), cv2.COLOR_BGR2RGB)
        reference_mask_image = cv2.cvtColor(cv2.imread(reference_mask_image_path), cv2.COLOR_BGR2RGB)

        reference_lm_image = cv2.resize(reference_lm_image,(128,128))
        reference_lm_image = torch.Tensor(reference_lm_image/255.0).permute(2, 0, 1)

        reference_depth_image = cv2.resize(reference_depth_image,(128,128))
        reference_depth_image = torch.Tensor(reference_depth_image/255.0).permute(2, 0, 1)

        reference_mask_image = cv2.resize(reference_mask_image,(128,128))
        reference_mask_image = torch.Tensor(reference_mask_image[:,:,0]/255).unsqueeze(0)
    
        #pdb.set_trace()
        outputs=dict(src=source_image, 
                     ref=reference_image,
                     src_id=source_id_image,
                     ref_id=reference_id_image,
                     src_lm=source_lm_image,
                     ref_lm=reference_lm_image,
                     src_depth=source_depth_image,
                     ref_depth=reference_depth_image,
                     src_mask=(1-source_mask_image), 
                     ref_mask=(1-reference_mask_image)
        )
        return outputs
    def __len__(self):
        return sum(self.num_per_folder)


class TestFaceDataSet_MPIE(data.Dataset):
    def __init__(self, data_path_list, test_img_list, transform=None, transform_seg=None):

        
        self.source_dataset = []
        self.reference_dataset = []
        self.data_path_list = data_path_list
        #pdb.set_trace()
        '''
        self.lm_image_path = data_path_list+ '_lm_images/' 
        self.mask_image_path = data_path_list + '_mask_images/'
        self.depth_image_path = data_path_list + '_depth_images/' 
        
        '''
        
        self.lm_image_path = data_path_list[0:-4]+ '_lm_images/' + data_path_list[-3:]+'/'
        self.mask_image_path = data_path_list[0:-4] + '_mask_images/' + data_path_list[-3:]+'/'
        self.depth_image_path = data_path_list[0:-4] + '_depth_images/' + data_path_list[-3:]+'/'
       
        
        print(self.lm_image_path)
        #pdb.set_trace()
        f=open(test_img_list,'r')
        for t, line in enumerate(f.readlines()):
            line.split(' ')
            self.source_dataset.append(line.split(' ')[0])
            self.reference_dataset.append(line.split(' ')[1].split('\n')[0])
            #pdb.set_trace()
            if t==0:
                #pdb.set_trace()
                self.src_data_path_list = line.split(' ')[0][0:-14]
                self.src_lm_image_path = line.split(' ')[0][0:-14] + '_v2_lm_images/'
                self.src_mask_image_path = line.split(' ')[0][0:-14] + '_v2_mask_images/'
                self.src_depth_image_path = line.split(' ')[0][0:-14] + '_v2_depth_images/'
                
                #self.src_data_path_list = line.split(' ')[0][0:-14]
                #self.src_lm_image_path = line.split(' ')[0][0:-17] + '_lm_images_'+ 'v2/'
                #self.src_mask_image_path = line.split(' ')[0][0:-17] + '_mask_images_'+ 'v2/'
                #self.src_depth_image_path = line.split(' ')[0][0:-17] + '_depth_images_'+ 'v2/'
        f.close()
        self.transform = transform
        self.transform_seg = transform_seg

    def __getitem__(self, item):
        #pdb.set_trace()
        source_image_path = self.source_dataset[item]
        #print(source_image_path)
        try:
            source_image = cv2.cvtColor(cv2.imread(source_image_path), cv2.COLOR_BGR2RGB)
            source_id_image = source_image
            source_image = cv2.resize(source_image,(128,128))
            source_image = torch.Tensor(source_image/255.0).permute(2, 0, 1)
            
        except:
            print('fail to read %s.jpg'%source_image_path)

        source_id_image = cv2.resize(source_id_image,(112,112))
        source_id_image = torch.Tensor(source_id_image/127.5-1.0).permute(2, 0, 1)

        #pdb.set_trace()
        souce_lm_image_path = self.src_lm_image_path + source_image_path.split('/')[-1]
        souce_mask_image_path = self.src_mask_image_path  + source_image_path.split('/')[-1]
        souce_depth_image_path = self.src_depth_image_path  + source_image_path.split('/')[-1]
        print("lm_path")
        print(souce_lm_image_path)
        print(souce_depth_image_path)
        print(souce_mask_image_path)
        #pdb.set_trace()
        source_lm_image = cv2.cvtColor(cv2.imread(souce_lm_image_path), cv2.COLOR_BGR2RGB)
        
        source_depth_image = cv2.cvtColor(cv2.imread(souce_depth_image_path), cv2.COLOR_BGR2RGB)
        source_mask_image =  cv2.cvtColor(cv2.imread(souce_mask_image_path), cv2.COLOR_BGR2RGB)
        
        #source_parsing_image = Image.open(source_parsing_image_path).convert('L')


        source_lm_image = cv2.resize(source_lm_image,(128,128))
        source_lm_image = torch.Tensor(source_lm_image/255.0).permute(2, 0, 1)

        source_depth_image = cv2.resize(source_depth_image,(128,128))
        source_depth_image = torch.Tensor(source_depth_image/255.0).permute(2, 0, 1)
        source_mask_image = cv2.resize(source_mask_image,(128,128))
        source_mask_image = torch.Tensor(source_mask_image[:,:,0]/255).unsqueeze(0)



        #pdb.set_trace()
        reference_image_path = self.reference_dataset[item]
        
        try:
            reference_image = Image.open(reference_image_path).convert('RGB')

            reference_image = cv2.cvtColor(cv2.imread(reference_image_path), cv2.COLOR_BGR2RGB)
            reference_id_image = reference_image
            reference_image = cv2.resize(reference_image,(128,128))
            reference_image = torch.Tensor(reference_image/255.0).permute(2, 0, 1)
            
        
        except:
            print('fail to read %s.jpg' %reference_image_path)

        reference_id_image = cv2.resize(reference_id_image,(112,112))
        reference_id_image = torch.Tensor(reference_id_image/127.5-1.0).permute(2, 0, 1)

        
        reference_lm_image_path = self.lm_image_path  + reference_image_path.split('/')[-1]
        reference_mask_image_path = self.mask_image_path + reference_image_path.split('/')[-1]
        reference_depth_image_path = self.depth_image_path  + reference_image_path.split('/')[-1]


        print(reference_mask_image_path)
        reference_lm_image = cv2.cvtColor(cv2.imread(reference_lm_image_path), cv2.COLOR_BGR2RGB)
        reference_depth_image = cv2.cvtColor(cv2.imread(reference_depth_image_path), cv2.COLOR_BGR2RGB)
        reference_mask_image =  cv2.cvtColor(cv2.imread(reference_mask_image_path), cv2.COLOR_BGR2RGB)


        reference_lm_image = cv2.resize(reference_lm_image,(128,128))
        reference_lm_image = torch.Tensor(reference_lm_image/255.0).permute(2, 0, 1)

        reference_depth_image = cv2.resize(reference_depth_image,(128,128))
        reference_depth_image = torch.Tensor(reference_depth_image/255.0).permute(2, 0, 1)
        
        reference_mask_image = cv2.resize(reference_mask_image,(128,128))
        reference_mask_image = torch.Tensor(reference_mask_image[:,:,0]/255).unsqueeze(0)

        

        outputs=dict(
            src=source_image, 
            ref=reference_image, 
            src_id=source_id_image, 
            ref_id=reference_id_image, 
            src_lm=source_lm_image, 
            ref_lm=reference_lm_image, 
            src_depth=source_depth_image, 
            ref_depth=reference_depth_image, 
            src_mask=1-source_mask_image, 
            ref_mask=1-reference_mask_image, 
            src_name=self.source_dataset[item], 
            ref_name=self.reference_dataset[item], 
            src_label=self.source_dataset[item], 
            ref_label=self.reference_dataset[item]
        )
        return outputs
    def __len__(self):
        return len(self.source_dataset)


class TestFaceDataSet_MPIE_tmp(data.Dataset):
    def __init__(self, data_path_list, test_img_list, transform=None, transform_seg=None):

        
        self.source_dataset = []
        self.reference_dataset = []
        self.data_path_list = data_path_list
        self.lm_image_path = data_path_list[:data_path_list.rfind('/')+1] \
                             + data_path_list[data_path_list.rfind('/')+1:] + '_lm_images/'
        self.mask_image_path = data_path_list[:data_path_list.rfind('/')+1] \
                             + data_path_list[data_path_list.rfind('/')+1:] + '_mask_images/'
        self.depth_image_path = data_path_list[:data_path_list.rfind('/')+1] \
                             + data_path_list[data_path_list.rfind('/')+1:] + '_depth_images/'
        self.biseg_parsing_path = data_path_list[:data_path_list.rfind('/')+1] \
                             + data_path_list[data_path_list.rfind('/')+1:] + '_parsing_images/'
        f=open(test_img_list,'r')
        for line in f.readlines():
            line.split(' ')
            self.source_dataset.append(line.split(' ')[0])
            self.reference_dataset.append(line.split(' ')[1])
        f.close()
        self.transform = transform
        self.transform_seg = transform_seg

    def __getitem__(self, item):
        source_image_path = self.source_dataset[item]
        try:
            source_image = Image.open(source_image_path).convert('RGB')
        except:
            print('fail to read %s'%source_image_path)
        #pdb.set_trace()
        souce_lm_image_path = self.lm_image_path+source_image_path.split('/')[-1]
        souce_mask_image_path = self.mask_image_path+source_image_path.split('/')[-1]
        souce_depth_image_path = self.depth_image_path+source_image_path.split('/')[-1]
        #source_parsing_image_path = self.biseg_parsing_path + self.source_dataset[item]
        source_lm_image = Image.open(souce_lm_image_path).convert('RGB')
        source_mask_image = Image.open(souce_mask_image_path).convert('RGB')
        source_depth_image = Image.open(souce_depth_image_path).convert('RGB')
        #source_parsing_image = Image.open(source_parsing_image_path).convert('L')
        if self.transform is not None:
            source_image = self.transform(source_image)
            source_lm_image = self.transform(source_lm_image)
            source_mask_image = self.transform_seg(source_mask_image)
            source_depth_image = self.transform_seg(source_depth_image)
            #source_parsing = self.transform_seg(source_parsing_image)
        reference_image_path = self.reference_dataset[item]
        print()
        try:
            reference_image = Image.open(reference_image_path).convert('RGB')
        except:
            print('fail to read %s' %reference_image_path)
        reference_lm_image_path = self.lm_image_path  + self.reference_dataset[item].split('/')[-1]
        reference_mask_image_path = self.mask_image_path + self.reference_dataset[item].split('/')[-1]
        reference_depth_image_path = self.depth_image_path + self.reference_dataset[item].split('/')[-1]
        #reference_parsing_image_path = self.biseg_parsing_path + self.reference_dataset[item][0:-1]
        #pdb.set_trace()
        #print(reference_lm_image_path)
        #print(reference_mask_image_path)
        reference_lm_image = Image.open(reference_lm_image_path).convert('RGB')
        reference_mask_image = Image.open(reference_mask_image_path).convert('RGB')
        reference_depth_image = Image.open(reference_depth_image_path).convert('RGB')
        #reference_parsing = Image.open(reference_parsing_image_path).convert('L')
        if self.transform is not None:
            reference_image = self.transform(reference_image)
            reference_lm_image = self.transform(reference_lm_image)
            reference_mask_image = self.transform_seg(reference_mask_image)
            reference_depth_image = self.transform_seg(reference_depth_image)
            #reference_parsing = self.transform_seg(reference_parsing)
        outputs=dict(src=source_image, ref=reference_image, src_lm=source_lm_image, ref_lm=reference_lm_image, src_mask=1-source_mask_image,
                      ref_mask=1-reference_mask_image, src_depth = source_depth_image, ref_depth = reference_depth_image,
                      src_name=self.source_dataset[item], ref_name=self.reference_dataset[item])
        return outputs
    def __len__(self):
        return len(self.source_dataset)
        
# def get_train_loader_tmp(root, which='source', img_size=256,
#                      batch_size=8, prob=0.5, num_workers=4):
#     print('Preparing DataLoader to fetch %s images '
#           'during the training phase...' % which)

#     crop = transforms.RandomResizedCrop(
#         img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
#     rand_crop = transforms.Lambda(
#         lambda x: crop(x) if random.random() < prob else x)

#     transform = transforms.Compose([
#         rand_crop,
#         transforms.Resize([img_size, img_size]),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                              std=[0.5, 0.5, 0.5]),
#     ])

#     if which == 'source':
#         dataset = ImageFolder(root, transform)
#     elif which == 'reference':
#         dataset = ReferenceDataset(root, transform)
#     else:
#         raise NotImplementedError

#     sampler = _make_balanced_sampler(dataset.targets)
#     return data.DataLoader(dataset=dataset,
#                            batch_size=batch_size,
#                            sampler=sampler,
#                            num_workers=num_workers,
#                            pin_memory=True,
#                            drop_last=True)


def get_train_loader(root, img_size=256,
                     batch_size=8, num_workers=4):
    print('Preparing dataLoader to fetch images during the training phase...')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
        Div(255.0),
    ])
    transform_id = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor(),
        Div(127.5),
        Subtract(1),
    ])

    transform_seg = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
    ])
    train_dataset = TrainFaceDataSet(root, transform_img=transform,transform_id=transform_id, transform_seg=transform_seg)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
    return train_loader


def get_LPFF_loader(root, img_size=128,
                     batch_size=8, num_workers=4):
    print('Preparing dataLoader to fetch images during the training phase...')
    train_dataset = LPFFDataSet(root,img_size)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
    return train_loader




def get_test_loader(root, test_img_list, img_size=256,
                     batch_size=8, num_workers=4):
    print('Preparing dataLoader to fetch images during the testing phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    transform_seg = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
    ])
    test_dataset = TestFaceDataSet(root, test_img_list, transform, transform_seg)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, drop_last=True)
    return test_loader



def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)




'''
def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


'''
class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


class InputFetcher_LPFF:
    def __init__(self, loader, mode=''):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
    def _fetch_inputs(self):
        try:
            inputs_data = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            inputs_data= next(self.iter)
        return inputs_data
    def __next__(self):
        #pdb.set_trace()
        t_inputs = self._fetch_inputs()
        inputs = Munch(src=t_inputs['src'], 
                       tar=t_inputs['ref'],
                       src_id=t_inputs['src_id'], 
                       tar_id=t_inputs['ref_id'], 
                       src_lm=t_inputs['src_lm'],
                       tar_lm=t_inputs['ref_lm'],
                       src_depth=t_inputs['src_depth'],
                       tar_depth=t_inputs['ref_depth'])
      
        inputs = Munch({k: t.to(self.device) for k, t in inputs.items()})
        return inputs, t_inputs['src_img_file'], t_inputs['ref_img_file']


class InputFetcher_TMP:
    def __init__(self, loader, mode=''):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
    def _fetch_inputs(self):
        try:
            inputs_data = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            inputs_data= next(self.iter)
        return inputs_data
    def __next__(self):
        #pdb.set_trace()
        t_inputs = self._fetch_inputs()
        inputs = Munch(src=t_inputs['src'], tar=t_inputs['ref'],src_id=t_inputs['src_id'], tar_id=t_inputs['ref_id'], src_lm=t_inputs['src_lm'],
                       tar_lm=t_inputs['ref_lm'],src_depth=t_inputs['src_depth'],
                       tar_depth=t_inputs['ref_depth'], src_mask=t_inputs['src_mask'], tar_mask=t_inputs['ref_mask'],
                       src_name = t_inputs['src_name'],tar_name=t_inputs['ref_name'],
                           src_label=t_inputs['src_label'],tar_label=t_inputs['ref_label']
                      )
        #pdb.set_trace()
        '''
        if self.mode=='train':
            inputs = Munch({k: t.to(self.device) for k, t in inputs.items()})
        elif self.mode=='test':
            inputs = Munch({k: t.to(self.device) for k, t in inputs.items()},src_name=t_inputs['src_name'],tar_name=t_inputs['ref_name'],
                           src_label=t_inputs['src_label'],tar_label=t_inputs['ref_label']
                          
                          )
        '''
        return inputs


class InputFetcher:
    def __init__(self, loader, mode=''):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
    def _fetch_inputs(self):
        try:
            inputs_data = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            inputs_data= next(self.iter)
        return inputs_data
    def __next__(self):
        #pdb.set_trace()
        t_inputs = self._fetch_inputs()
        inputs = Munch(src=t_inputs['src'], tar=t_inputs['ref'],src_id=t_inputs['src_id'], tar_id=t_inputs['ref_id'], src_lm=t_inputs['src_lm'],
                       tar_lm=t_inputs['ref_lm'],src_depth=t_inputs['src_depth'],
                       tar_depth=t_inputs['ref_depth'], src_mask=t_inputs['src_mask'], tar_mask=t_inputs['ref_mask'])
        #pdb.set_trace()
        if self.mode=='train':
            inputs = Munch({k: t.to(self.device) for k, t in inputs.items()})
        elif self.mode=='test':
            inputs = Munch({k: t.to(self.device) for k, t in inputs.items()}, src_parsing=t_inputs['src_parsing'].to(self.device),
                           tar_parsing=t_inputs['ref_parsing'].to(self.device), src_name=t_inputs['src_name'],tar_name=t_inputs['ref_name'],
                           src_label=t_inputs['src_label'],tar_label=t_inputs['ref_label']
                          
                          )
        return inputs


class TestFaceDataSet_MPIE_backup(data.Dataset):
    def __init__(self, data_path_list, test_img_list, transform=None, transform_seg=None):
        self.source_dataset = []
        self.reference_dataset = []
        self.data_path_list = data_path_list
        #pdb.set_trace()
        self.lm_image_path = data_path_list[0:-4]+ '_lm_images/' + data_path_list[-3:]+'/'
        self.mask_image_path = data_path_list[0:-4] + '_mask_images/' + data_path_list[-3:]+'/'
        self.depth_image_path = data_path_list[0:-4] + '_depth_images/' + data_path_list[-3:]+'/'
        '''
        self.biseg_parsing_path = data_path_list[:data_path_list.rfind('/')+1] \
                             + data_path_list[data_path_list.rfind('/')+1:] + '_parsing_images/'
        '''
        f=open(test_img_list,'r')
        for t, line in enumerate(f.readlines()):
            line.split(' ')
            self.source_dataset.append(line.split(' ')[0])
            self.reference_dataset.append(line.split(' ')[1].split('\n')[0])
            #pdb.set_trace()
            if t==0:
                self.src_data_path_list = line.split(' ')[0][0:-10]
                self.src_lm_image_path = line.split(' ')[0][0:-11] + '_lm_images/'
                self.src_mask_image_path = line.split(' ')[0][0:-11] + '_mask_images/'
                self.src_depth_image_path = line.split(' ')[0][0:-11] + '_mask_images/'
        f.close()
        self.transform = transform
        self.transform_seg = transform_seg
    def __getitem__(self, item):
        source_image_path = self.source_dataset[item]
        try:
            source_image = Image.open(source_image_path).convert('RGB')
        except:
            print('fail to read %s'%source_image_path)
        #pdb.set_trace()
        souce_lm_image_path = self.src_lm_image_path+source_image_path.split('/')[-1]
        souce_mask_image_path = self.src_mask_image_path+source_image_path.split('/')[-1]
        souce_depth_image_path = self.src_depth_image_path+source_image_path.split('/')[-1]
        #source_parsing_image_path = self.biseg_parsing_path + self.source_dataset[item]
        source_lm_image = Image.open(souce_lm_image_path).convert('RGB')
        source_mask_image = Image.open(souce_mask_image_path).convert('RGB')
        source_depth_image = Image.open(souce_depth_image_path).convert('RGB')
        #source_parsing_image = Image.open(source_parsing_image_path).convert('L')
        if self.transform is not None:
            source_image = self.transform(source_image)
            source_lm_image = self.transform(source_lm_image)
            source_mask_image = self.transform_seg(source_mask_image)
            source_depth_image = self.transform_seg(source_depth_image)
            #source_parsing = self.transform_seg(source_parsing_image)
        reference_image_path = self.reference_dataset[item]
        print()
        try:
            reference_image = Image.open(reference_image_path).convert('RGB')
        except:
            print('fail to read %s' %reference_image_path)
        reference_lm_image_path = self.lm_image_path  + self.reference_dataset[item].split('/')[-1]
        reference_mask_image_path = self.mask_image_path + self.reference_dataset[item].split('/')[-1]
        reference_depth_image_path = self.depth_image_path + self.reference_dataset[item].split('/')[-1]
        #reference_parsing_image_path = self.biseg_parsing_path + self.reference_dataset[item][0:-1]
        #pdb.set_trace()
        #print(reference_lm_image_path)
        #print(reference_mask_image_path)
        reference_lm_image = Image.open(reference_lm_image_path).convert('RGB')
        reference_mask_image = Image.open(reference_mask_image_path).convert('RGB')
        reference_depth_image = Image.open(reference_depth_image_path).convert('RGB')
        #reference_parsing = Image.open(reference_parsing_image_path).convert('L')
        if self.transform is not None:
            reference_image = self.transform(reference_image)
            reference_lm_image = self.transform(reference_lm_image)
            reference_mask_image = self.transform_seg(reference_mask_image)
            reference_depth_image = self.transform_seg(reference_depth_image)
            #reference_parsing = self.transform_seg(reference_parsing)
        outputs=dict(src=source_image, ref=reference_image, src_lm=source_lm_image, ref_lm=reference_lm_image, src_mask=1-source_mask_image,
                      ref_mask=1-reference_mask_image, src_depth = source_depth_image, ref_depth = reference_depth_image,
                      src_name=self.source_dataset[item], ref_name=self.reference_dataset[item])
        return outputs
    def __len__(self):
        return len(self.source_dataset)


def get_test_loader_MPIE(root, test_img_list, img_size=256,
                     batch_size=8, num_workers=4):
    print('Preparing dataLoader to fetch images during the testing phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    transform_seg = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
    ])
    test_dataset = TestFaceDataSet_MPIE(root, test_img_list, transform, transform_seg)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, drop_last=True)
    return test_loader

class InputFetcher_MPIE:
    def __init__(self, loader, mode=''):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
    def _fetch_inputs(self):
        try:
            inputs_data = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            inputs_data= next(self.iter)
        return inputs_data
    def __next__(self):
        t_inputs = self._fetch_inputs()
        inputs = Munch(src=t_inputs['src'], tar=t_inputs['ref'], src_lm=t_inputs['src_lm'],
                       tar_lm=t_inputs['ref_lm'], src_mask=t_inputs['src_mask'], tar_mask=t_inputs['ref_mask'], src_depth=t_inputs['src_depth'], tar_depth=t_inputs['ref_depth'])
        if self.mode=='train':
            inputs = Munch({k: t.to(self.device) for k, t in inputs.items()})
        elif self.mode=='test':
            inputs = Munch({k: t.to(self.device) for k, t in inputs.items()}, src_name=t_inputs['src_name'],tar_name=t_inputs['ref_name'])
        return inputs



# class InputFetcher:
#     def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
#         self.loader = loader
#         self.loader_ref = loader_ref
#         self.latent_dim = latent_dim
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.mode = mode

#     def _fetch_inputs(self):
#         try:
#             x, y = next(self.iter)
#         except (AttributeError, StopIteration):
#             self.iter = iter(self.loader)
#             x, y = next(self.iter)
#         return x, y

#     def _fetch_refs(self):
#         try:
#             x, x2, y = next(self.iter_ref)
#         except (AttributeError, StopIteration):
#             self.iter_ref = iter(self.loader_ref)
#             x, x2, y = next(self.iter_ref)
#         return x, x2, y

#     def __next__(self):
#         x, y = self._fetch_inputs()
#         if self.mode == 'train':
#             x_ref, x_ref2, y_ref = self._fetch_refs()
#             z_trg = torch.randn(x.size(0), self.latent_dim)
#             z_trg2 = torch.randn(x.size(0), self.latent_dim)
#             inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
#                            x_ref=x_ref, x_ref2=x_ref2,
#                            z_trg=z_trg, z_trg2=z_trg2)
#         elif self.mode == 'val':
#             x_ref, y_ref = self._fetch_inputs()
#             inputs = Munch(x_src=x, y_src=y,
#                            x_ref=x_ref, y_ref=y_ref)
#         elif self.mode == 'test':
#             inputs = Munch(x=x, y=y)
#         else:
#             raise NotImplementedError

#         return Munch({k: v.to(self.device)
#                       for k, v in inputs.items()})