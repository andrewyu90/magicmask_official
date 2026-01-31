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

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


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



class TrainFaceDataSet(data.Dataset):
    def __init__(self, data_path_list, transform=None, transform_seg=None):
        self.datasets = []
        self.num_per_folder =[]
        self.lm_image_path = data_path_list[0][:data_path_list[0].rfind('/')+1] \
                             + data_path_list[0][data_path_list[0].rfind('/')+1:] + '_lm_images/'
        self.mask_image_path = data_path_list[0][:data_path_list[0].rfind('/')+1] \
                             + data_path_list[0][data_path_list[0].rfind('/')+1:] + '_mask_images/'
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            self.datasets.append(image_list)
            self.num_per_folder.append(len(image_list))
        self.transform = transform
        self.transform_seg = transform_seg

    def __getitem__(self, item):
        idx = 0
        while item >= self.num_per_folder[idx]:
            item -= self.num_per_folder[idx]
            idx += 1
        image_path = self.datasets[idx][item]
        souce_lm_image_path = self.lm_image_path  + image_path.split('/')[-1]
        souce_mask_image_path = self.mask_image_path  + image_path.split('/')[-1]
        source_image = Image.open(image_path).convert('RGB')
        source_lm_image = Image.open(souce_lm_image_path).convert('RGB')
        source_mask_image = Image.open(souce_mask_image_path).convert('L')
        if self.transform is not None:
            source_image = self.transform(source_image)
            source_lm_image = self.transform(source_lm_image)
            source_mask_image = self.transform_seg(source_mask_image)
        #choose ref from the same folder image
        temp = copy.deepcopy(self.datasets[idx]) 
        temp.pop(item)
        reference_image_path = temp[random.randint(0, len(temp)-1)]
        reference_lm_image_path = self.lm_image_path + reference_image_path.split('/')[-1]
        reference_mask_image_path = self.mask_image_path  + reference_image_path.split('/')[-1]
        reference_image = Image.open(reference_image_path).convert('RGB')
        reference_lm_image = Image.open(reference_lm_image_path).convert('RGB')
        reference_mask_image = Image.open(reference_mask_image_path).convert('L')
        if self.transform is not None:
            reference_image = self.transform(reference_image)
            reference_lm_image = self.transform(reference_lm_image)
            reference_mask_image = self.transform_seg(reference_mask_image)
        outputs=dict(src=source_image, ref=reference_image, src_lm=source_lm_image, ref_lm=reference_lm_image,
                      src_mask=1-source_mask_image, ref_mask=1-reference_mask_image)
        return outputs
    def __len__(self):
        return sum(self.num_per_folder)

class TestFaceDataSet(data.Dataset):
    def __init__(self, data_path_list, test_img_list, transform=None, transform_seg=None):
        self.source_dataset = []
        self.reference_dataset = []
        self.data_path_list = data_path_list
        self.lm_image_path = data_path_list[:data_path_list.rfind('/')+1] \
                             + data_path_list[data_path_list.rfind('/')+1:] + '_lm_images/'
        self.mask_image_path = data_path_list[:data_path_list.rfind('/')+1] \
                             + data_path_list[data_path_list.rfind('/')+1:] + '_mask_images/'
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
        source_image_path = self.data_path_list  + '/' + self.source_dataset[item]
        try:
            source_image = Image.open(source_image_path).convert('RGB')
        except:
            print('fail to read %s.jpg'%source_image_path)
        souce_lm_image_path = self.lm_image_path + self.source_dataset[item]
        souce_mask_image_path = self.mask_image_path  + self.source_dataset[item]
        source_parsing_image_path = self.biseg_parsing_path + self.source_dataset[item]
        source_lm_image = Image.open(souce_lm_image_path).convert('RGB')
        source_mask_image = Image.open(souce_mask_image_path).convert('L')
        source_parsing_image = Image.open(source_parsing_image_path).convert('L')
        if self.transform is not None:
            source_image = self.transform(source_image)
            source_lm_image = self.transform(source_lm_image)
            source_mask_image = self.transform_seg(source_mask_image)
            source_parsing = self.transform_seg(source_parsing_image)
        reference_image_path = self.data_path_list + '/' + self.reference_dataset[item][0:-1]
        try:
            reference_image = Image.open(reference_image_path).convert('RGB')
        except:
            print('fail to read %s.jpg' %reference_image_path)
        reference_lm_image_path = self.lm_image_path  + self.reference_dataset[item][0:-1]
        reference_mask_image_path = self.mask_image_path + self.reference_dataset[item][0:-1]
        reference_parsing_image_path = self.biseg_parsing_path + self.reference_dataset[item][0:-1]
        reference_lm_image = Image.open(reference_lm_image_path).convert('RGB')
        reference_mask_image = Image.open(reference_mask_image_path).convert('L')
        reference_parsing = Image.open(reference_parsing_image_path).convert('L')
        if self.transform is not None:
            reference_image = self.transform(reference_image)
            reference_lm_image = self.transform(reference_lm_image)
            reference_mask_image = self.transform_seg(reference_mask_image)
            reference_parsing = self.transform_seg(reference_parsing)
        outputs=dict(src=source_image, ref=reference_image, src_lm=source_lm_image, ref_lm=reference_lm_image, src_mask=1-source_mask_image,
                      ref_mask=1-reference_mask_image, src_parsing=source_parsing, ref_parsing=reference_parsing,
                      src_name=self.source_dataset[item], ref_name=self.reference_dataset[item])
        return outputs
    def __len__(self):
        return len(self.source_dataset)

def get_train_loader_tmp(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    if which == 'source':
        dataset = ImageFolder(root, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_train_loader(root, img_size=256,
                     batch_size=8, num_workers=4):
    print('Preparing dataLoader to fetch images during the training phase...')
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
    train_dataset = TrainFaceDataSet(root, transform, transform_seg)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
    return train_loader

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


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)



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




class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})