
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from data.util import *
from torchvision import transforms as t
import torch.nn.functional as F

class LOLBlurDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLBlurDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        while True:
            seed = random.randint(1, 1000000)
            random.seed(seed) 
            index = random.randint(0, 259)
            fill_index = str(index+1).zfill(4)
            folder = join(self.data_dir+'/low_blur', fill_index)
            folder2 = join(self.data_dir+'/high_sharp_scaled', fill_index)
            if  not os.path.exists(folder):
                continue
            data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
            data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
            num = len(data_filenames)
            if num != 0: break
        index1 = random.randint(1,num)

        im1 = load_img(data_filenames[index1-1])
        im2 = load_img(data_filenames2[index1-1])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        return im1, im2, data_filenames[index1-1], data_filenames2[index1-1]

    def __len__(self):
        return 10200
    

class SIDDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(SIDDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        while True:
            seed = random.randint(1, 1000000)
            random.seed(seed) 
            index = random.randint(0, 233)
            fill_index = str(index+1).zfill(5)
            folder = join(self.data_dir+'/short', fill_index)
            folder2 = join(self.data_dir+'/long', fill_index)
            if os.path.exists(folder): 
                data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
                data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
                num = len(data_filenames)
                break
            else:
                continue
        index1 = random.randint(1,num)


        im1 = load_img(data_filenames[index1-1])
        im2 = load_img(data_filenames2[0])
        _, file1 = os.path.split(data_filenames[index1-1])
        _, file2 = os.path.split(data_filenames2[0])
        seed = np.random.randint(random.randint(1, 1000000)) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 2099
    
    
    
class SICEDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(SICEDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        while True:
            seed = random.randint(1, 1000000)
            random.seed(seed) 
            index = random.randint(0, 590)
            fill_index = str(index+1)
            train, tail = os.path.split(self.data_dir)
            folder = join(self.data_dir, fill_index)
            data_gt = join(train+'/label', fill_index+'.JPG')
            if os.path.exists(folder): 
                data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
                num = len(data_filenames)
                break
            else:
                continue
        index1 = random.randint(1,num)

        im1 = load_img(data_filenames[index1-1])
        im2 = load_img(data_gt)
        _, file1 = os.path.split(data_filenames[index1-1])
        _, file2 = os.path.split(data_gt)
        seed = np.random.randint(random.randint(1, 1000000)) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 4803