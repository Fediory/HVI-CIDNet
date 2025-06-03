# Add new fivek dataset follow Retinexformer(https://github.com/caiyuanhao1998/Retinexformer)

import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *

class FiveKDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(FiveKDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):

        folder = self.data_dir+'/input'
        folder2= self.data_dir+'/target'
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]


        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 4500