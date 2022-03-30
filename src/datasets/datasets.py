import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torch import optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import cv2

import torch.optim as optim
import torchvision.models as models
from tqdm.notebook import tqdm
from torch.nn import functional as fnn

import os
import pandas as pd
import PIL
from PIL import Image, ImageDraw
#from matplotlib import pyplot as plt
#import seaborn as sns
import glob

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)



class DogsDataset(Dataset):
    def __init__(self, root, transforms, split="train", noise_level = 0):
        super(DogsDataset, self).__init__()
        self.root = root
        self.image_names = []
        self.transforms = transforms
        self.labels = []

        # dirs = glob.glob(os.path.join(root, f'{split}/*'))
        # print(dirs)
        dirs = ['n02099601', 'n02093754', 'n02089973', 'n02096294', 'n02088364', 'n02087394', 'n02086240', 'n02115641',
                'n02105641', 'n02111889']

        files = glob.glob(os.path.join(root, f"{split}/*/*.JPEG"))

        print(len(files))
        for filename in files[:]:
            # print(filename)
            self.image_names.append(filename[7:])

        if split == 'train':
            # dirs = [dirs[i][13:] for i in range(10)]
            # print(dirs)
            labels = pd.read_csv(os.path.join(f"{root}/noisy_imagewoof.csv"))
            print(labels.head())
            labels_dict = {k: dirs.index(v) for k, v in zip(labels['path'].values, labels['noisy_labels_' + str(noise_level)].values)}
            print(labels_dict)
        else:
            # dirs = [dirs[i][11:] for i in range(10)]
            # print(dirs)
            labels = [self.image_names[i][4:13] for i in range(len(self.image_names))]
            print(labels)
            labels_dict = {k: dirs.index(v) for k, v in zip(self.image_names, labels)}
            print(labels_dict)

        self.labels_dict = labels_dict

        for image_name in self.image_names:
            # print(filename)

            self.labels.append(self.labels_dict[image_name])

    def __getitem__(self, idx):

        sample = {}

        # #print(self.image_names[idx])
        image = cv2.imread((os.path.join(self.root, self.image_names[idx] + "_mask")))
        # print(self.image_names[idx])
        # #print(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image
        # sample['label'] = self.labels_dict[self.image_names[idx]]
        sample['label'] = self.labels[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)

        # print(sample)
        return sample

    def __len__(self):
        return len(self.image_names)