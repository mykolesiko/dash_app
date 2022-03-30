import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
import torch.nn.functional as F
from torchvision import transforms

from tqdm.notebook import tqdm
from torch import optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import cv2

import torch.optim as optim
import torchvision.models as models
from tqdm.notebook import tqdm
from torch.nn import functional as fnn
from src.constants import NUM_CLASSES, THRESHOLD_MASK, NAMES

import pandas as pd
import PIL
import os
from PIL import Image, ImageDraw



class ModelPredict:
    def __init__(self, model_path:str, data_path:str, device, transforms):
        self.model_path = model_path
        self.data_path = data_path
        self.transforms = transforms
        self.COCO_CLASS_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.load_segmentation_model()
        self.device = device
        self.model = None
        self.dirs = ['n02099601', 'n02093754', 'n02089973', 'n02096294', 'n02088364', 'n02087394', 'n02086240', 'n02115641', 'n02105641', 'n02111889']
    def get_name(self, image_name):
        return  NAMES[self.dirs.index(image_name[0:9])]
    def load_model(self, model_file: str):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES, bias=True)

        with open(os.path.join(self.model_path, model_file), "rb") as fp:
            print(fp)
            best_state_dict = torch.load(fp, map_location="cpu")
            self.model.load_state_dict(best_state_dict)
        return self.model

    def predict(self, image_file: str):
        #model = self.load_model(model_file)
        mask_file_path = self.save_dog_mask(os.path.join(self.data_path, image_file))
        image = cv2.imread(mask_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = {}
        sample['label'] = 0
        sample['image'] = image
        image = self.transforms(sample)
        img_tensor = sample['image'].unsqueeze(0)
        #print(img_tensor.shape)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(img_tensor.to(self.device))
        labels_pred = predictions.argmax(1)
        prediction = labels_pred[0]
        print(NAMES[prediction])
        return prediction



    def load_segmentation_model(self):
        mask_model = models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,
            pretrained_backbone=True,
            progress=True,
            num_classes=91,
        )
        self.mask_model = mask_model
        return mask_model

    def save_dog_mask(self, file  :  str):
            img1 = Image.open(file)
            if img1.mode != 'RGB':
                print("***")
            # print(img.mode)
            to_tensor = transforms.ToTensor()
            thresh = 0.0

            img_tensor = to_tensor(img1).unsqueeze(0)# делаем тензор с 3 каналами из изображением
            self.mask_model.to(self.device)
            # eval режим
            self.mask_model.eval()
            with torch.no_grad():
                predictions = self.mask_model(img_tensor.to(self.device))  # list of size 1
            prediction = predictions[0]

            #if n_colors is None:
            #    n_colors = model.roi_heads.box_predictor.cls_score.out_features

            #palette = sns.color_palette(None, n_colors)

            # visualize
            img = cv2.imread(file, cv2.COLOR_BGR2RGB)
            # show_image(img)
            score_max = 0
            found = False
            for i in range(len(prediction["boxes"])):

                x1, x2, x3, x4 = map(int, prediction["boxes"][i].tolist())
                label = int(prediction["labels"][i].cpu())
                if label != 18:
                    continue
                # print(label)
                score = float(prediction["scores"][i].cpu())
                #name = self.COCO_CLASS_NAMES[label]
                #color = palette[label]
                # if verbose:
                # if score > thresh:
                # print ("Class: {}, Confidence: {}".format(name, score))
                if score > thresh:
                    if score > score_max:
                        score_max = score
                        x1_max, x2_max, x3_max, x4_max = x1, x2, x3, x4
                        mask = prediction['masks'][i][0, :, :].cpu().numpy()
                        found = True
                        # print(x1_max, x2_max, x3_max, x4_max)

            if (found == True) and (img1.mode == 'RGB'):
                if (mask.shape[0] != img.shape[0]) or (mask.shape[1] != img.shape[1]):
                    print("1111")
                    image = img
                    isWritten = cv2.imwrite(file + "_mask", image)
                    return
                # print("11")
                # image = cv2.rectangle(img.copy(), (x1_max, x2_max), (x3_max, x4_max),
                #              np.array(color) * 255, 2)
                # show_image(image)
                # print(mask.shape, img.shape)
                img[mask < THRESHOLD_MASK] = (0, 0, 0)
                image = img[x2_max: x4_max, x1_max: x3_max, ]

                # show_image(image)
                isWritten = cv2.imwrite(file + "_mask", image)
            else:
                image = img
                isWritten = cv2.imwrite(file + "_mask", image)
                print(isWritten)
            return (file + "_mask")






