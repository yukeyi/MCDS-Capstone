import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        label = self.target_list[index]
        return img, label

#given root directory parse data
def parse_data(datadir, max_count):
    img_list = []
    label_list = []

    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.png'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                if "NYC" in filei:
                    label_list.append(0)
                else:

                    label_list.append(1)
    print('{}\t\t{}\n'.format('#Images', len(img_list)))
    return img_list, label_list


def get_loader(mode="train"):
    loader = None
    if mode == "train":
        data_path = "/pylon5/ac5616p/baij/DeepMiner/SmallTrain/"
        shuffle = True
        img_list, label_list = parse_data(data_path, 5000)
        dataset = ImageDataset(img_list, label_list)
        loader = DataLoader(dataset, shuffle=shuffle, batch_size=32, drop_last=False)
    if mode == "val":
        data_path = "/pylon5/ac5616p/baij/DeepMiner/SmallVal/"
        shuffle = False
        img_list, label_list = parse_data(data_path, 1000)
        dataset = ImageDataset(img_list, label_list)
        loader = DataLoader(dataset, shuffle=shuffle, batch_size=32, drop_last=False)

    return loader
