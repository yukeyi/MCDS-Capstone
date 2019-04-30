import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        resize = 227
        img = Image.open(self.file_list[index])
        #divide the image to 8 patches
        img = np.array(img)
        img= img[100:, 100:, :]
        height, length, x = img.shape
        #print("hi")
        patch_size = int (height * 2 / 3)
        patch_starting_pts = []
        #print("hi")
        shift_len = int(length - patch_size)
        for i in range(2):
            for j in range(2):
                patch_starting_pts.append([i*shift_len,j*(height-patch_size)])
        img_list = [ img[i:i+patch_size,j:j+patch_size, :] for i, j in patch_starting_pts]
        img_list = [ cv2.resize(img, (resize,resize), interpolation=cv2.INTER_CUBIC) for img in img_list]
        img = np.vstack(img_list)
        #img_list = [ torch.from_numpy(img) for img in img_list]
        img = torchvision.transforms.ToTensor()(img)
        #img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        label = self.target_list[index]
        return img, label, patch_starting_pts, patch_size

#given root directory parse data
def parse_data(datadir, max_count):
    img_list = []
    label_list = []
    NYC_count = 0
    PIT_count = 0
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                if "NYC" in filei:
                    if NYC_count < max_count:
                        img_list.append(filei)
                        label_list.append(0)
                        NYC_count += 1
                else:
                    if PIT_count < max_count:
                        img_list.append(filei)
                        label_list.append(1)
                        PIT_count += 1
    print('{}\t\t{}\n'.format('#Images', len(img_list)))
    return img_list, label_list


def parse_data_full(datadir):
    img_list = []
    label_list = []

    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
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
        data_path = "/pylon5/ac5616p/baij/DeepMiner/train_fullSize/"
        shuffle = True
        img_list, label_list = parse_data_full(data_path)
        dataset = ImageDataset(img_list, label_list)
        loader = DataLoader(dataset, shuffle=shuffle, batch_size=256, drop_last=False)
    if mode == "val":
        data_path = "/pylon5/ac5616p/baij/DeepMiner/val_fullSize/"
        shuffle = False
        img_list, label_list = parse_data_full(data_path)
        dataset = ImageDataset(img_list, label_list)
        loader = DataLoader(dataset, shuffle=shuffle, batch_size=256, drop_last=False)

    return loader
