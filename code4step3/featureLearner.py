import cv2
import numpy as np
import os
import copy
import SimpleITK as sitk
import random
from collections import defaultdict
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from scipy import spatial
import argparse
import pandas as pd
import xarray as xr
import pickle as pkl

ROOT_DIR = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/"
trainFileName = "trainfiles.txt"
testFileName = "testfiles.txt"
os.chdir(ROOT_DIR)
num_points = 10
margin = 0.5

register_pairs = {}

class Image:
    def __init__(self, reg_dir):
        self.image_list = []
        self.aseg_list = []
        self.reg_dir = reg_dir
        self.parse_images()
        self.parse_registration()
        self.make_xarray()
        
    def parse_images(self):
        images = self.reg_dir.split("-")
        assert(len(images)==2)
        self.fixed_image = sitk.ReadImage(ROOT_DIR + "Brain2NIFI/" + images[0] + "/norm.nii")
        self.fixed_aseg = sitk.ReadImage(ROOT_DIR + "Brain2NIFI/" + images[0] + "/aseg.nii")
        self.moving_image = sitk.ReadImage(ROOT_DIR + "Brain2NIFI/" + images[1] + "/norm.nii")
        self.moving_aseg = sitk.ReadImage(ROOT_DIR + "Brain2NIFI/" + images[1] + "/aseg.nii")
        self.image_list.append(self.fixed_image)
        self.image_list.append(self.moving_image)
        self.aseg_list.append(self.fixed_aseg)
        self.aseg_list.append(self.moving_aseg)
        
    def parse_registration(self):
        param0=sitk.ReadParameterFile("BrainParameterMaps/"+self.reg_dir+"/TransformParameters.0.txt")
        param1=sitk.ReadParameterFile("BrainParameterMaps/"+self.reg_dir+"/TransformParameters.1.txt")
        param2=sitk.ReadParameterFile("BrainParameterMaps/"+self.reg_dir+"/TransformParameters.2.txt")
        transformixImageFilter=sitk.TransformixImageFilter()
        transformixImageFilter.AddTransformParameterMap(param0)
        transformixImageFilter.AddTransformParameterMap(param1)
        transformixImageFilter.AddTransformParameterMap(param2)
        self.transformixImageFilter=transformixImageFilter
        
    def make_xarray(self):
        self.fixed_ds = xr.Dataset({'image': (['x','y','z'], sitk.GetArrayFromImage(self.fixed_image))},
                          coords={
                              'x':np.arange(256),
                              'y':np.arange(256),
                              'z':np.arange(256)
                          })
        self.moving_ds = xr.Dataset({'image': (['x','y','z'], sitk.GetArrayFromImage(self.moving_image))},
                          coords={
                              'x':np.arange(256),
                              'y':np.arange(256),
                              'z':np.arange(256)
                          })
    
    def register_points(self, test_file='test.pts'):
        if os.path.exists('outputpoints.txt'):
            os.remove('outputpoints.txt')
        self.transformixImageFilter.SetFixedPointSetFileName(test_file)
        self.transformixImageFilter.SetMovingImage(self.moving_image)
        self.transformixImageFilter.Execute()

def get_data(path):
	heart = sitk.ReadImage(path)
	heartArray = sitk.GetArrayFromImage(heart)
	return heartArray

def conv_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        act_fn,
    )
    return model

def get_output_points(filename='outputpoints.txt'):
    fr = open(filename, 'r')
    res = None
    for line in fr.readlines():
        # Todo: make sure whether we should use OutputIndexMoving or OutputIndexFixed
        # modify the following line, seems to fix the bug

        line = line[line.index('OutputIndexMoving = ') + len('OutputIndexMoving = '):]
        line = line[:line.index('\n')].lstrip('[').rstrip(']')
        array = np.fromstring(line, dtype=int, sep=' ')
        if res is None:
            res = array.reshape(1, 3)
        else:
            res = np.concatenate((res, array.reshape(1, 3)), 0)
    return res


def find_point(point, image):
    # 1) write the point to file
    if os.path.exists('test.pts'):
        os.remove('test.pts')
    fr = open('test.pts', 'w')
    fr.write('index'+'\n'+str(1)+'\n'+str(fixed_point[0])+' '+str(fixed_point[1])+' '+str(fixed_point[2]))
    fr.close()

    # find the corresponding point
    image.register_points()
    transformed_points = get_output_points()
    return transformed_points

class BrainImageDataset(Dataset):
    def __init__(self, dirList):
        self.data = dirList

    def __getitem__(self, index):

        fix = self.data[index]
        fixed_image = get_data(ROOT_DIR + "Brain2NIFI/" + fix + "/norm.nii")
        moving = register_pairs[fix]
        moving_image = get_data(ROOT_DIR + "Brain2NIFI/" + moving + "/norm.nii")
        return (fixed_image_array, moving_image_array, fix, moving)

    def __len__(self):
        return len(self.data)

class featureLearner(nn.Module):
    def __init__(self):
        super(featureLearner, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.ReLU()

        print("\n------Initiating Network------\n")

        self.cnn1 = conv_block_3d(self.in_dim, self.num_filter, act_fn)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if (isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d)):
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
    	out = self.cnn1(x)
        return out
    def save(self,epoch):
        torch.save(self.state_dict(),"featureLearner"+'_'+str(epoch)+'.pt')

def load_Directory(is_train):
    #read file with train data path
    if is_train:
        file = open(trainFileName, "r+")
    #read file with test data path
    else:
        file = open(testFileName, "r+")

    dirnames = file.readlines()
    data_directory = [x.strip() for x in dirnames]

    dirList = []
    for i in data_directory:
        if i in register_pairs:
            dirList.append(i)
    return dirList

def load_pairs():
    for root, directories in os.walk(ROOT_DIR +"BrainParameterMaps"):
        for pairname in directories:
            images = self.reg_dir.split("-")
            assert(len(images)==2)
            fixed_image = images[0]
            moving_image = images[1]
            register_pairs[fixed_image] = moving_image

def generate_negative(fixed_point):
    a = random.randint(0,256)
    b = random.randint(0,256)
    c = random.randint(0,256)
    negative_point = np.array([a,b,c]).astype('int')
    if (negative_point == fixed_point).all():
        negative_point = generate_negative(fixed_point)
    return negative_point

def find_postive_negative_points(image):

    point_list = []
    positive_point_list = []
    negative_point_list = []

    for i in range(num_points):
        x = random.randint(0,256)
        y = random.randint(0,256)
        z = random.randint(0,256)
        fixed_point = np.array([x,y,z]).astype('int')
        positive_point = find_point(fixed_point,image)
        #generate negative point
        negative_point = generate_negative(fixed_point)
        point_list.append(fixed_point)
        positive_point_list.append(positive_point)
        negative_point_list.append(negative_point)

    return point_list, positive_point_list, negative_point_list


class CorrespondenceContrastiveLoss(nn.Module):
    """
    Correspondence Contrastive loss
    Takes feature of pairs of points and a target label == 1 if positive pair and label == 0 otherwise
    """

    def __init__(self, margin, N):
        super(CorrespondenceContrastiveLoss, self).__init__()
        self.margin = margin
        self.N = N

    def forward(self, fix_image_feature, moving_image_feature, fixed_points, positive_points, negative_points):
        loss = 0
        for i in range(self.N):
            x, y, z = fixed_points[i]
            a, b, c = positive_points[i]
            label = 1 # positive pair
            distance = (fix_image_feature[x][y][z] - moving_image_feature[a][b][c]).pow(2).sum(1)  # squared distance
            loss += label * distance + (1-label) * (max(0, self.margin-math.sqrt(distance))) ** 2
        for i in range(self.N):
            x, y, z = fixed_points[i]
            a, b, c = negative_points[i]
            label = 0 # negative pair
            distance = (fix_image_feature[x][y][z] - moving_image_feature[a][b][c]).pow(2).sum(1)  # squared distance
            loss += label * distance + (1-label) * (max(0, self.margin-math.sqrt(distance))) ** 2
        loss /= (4*self.N)
        return loss

  

def train(args, model, device, loader, optimizer, epoch):

    model.train()
    criterion = CorrespondenceContrastiveLoss(margin, num_points)

    for batch_idx, (fixed_image_array, moving_image_array, fix, moving) in enumerate(loader):
        image = Image(fix+"-"+moving)
        point_list, positive_point_list, negative_point_list = find_postive_negative_points(image)
        
        fixed_image_array, moving_image_array = fixed_image_array.to(device), moving_image_array.to(device)
        optimizer.zero_grad()
        fixed_image_feature = model(fixed_image_array)
        moving_image_feature = model(moving_image_array)

        loss = criterion(fixed_image_feature,moving_image_feature, point_list, positive_point_list, negative_point_list)
        loss.backward()
        optimizer.step()
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx,
                100. * batch_idx, loss.item()))

      return loss


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--test-model', type=str, default='', metavar='N',
                    help='If test-model has a name, load pre-trained model')
parser.add_argument('--predict-model', type=str, default='', metavar='N',
                    help='If predict-model has a name, do not do training, just give result on dev and test set')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.00001)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='LR',
                    help='weight decay')

input_args = parser.parse_args()


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using device: "+str(device))

#store all pairs of registration
load_pairs()

model = featureLearner().to(device)
optimizer = optim.Adam(model.parameters(), lr=input_args.lr, betas=(0.9, 0.99), weight_decay=input_args.wd)

train_dataset = BrainImageDataset(load_Directory(True))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=50, shuffle=True)

for epoch in range(1, args['epochs']+1):
    train(args, model, device, train_loader, optimizer, epoch)
    model.save(epoch)

