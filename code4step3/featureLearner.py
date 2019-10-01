import cv2
import numpy as np
import os
import copy
import SimpleITK as sitk
import random
from collections import defaultdict
import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from scipy import spatial
import argparse
import pandas as pd
import pickle as pkl

torch.backends.cudnn.enabled = False

ROOT_DIR = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/"
trainFileName = "trainfiles.txt"
testFileName = "testfiles.txt"
os.chdir(ROOT_DIR)
margin = 0.01
epoch = 10
cubic_size = 256

register_pairs = {}

class Image:
    def __init__(self, reg_dir):
        #self.image_list = []
        #self.aseg_list = []
        self.reg_dir = reg_dir
        self.parse_images()
        self.parse_registration()
        #self.make_xarray()
        
    def parse_images(self):
        images = self.reg_dir.split("-")
        assert(len(images)==2)
        self.moving_image = sitk.ReadImage(ROOT_DIR + "Brain2NIFI/" + images[1] + "/norm.nii")

        
    def parse_registration(self):
        param0=sitk.ReadParameterFile("BrainParameterMaps/"+self.reg_dir+"/TransformParameters.0.txt")
        param1=sitk.ReadParameterFile("BrainParameterMaps/"+self.reg_dir+"/TransformParameters.1.txt")
        param2=sitk.ReadParameterFile("BrainParameterMaps/"+self.reg_dir+"/TransformParameters.2.txt")
        transformixImageFilter=sitk.TransformixImageFilter()
        transformixImageFilter.LogToConsoleOff()
        transformixImageFilter.AddTransformParameterMap(param0)
        transformixImageFilter.AddTransformParameterMap(param1)
        transformixImageFilter.AddTransformParameterMap(param2)
        self.transformixImageFilter=transformixImageFilter

    
    def register_points(self, test_file='test.pts'):
        if os.path.exists('outputpoints.txt'):
            os.remove('outputpoints.txt')
        self.transformixImageFilter.SetFixedPointSetFileName(test_file)
        self.transformixImageFilter.SetMovingImage(self.moving_image)
        self.transformixImageFilter.Execute()

def get_data(path):
	heart = sitk.ReadImage(path)
	heartArray = np.array([sitk.GetArrayFromImage(heart)]) / 256
	return heartArray

def conv_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        act_fn,
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
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


def find_points(point_list, image):
    # 1) write the point to file
    if os.path.exists('test.pts'):
        os.remove('test.pts')
    fr = open('test.pts', 'w')
    fr.write('index' + '\n' + str(len(point_list)))
    for point in point_list:
        fr.write('\n'+str(point[0])+' '+str(point[1])+' '+str(point[2]))
    fr.close()

    # find the corresponding point
    image.register_points()
    transformed_points = get_output_points()
    print(transformed_points.shape)
    return transformed_points

class BrainImageDataset(Dataset):
    def __init__(self, dirList):
        self.data = dirList

    def __getitem__(self, index):

        fix = self.data[index]
        fixed_image_array = get_data(ROOT_DIR + "Brain2NIFI/" + fix + "/norm.nii")
        moving = register_pairs[fix]
        moving_image_array = get_data(ROOT_DIR + "Brain2NIFI/" + moving + "/norm.nii")
        return (fixed_image_array, moving_image_array, fix, moving)

    def __len__(self):
        return len(self.data)

class featureLearner(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(featureLearner, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = nn.LeakyReLU()

        print("\n------Initiating Network------\n")

        self.cnn1 = conv_block_3d(self.in_dim, self.out_dim, act_fn)
        #self.reset_params()

    @staticmethod
    def weight_init(m):
        if (isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d)):
            # todo: change it to kaiming initialization
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
        file = open(ROOT_DIR + trainFileName, "r")
    #read file with test data path
    else:
        file = open(ROOT_DIR + testFileName, "r")

    dirnames = file.readlines()
    data_directory = [x.strip() for x in dirnames]

    dirList = []
    for i in data_directory:
        if i in register_pairs:
            dirList.append(i)
    return dirList

def load_pairs():
    for root, directories, filenames in os.walk(ROOT_DIR +"BrainParameterMaps"):
        for pairname in directories:
            images = pairname.split("-")
            assert(len(images)==2)
            register_pairs[images[0]] = images[1]


def find_postive_negative_points(image, fixed_image_array, moving_image_array, Npoints):

    point_list = []
    negative_point_list = []

    for i in range(Npoints):
        while(1):
            x = random.randint(0,255)
            y = random.randint(0,255)
            z = random.randint(0,255)
            fixed_point = np.array([x,y,z]).astype('int')
            if(fixed_image_array[0][0][x][y][z] != 0):
                break
        #generate negative point
        while(1):
            x = random.randint(0,255)
            y = random.randint(0,255)
            z = random.randint(0,255)
            negative_point = np.array([x,y,z]).astype('int')
            if(moving_image_array[0][0][x][y][z] != 0):
                break
        point_list.append(fixed_point)
        negative_point_list.append(negative_point)

    positive_point_list = find_points(point_list,image)

    return point_list, positive_point_list, negative_point_list


def check_boundary(a,b,c,x,y,z):
	return (a>=0 and a<256) and (b>=0 and b<256) and (c>=0 and c<256) and (x>=0 and x<256) and (y>=0 and y<256) and (z>=0 and z<256)  


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
        cnt = 0

        '''
        for i in range(self.N):
            x, y, z = fixed_points[i]
            a, b, c = positive_points[i]
            if(check_boundary(a,b,c,x,y,z) == 0):
                continue
            label = 1 # positive pair
            distance = (fix_image_feature[0][:,x,y,z] - moving_image_feature[0][:,a,b,c]).pow(2).sum()  # squared distance
            #print("pos "+str(math.sqrt(distance)))
            loss += label * (distance ** 2) + (1-label) * ((max(0, self.margin-math.sqrt(distance))) ** 2)
            cnt += 1
        '''
        for i in range(self.N):
            x, y, z = fixed_points[i]
            a, b, c = negative_points[i]
            if(check_boundary(a,b,c,x,y,z) == 0):
                continue
            label = 0 # negative pair
            distance = (fix_image_feature[0][:,x,y,z] - moving_image_feature[0][:,a,b,c]).pow(2).sum()  # squared distance
            #print("neg " + str(math.sqrt(distance)))
            #loss += label * (distance ** 2) + (1-label) * ((max(0, self.margin-math.sqrt(distance))) ** 2)
            loss += (0.01-distance)
            cnt += 1

        loss /= (2*cnt)
        loss *= 10000
        return loss

  

def train(args, model, device, loader, optimizer, epoch):

    model.train()
    criterion = CorrespondenceContrastiveLoss(margin, args.batch)
    save_loss_filename = "loss"+str(epoch)+".npy"
    loss_history = []

    for batch_idx, (fixed_image_array, moving_image_array, fix, moving) in enumerate(loader):
        #print(fix, type(fix))
        #print(moving, type(moving))
        image = Image("".join(fix)+"-"+"".join(moving))
        point_list, positive_point_list, negative_point_list = \
            find_postive_negative_points(image, fixed_image_array, moving_image_array, args.Npoints)

        mini_batch = 0
        losses = []
        while(1):
            fixed_image_array, moving_image_array = fixed_image_array.to(device), moving_image_array.to(device)
            optimizer.zero_grad()
            fixed_image_feature = model(fixed_image_array.float())
            moving_image_feature = model(moving_image_array.float())

            start_pos = mini_batch * args.batch
            end_pos = (mini_batch+1) * args.batch
            loss = criterion(fixed_image_feature, moving_image_feature,
                             point_list[start_pos:end_pos],
                             positive_point_list[start_pos:end_pos],
                             negative_point_list[start_pos:end_pos])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            mini_batch += 1
            if(mini_batch*args.batch == args.Npoints):
                mini_batch = 0
                '''
                randnum = random.randint(0, 100)
                random.seed(randnum)
                random.shuffle(point_list)
                random.seed(randnum)
                random.shuffle(positive_point_list)
                random.seed(randnum)
                random.shuffle(negative_point_list)
                '''
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx,
                    100. * batch_idx / loader.__len__(), np.array(losses).mean()))
                losses = []
            #loss_history.append(loss.item())
            #if(len(loss_history) % 10 == 0):
            #    np.save(save_loss_filename,np.array(loss_history))


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--test-model', type=str, default='', metavar='N',
                    help='If test-model has a name, load pre-trained model')
parser.add_argument('--predict-model', type=str, default='', metavar='N',
                    help='If predict-model has a name, do not do training, just give result on dev and test set')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.00001)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='LR',
                    help='weight decay')
parser.add_argument('--epoch', type=int, default=1, metavar='LR',
                    help='epoch')
parser.add_argument('--Npoints', type=int, default=200, metavar='LR',
                    help='number of points for each image')
parser.add_argument('--batch', type=int, default=200, metavar='LR',
                    help='batch size of each update')
parser.add_argument('--log_interval', type=int, default=1, metavar='LR',
                    help='log_interval')

input_args = parser.parse_args()


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using device: "+str(device))

#store all pairs of registration
load_pairs()

model = featureLearner(1,3).to(device)
optimizer = optim.Adam(model.parameters(), lr=input_args.lr, betas=(0.9, 0.99), weight_decay=input_args.wd)

train_dataset = BrainImageDataset(load_Directory(True))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1, shuffle=True)

for epoch in range(1, input_args.epoch+1):
    train(input_args, model, device, train_loader, optimizer, epoch)
    model.save(epoch)

