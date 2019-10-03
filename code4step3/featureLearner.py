import cv2
import numpy as np
import os
import copy
import SimpleITK as sitk
import random
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from scipy import spatial
import argparse
import pandas as pd
import pickle as pkl
import json
import sys
from pynvml import *
from torchsummary import summary


# following three functions are used for checking gpu usage
def getGpuUtilization(handle):
    try:
        util = nvmlDeviceGetUtilizationRates(handle)
        gpu_util = int(util.gpu)
    except NVMLError as err:
        error = handleError(err)
        gpu_util = error
    return gpu_util

def getMB(BSize):
    return BSize / (1024 * 1024)

def get_gpu_info(flag):
    print(flag)
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    data = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        gpu_util = getGpuUtilization(handle)
        one = {"gpuUtil": gpu_util}
        one["gpuId"] = i
        one["memTotal"] = getMB(meminfo.total)
        one["memUsed"] = getMB(meminfo.used)
        one["memFree"] = getMB(meminfo.total)
        one["temperature"] = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        data.append(one)
    data = {"gpuCount": deviceCount, "util": "Mb", "detail": data}
    print(json.dumps(data))


torch.backends.cudnn.enabled = False

ROOT_DIR = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/"
trainFileName = "trainfiles.txt"
testFileName = "testfiles.txt"
os.chdir(ROOT_DIR)


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

def conv_block_3d(in_dim,out_dim,act_fn,is_final=False):
    if(is_final):
        model = nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1)
    else:
        model = nn.Sequential(
            nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
            act_fn,
        )
    return model

def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool

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
    def __init__(self):
        super(featureLearner, self).__init__()

        self.in_dim = 1
        self.mid1_dim = 8
        self.mid2_dim = 16
        self.mid3_dim = 32
        self.out_dim = 32
        #act_fn = nn.LeakyReLU()
        act_fn = nn.ReLU()

        print("\n------Initiating Network------\n")

        self.cnn1 = conv_block_3d(self.in_dim, self.mid1_dim, act_fn)
        self.pool1 = maxpool_3d()
        self.cnn2 = conv_block_3d(self.mid1_dim, self.mid2_dim, act_fn)
        self.pool2 = maxpool_3d()
        self.cnn3 = conv_block_3d(self.mid2_dim, self.mid3_dim, act_fn)
        self.pool3 = maxpool_3d()
        self.cnn4 = conv_block_3d(self.mid3_dim, self.out_dim, act_fn, True)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if (isinstance(m, nn.Conv3d)):
            # todo: change it to kaiming initialization
            nn.init.kaiming_normal(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        #get_gpu_info(2)
        x = self.cnn1(x)
        #get_gpu_info(3)
        x = self.pool1(x)
        #get_gpu_info(4)
        x = self.cnn2(x)
        #get_gpu_info(5)
        x = self.pool2(x)
        #get_gpu_info(6)
        x = self.cnn3(x)
        #get_gpu_info(7)
        x = self.pool3(x)
        #get_gpu_info(8)
        out = self.cnn4(x)
        #get_gpu_info(9)
        return out

    def save(self,epoch):
        torch.save(self.state_dict(),"featureLearner"+'_'+str(epoch)+'.pt')


def point_redirection(x, y, z):
    x = x//8
    y = y//8
    z = z//8
    return x, y, z

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

    def __init__(self, margin, batch):
        super(CorrespondenceContrastiveLoss, self).__init__()
        self.margin = margin
        self.batch = batch

    def forward(self, fix_image_feature, moving_image_feature, fixed_points, positive_points, negative_points):
        loss = 0
        cnt = 0

        #print([len(fixed_points),len(positive_points),len(negative_points)])
        # positive pairs
        for i in range(self.batch):
            x, y, z = fixed_points[i]
            a, b, c = positive_points[i]
            if(check_boundary(a,b,c,x,y,z) == 0):
                continue
            x, y, z = point_redirection(x, y, z)
            a, b, c = point_redirection(a, b, c)
            distance = (fix_image_feature[0][:,x,y,z] - moving_image_feature[0][:,a,b,c]).pow(2).sum()  # squared distance
            #print("pos "+str(math.sqrt(distance)))
            loss += (distance ** 2)
            cnt += 1

        # negative pairs
        for i in range(self.batch):
            x, y, z = fixed_points[i]
            a, b, c = negative_points[i]
            if(check_boundary(a,b,c,x,y,z) == 0):
                continue
            x, y, z = point_redirection(x, y, z)
            a, b, c = point_redirection(a, b, c)
            distance = (fix_image_feature[0][:,x,y,z] - moving_image_feature[0][:,a,b,c]).pow(2).sum()  # squared distance
            #print("neg " + str(math.sqrt(distance)))
            loss += ((max(0, self.margin-torch.sqrt(distance))) ** 2)
            #loss += ((0.01-torch.sqrt(distance))**2)
            cnt += 1

        loss /= (2*cnt)
        loss *= 1000000
        return loss



def train(args, model, device, loader, optimizer):

    model.train()
    criterion = CorrespondenceContrastiveLoss(args.margin, args.batch)
    timeStr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    save_loss_filename = "loss_"+timeStr+".npy"
    loss_history = []

    for epoch_idx, (fixed_image_array, moving_image_array, fix, moving) in enumerate(loader):
        #print(fix, type(fix))
        #print(moving, type(moving))
        # if we only want to generate points
        print(epoch_idx)
        if(os.path.exists("points_data/"+"".join(fix)+"-"+"".join(moving)+"-points.npy")):
            points_data = np.load("points_data/"+"".join(fix)+"-"+"".join(moving)+"-points.npy")
            point_list = points_data[0]
            positive_point_list = points_data[1]
            negative_point_list = points_data[2]
        else:
            image = Image("".join(fix)+"-"+"".join(moving))
            print("".join(fix)+"-"+"".join(moving))
            point_list, positive_point_list, negative_point_list = \
                find_postive_negative_points(image, fixed_image_array, moving_image_array, args.Npoints)
            points_data = np.array([point_list, positive_point_list, negative_point_list])
            np.save("points_data/"+"".join(fix)+"-"+"".join(moving)+"-points.npy",points_data)

        mini_batch = 0
        losses = []
        while(1):
            fixed_image_array, moving_image_array = fixed_image_array.to(device), moving_image_array.to(device)
            optimizer.zero_grad()
            #get_gpu_info(1)
            #print(sys.getsizeof(fixed_image_array)/(8*1024*1024*1024))
            fixed_image_feature = model(fixed_image_array.float())

            #print(sys.getsizeof(fixed_image_feature)/(8*1024*1024*1024))
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

            if(mini_batch % args.log_interval == 0):
                print('Train Epoch: '+str(epoch_idx) + " mini_batch: "+str(mini_batch)+"  percentage: "+
                      str(100. * epoch_idx / loader.__len__())+"% loss: "+
                      str(np.array(losses[-1*args.log_interval:]).mean()))

            if(mini_batch*args.batch == args.Npoints):

                '''
                # if only train one time, do not need to reset mini_batch and shuffle data
                mini_batch = 0
                losses = []
                randnum = random.randint(0, 100)
                random.seed(randnum)
                random.shuffle(point_list)
                random.seed(randnum)
                random.shuffle(positive_point_list)
                random.seed(randnum)
                random.shuffle(negative_point_list)
                '''

                loss_history.append(np.array(losses).mean())
                if(len(loss_history) % args.loss_save_interval == 0):
                    np.save(save_loss_filename,np.array(loss_history))
                break




parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--test-model', type=str, default='', metavar='N',
                    help='If test-model has a name, load pre-trained model')
parser.add_argument('--predict-model', type=str, default='', metavar='N',
                    help='If predict-model has a name, do not do training, just give result on dev and test set')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.00001)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='LR',
                    help='weight decay')
parser.add_argument('--margin', type=float, default=0.01, metavar='LR',
                    help='margin')
parser.add_argument('--epoch', type=int, default=1, metavar='LR',
                    help='epoch')
parser.add_argument('--Npoints', type=int, default=10000, metavar='LR',
                    help='number of points for each image')
parser.add_argument('--batch', type=int, default=200, metavar='LR',
                    help='batch size of each update')
parser.add_argument('--log_interval', type=int, default=10, metavar='LR',
                    help='log_interval')
parser.add_argument('--loss_save_interval', type=int, default=1, metavar='LR',
                    help='loss_save_interval')
parser.add_argument('--model_save_interval', type=int, default=10, metavar='LR',
                    help='model_save_interval')
parser.add_argument('--cubic_size', type=int, default=256, metavar='LR',
                    help='cubic_size')

input_args = parser.parse_args()


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using device: "+str(device))

#store all pairs of registration
load_pairs()

model = featureLearner().to(device)
summary(model, input_size=(1, 256,256,256))
print(model)
optimizer = optim.Adam(model.parameters(), lr=input_args.lr, betas=(0.9, 0.99), weight_decay=input_args.wd)

train_dataset = BrainImageDataset(load_Directory(True))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1, shuffle=True)

train(input_args, model, device, train_loader, optimizer)
#model.save(0)

