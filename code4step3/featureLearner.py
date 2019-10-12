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

def conv_block_3d(in_dim,out_dim,act_fn,dilation,is_final=False):
    if(is_final):
        model = nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1, dilation=dilation)
    else:
        model = nn.Sequential(
            nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1, dilation=dilation),
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

        self.cnn1 = conv_block_3d(self.in_dim, self.mid1_dim, act_fn, 1)
        self.cnn2 = conv_block_3d(self.mid1_dim, self.mid2_dim, act_fn, 1)
        self.cnn3 = conv_block_3d(self.mid2_dim, self.mid3_dim, act_fn, 1)
        self.cnn4 = conv_block_3d(self.mid3_dim, self.out_dim, act_fn, 1, True)
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
        #get_gpu_info(4)
        x = self.cnn2(x)
        #get_gpu_info(6)
        x = self.cnn3(x)
        #get_gpu_info(8)
        out = self.cnn4(x)
        #get_gpu_info(9)
        return out

    def save(self,epoch):
        torch.save(self.state_dict(),"featureLearner"+'_'+str(epoch)+'.pt')


def point_redirection(x, y, z):
    # Todo: fix this
    x = (x-crop_index[0]) % crop_half_size[0]
    y = (y-crop_index[1]) % crop_half_size[1]
    z = (z-crop_index[2]) % crop_half_size[2]
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

    for x_shard in range(2):
        for y_shard in range(2):
            for z_shard in range(2):
                for i in range(Npoints*2//8):
                    while(1):
                        x = random.randint(crop_index[0],crop_index[0]+crop_half_size[0])+crop_half_size[0]*x_shard
                        y = random.randint(crop_index[2],crop_index[2]+crop_half_size[1])+crop_half_size[1]*y_shard
                        z = random.randint(crop_index[4],crop_index[4]+crop_half_size[2])+crop_half_size[2]*z_shard
                        fixed_point = np.array([x,y,z]).astype('int')
                        if(fixed_image_array[0][0][x][y][z] != 0):
                            break
                    #generate negative point
                    while(1):
                        x = random.randint(crop_index[0],crop_index[0]+crop_half_size[0])+crop_half_size[0]*x_shard
                        y = random.randint(crop_index[2],crop_index[2]+crop_half_size[1])+crop_half_size[1]*y_shard
                        z = random.randint(crop_index[4],crop_index[4]+crop_half_size[2])+crop_half_size[2]*z_shard
                        negative_point = np.array([x,y,z]).astype('int')
                        if(moving_image_array[0][0][x][y][z] != 0):
                            break
                    point_list.append(fixed_point)
                    negative_point_list.append(negative_point)

    positive_point_list = find_points(point_list,image)
    print(positive_point_list.shape)
    point_list = list(np.array(point_list).reshape((8, Npoints * 2 // 8, 3)))
    negative_point_list = list(np.array(negative_point_list).reshape((8, Npoints * 2 // 8, 3)))
    positive_point_list = list(positive_point_list.reshape((8, Npoints * 2 // 8, 3)))

    for i in range(8):
        x_shard = i//4
        y_shard = (i%4)//2
        z_shard = i%2

        cnt = 0
        good_list = []
        for item in positive_point_list[i]:
            if(check_boundary_new(item[0],item[1],item[2], x_shard, y_shard, z_shard)):
                good_list.append(cnt)
                if(len(good_list) == Npoints//8):
                    break
                cnt += 1
        point_list[i] = [point_list[i][index] for index in good_list]
        positive_point_list[i] = [positive_point_list[i][index] for index in good_list]
        negative_point_list[i] = [negative_point_list[i][index] for index in good_list]

        if(len(good_list) != Npoints//8):
            print("only part data generated : "+str(len(good_list)))

    return point_list, positive_point_list, negative_point_list


def check_boundary_new(a,b,c, x_shard, y_shard, z_shard):
    # Todo: fix that
	return (a>=crop_index[0]+x_shard*crop_half_size[0] and a<crop_index[0]+crop_half_size[0]+x_shard*crop_half_size[0]) \
           and (b>=crop_index[2]+y_shard*crop_half_size[1] and b<crop_index[2]+crop_half_size[1]+y_shard*crop_half_size[1])\
           and (c>=crop_index[4]+z_shard*crop_half_size[2] and c<crop_index[4]+crop_half_size[2]+z_shard*crop_half_size[2])


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
            x, y, z = point_redirection(x, y, z)
            a, b, c = point_redirection(a, b, c)
            distance = (fix_image_feature[0][:,x,y,z] - moving_image_feature[0][:,a,b,c]).pow(2).sum()  # squared distance
            print(distance)
            #print("pos "+str(math.sqrt(distance)))
            loss += (distance ** 2)
            cnt += 1

        # negative pairs
        for i in range(self.batch):
            x, y, z = fixed_points[i]
            a, b, c = negative_points[i]
            x, y, z = point_redirection(x, y, z)
            a, b, c = point_redirection(a, b, c)
            distance = (fix_image_feature[0][:,x,y,z] - moving_image_feature[0][:,a,b,c]).pow(2).sum()  # squared distance
            #print("neg " + str(math.sqrt(distance)))
            print(distance)
            loss += ((max(0, self.margin-torch.sqrt(distance))) ** 2)
            #loss += ((0.01-torch.sqrt(distance))**2)
            cnt += 1

        loss /= (2*cnt)
        loss *= 1000000
        return loss

crop_index = [25, 225, 28, 204, 48, 208]
crop_size = [200, 176, 160]
crop_half_size = [100, 88, 80]

def train(args, model, device, loader, optimizer):

    model.train()
    criterion = CorrespondenceContrastiveLoss(args.margin, args.batch)
    timeStr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    os.mkdir(timeStr)
    save_loss_filename = timeStr+"/loss.npy"
    save_model_filename = timeStr+"/model"
    loss_history = []

    #index_range = [255,0,255,0,255,0]

    for epoch_idx, (fixed_image_array, moving_image_array, fix, moving) in enumerate(loader):
        #print(fix, type(fix))
        #print(moving, type(moving))
        # if we only want to generate points
        print(epoch_idx)

        '''
        # for finding the boundary, which is [11, 228, 25, 221, 47, 209]
        # max 100 boundary, is [25, 221, 28, 205, 48, 209]
        # for now we use [25, 224, 28, 203, 48, 207]
        for image in [fixed_image_array, moving_image_array]:
            index = []
            image = image[0][0]
            dim = image.sum(dim=[1, 2])
            for i in range(0,256):
                if(dim[i] != 0):
                    index.append(i)
                    break
            for i in range(255,-1,-1):
                if(dim[i] != 0):
                    index.append(i)
                    break
            dim = image.sum(dim=[0, 2])
            for i in range(0,256):
                if(dim[i] != 0):
                    index.append(i)
                    break
            for i in range(255,-1,-1):
                if(dim[i] != 0):
                    index.append(i)
                    break
            dim = image.sum(dim=[0, 1])
            for i in range(0,256):
                if(dim[i] != 0):
                    index.append(i)
                    break
            for i in range(255,-1,-1):
                if(dim[i] != 0):
                    index.append(i)
                    break
            #print(index)
            old_index_range = copy.deepcopy(index_range)
            index_range[0] = min(index_range[0], index[0])
            index_range[1] = max(index_range[1], index[1])
            index_range[2] = min(index_range[2], index[2])
            index_range[3] = max(index_range[3], index[3])
            index_range[4] = min(index_range[4], index[4])
            index_range[5] = max(index_range[5], index[5])
            if(old_index_range != index_range):
                print(index_range)
        continue
        '''

        if(os.path.exists("points_data/"+"".join(fix)+"-"+"".join(moving)+"-points.npy")):
            points_data = np.load("points_data/"+"".join(fix)+"-"+"".join(moving)+"-points.npy")
            point_list = np.array(points_data[0])
            positive_point_list = np.array(points_data[1])
            negative_point_list = np.array(points_data[2])
        else:
            image = Image("".join(fix)+"-"+"".join(moving))
            print("".join(fix)+"-"+"".join(moving))
            point_list, positive_point_list, negative_point_list = \
                find_postive_negative_points(image, fixed_image_array, moving_image_array, args.Npoints)
            points_data = np.array([point_list, positive_point_list, negative_point_list])
            np.save("points_data/"+"".join(fix)+"-"+"".join(moving)+"-points.npy",points_data)

        #continue

        # crop image and triple points here
        fixed_image_array = fixed_image_array[:, :, crop_index[0]:crop_index[1], crop_index[2]:crop_index[3],
                            crop_index[4]:crop_index[5]]
        moving_image_array = moving_image_array[:, :, crop_index[0]:crop_index[1], crop_index[2]:crop_index[3],
                             crop_index[4]:crop_index[5]]
        point_list -= crop_index[0]
        positive_point_list -= crop_index[2]
        negative_point_list -= crop_index[4]

        losses = []

        for x_shard in range(2):
            for y_shard in range(2):
                for z_shard in range(2):

                    mini_batch = 0

                    while (1):
                        part_fixed_image_array = fixed_image_array[:, :,
                                            x_shard * crop_half_size[0]:(x_shard + 1) * crop_half_size[0],
                                            y_shard * crop_half_size[1]:(y_shard + 1) * crop_half_size[1],
                                            z_shard * crop_half_size[2]:(z_shard + 1) * crop_half_size[2]]
                        part_moving_image_array = moving_image_array[:, :,
                                             x_shard * crop_half_size[0]:(x_shard + 1) * crop_half_size[0],
                                             y_shard * crop_half_size[1]:(y_shard + 1) * crop_half_size[1],
                                             z_shard * crop_half_size[2]:(z_shard + 1) * crop_half_size[2]]

                        part_fixed_image_array, part_moving_image_array = part_fixed_image_array.to(device), part_moving_image_array.to(device)
                        optimizer.zero_grad()

                        fixed_image_feature = model(part_fixed_image_array.float())
                        moving_image_feature = model(part_moving_image_array.float())

                        start_pos = mini_batch * args.batch
                        end_pos = (mini_batch+1) * args.batch
                        loss = criterion(fixed_image_feature, moving_image_feature,
                                         point_list[4*x_shard+2*y_shard+z_shard][start_pos:end_pos],
                                         positive_point_list[4*x_shard+2*y_shard+z_shard][start_pos:end_pos],
                                         negative_point_list[4*x_shard+2*y_shard+z_shard][start_pos:end_pos])
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())

                        mini_batch += 1

                        if(mini_batch % args.log_interval == 0):
                            print('Train Epoch: '+str(epoch_idx) + " Corner : "+str(4*x_shard+2*y_shard+z_shard)+" mini_batch: "+
                                  str(mini_batch)+"  percentage: "+
                                  str(100. * epoch_idx / loader.__len__())+"% loss: "+
                                  str(np.array(losses[-1*args.log_interval:]).mean()))

                        if(mini_batch*args.batch*8 == args.Npoints):
                            break


        loss_history.append(np.array(losses).mean())
        if(len(loss_history) % args.loss_save_interval == 0):
            np.save(save_loss_filename,np.array(loss_history))
        if(len(loss_history) % args.model_save_interval == 0):
            torch.save(model, save_model_filename+str(epoch_idx)+'.pt')




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
parser.add_argument('--batch', type=int, default=50, metavar='LR',
                    help='batch size of each update')
parser.add_argument('--log_interval', type=int, default=5, metavar='LR',
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
#summary(model, input_size=(1, 256,256,256))
print(model)
optimizer = optim.Adam(model.parameters(), lr=input_args.lr, betas=(0.9, 0.99), weight_decay=input_args.wd)

train_dataset = BrainImageDataset(load_Directory(True))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1, shuffle=True)

train(input_args, model, device, train_loader, optimizer)
#model.save(0)

