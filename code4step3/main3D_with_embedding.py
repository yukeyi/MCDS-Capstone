import os
import random
import time
import cv2
import shutil
import argparse
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tensorboardX import SummaryWriter

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse
import SimpleITK as sitk

import torchvision.transforms as transforms


def conv_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model


def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim,out_dim//2,act_fn),
        nn.Conv3d(out_dim//2,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model


def conv_block_3_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model

def conv_block_4_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def fl_conv_block_3d(in_dim,out_dim,act_fn,dilation,is_final=False):
    if(is_final):
        model = nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
    else:
        model = nn.Sequential(
            nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            act_fn,
        )
    return model


class featureLearner(nn.Module):
    def __init__(self):
        super(featureLearner, self).__init__()

        self.in_dim = 1
        self.mid1_dim = 32
        self.mid2_dim = 32
        self.mid3_dim = 16
        self.out_dim = 8
        act_fn = nn.ReLU()

        print("\n------Initiating Network------\n")

        self.cnn1 = fl_conv_block_3d(self.in_dim, self.mid1_dim, act_fn, 1)
        self.cnn2 = fl_conv_block_3d(self.mid1_dim, self.mid2_dim, act_fn, 2)
        self.cnn3 = fl_conv_block_3d(self.mid2_dim, self.mid3_dim, act_fn, 4)
        self.cnn4 = fl_conv_block_3d(self.mid3_dim, self.out_dim, act_fn, 8, True)
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
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        out = self.cnn4(x)
        return out

    def save(self,epoch):
        torch.save(self.state_dict(),"featureLearner"+'_'+str(epoch)+'.pt')


class UnetGenerator_3d(nn.Module):

    def __init__(self, in_dim, out_dim, num_filter):
        super(UnetGenerator_3d, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.ReLU()

        print("\n------Initiating U-Net------\n")

        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_2_3d(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = maxpool_3d()
        self.down_3 = conv_block_2_3d(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = maxpool_3d()

        self.bridge = conv_block_2_3d(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1 = conv_trans_block_3d(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = conv_block_3_3d(self.num_filter * 12, self.num_filter * 4, act_fn)
        self.trans_2 = conv_trans_block_3d(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = conv_block_3_3d(self.num_filter * 6, self.num_filter * 2, act_fn)
        self.trans_3 = conv_trans_block_3d(self.num_filter * 2, self.num_filter * 2, act_fn)
        self.up_3 = conv_block_3_3d(self.num_filter * 3, self.num_filter * 1, act_fn)

        self.out = conv_block_4_3d(self.num_filter, out_dim, nn.LogSoftmax())
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
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        bridge = self.bridge(pool_3)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        up_3 = self.up_3(concat_3)
        out = self.out(up_3)

        return out

def get_data(path):
    heart = sitk.ReadImage(path)
    heartArray = np.array([sitk.GetArrayFromImage(heart)]) / 256
    return heartArray

def load_Directory(is_train):
    #read file with train data path
    if is_train:
        file = open(ROOT_DIR + trainFileName, "r")
    #read file with test data path
    else:
        file = open(ROOT_DIR + testFileName, "r")

    dirnames = file.readlines()
    data_directory = [x.strip() for x in dirnames]
    if is_train:
        data_directory = data_directory[:args.num_train]
    else:
        data_directory = data_directory[:args.num_dev]
    return data_directory


def abstract_feature(image):
    #print(image.shape)
    with torch.no_grad():
        feature = fl_model(torch.tensor([image]).to(device))
    #print(feature.shape)
    return feature[0]


class BrainImageDataset(Dataset):
    def __init__(self, dirList):
        self.data = dirList

    def __getitem__(self, index):

        name = self.data[index]
        try:
            image = get_data(ROOT_DIR + "Brain2NIFI/" + name + "/norm.nii").astype("float32")
            target = get_data(ROOT_DIR + "Brain2NIFI/" + name + "/aseg.nii")*256
        except:
            return (np.array([]), np.array([]))

        label = np.zeros(target.shape)
        for i in range(len(label_list)):
            label += ((target == label_list[i])*i)
        label = label[0]
        image = abstract_feature(image)

        return (image, label)

    def __len__(self):
        return len(self.data)


def get_loss(dl, model):

    model.eval()
    total_loss = 0.0
    for batch_idx, (whole_data, whole_label) in enumerate(dl):
        slice_depth = 256 // args.shard
        for shard in range(args.shard):
            data, target = whole_data[:,:,shard*slice_depth:(shard+1)*slice_depth].to(device), \
                           whole_label[:,shard*slice_depth:(shard+1)*slice_depth].to(device)

            logsoftmax_output_z = model(data)
            loss = nn.NLLLoss(reduce=False)(logsoftmax_output_z, target.long())
            #loss = loss.float().mean()
            loss = (loss.float() * (args.augmentation * (target > 0) + 1).float()).mean()
            total_loss += loss.item()

    model.train()

    return total_loss / len(dl)


'''
main_class: the class that you want to predict as one, must be a single value
'''


def binary_vector(x, main_class):
    length = len(x)
    binary = main_class

    return (binary == x).astype(int)


'''
classes: a list of labels that we want for binary comparison
e.g. [1, 2] will return a list of two scores. The first index
is the score of regarding 1 as 1 and 2 as 0. The second is the
score of regarding 2 as 1 and 1 as 0. 

*WARNING: label and target must be of the same dimension. 
'''


def binary_dice_score(label, target, classes):
    scores = []
    smooth = 1

    for cls in classes:
        label_binary = binary_vector(label.flatten(), cls)
        target_binary = binary_vector(target.flatten(), cls)
        intersection = np.sum(label_binary * target_binary)
        normalization = np.sum(label_binary + target_binary)
        score = ((2. * intersection + smooth).sum() /
                 (normalization + smooth).sum())
        scores.append(score)
    return scores


def get_accuracy(dl, model):

    total_num = 0
    correct_num = 0
    slice_depth = 256 // args.shard
    for whole_data, y in dl:
        predicted = np.zeros(y.shape)
        for shard in range(args.shard):
            X = whole_data[:,:,shard*slice_depth:(shard+1)*slice_depth].to(device)
            X = Variable(X).to(device)
            output = model(X).cpu()
            output = np.argmax(output.data.numpy(), axis=1)
            predicted[:, shard * slice_depth:(shard + 1) * slice_depth] = output
        correct_num += (predicted == y.data.numpy().astype("int64")).sum().item()
        total_num += y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]
    return correct_num/total_num

def get_dice_score(dl, model):

    score = []
    slice_depth = 256 // args.shard
    for whole_data, y in dl:
        predicted = np.zeros(y.shape)
        for shard in range(args.shard):
            X = whole_data[:,:,shard*slice_depth:(shard+1)*slice_depth].to(device)
            X = Variable(X).to(device)
            output = model(X).cpu()
            output = np.argmax(output.data.numpy(),axis=1)
            predicted[:,shard*slice_depth:(shard+1)*slice_depth] = output
        #predicted.resize((predicted.shape[0]*predicted.shape[1],predicted.shape[2],predicted.shape[3]))

        score.append(binary_dice_score(np.array(y),predicted.astype("int64"), list(np.arange(0,46))))

    return np.mean(score)


def value2label(x):
    return label_map[x]

def get_label_map():
    label_list = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13,
                  14, 15, 16, 17, 18, 24, 26, 28, 30,
                  31, 41, 42, 43, 44, 46, 47, 49, 50,
                  51, 52, 53, 54, 58, 60, 62, 63, 72,
                  77, 80, 85, 251, 252, 253, 254, 255]
    label_map = {}
    for i in range(len(label_list)):
        label_map[label_list[i]] = i
    return label_map, label_list


parser = argparse.ArgumentParser(description='UNET Implementation')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--classes', type=int, default=46, metavar='N',
                    help='total classes of task')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--shard', type=int, default=4, metavar='N',
                    help='split how many shards for one image')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--num_train', type=int, default=3000, metavar='N',
                    help='number of data for training')
parser.add_argument('--num_dev', type=int, default=5, metavar='N',
                    help='number of data for evaluation')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--augmentation', type=float, default=10.0, metavar='LR',
                    help='weight for lebeled object')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--channel-base', type=int, default=8, metavar='CB',
                    help='number of channel for first convolution (default: 8)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many epoches between logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--test-model', type=str, default='', metavar='N',
                    help='If test-model has a name, do not do training, just testing on dev and train set')
parser.add_argument('--load-model', type=str, default=None, metavar='N',
                    help='If load-model has a name, use pretrained model')
parser.add_argument('--embedding-model', type=str, default='featureLearningModel.pt', metavar='N',
                    help='pretrained feature learning model')
args = parser.parse_args()

#ROOT_DIR = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/"
ROOT_DIR = "/Users/yukeyi/Desktop/"
trainFileName = "trainfiles.txt"
testFileName = "testfiles.txt"
label_map, label_list = get_label_map()

timeStr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
os.mkdir(timeStr)
writer = SummaryWriter(timeStr+'/log')
save_model_filename = timeStr + "/model"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = BrainImageDataset(load_Directory(True))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
dev_dataset = BrainImageDataset(load_Directory(False))
dev_loader = torch.utils.data.DataLoader(dev_dataset,batch_size=args.test_batch_size, shuffle=False)

model = UnetGenerator_3d(8, args.classes, args.channel_base)
if(args.load_model is not None):
    print("Load model : "+args.load_model)
    model = torch.load(args.load_model)
print(device)
model = model.to(device)
print(model)

if(device == torch.device('cpu')):
    fl_model = torch.load(args.embedding_model,map_location='cpu')
else:
    fl_model = torch.load(args.embedding_model)
#torch.save(fl_model.state_dict(), "test.pt")
#fl_model = featureLearner()
#if(device == torch.device('cpu')):
#    fl_model.load_state_dict(torch.load("test.pt", map_location='cpu'))
#else:
#    fl_model.load_state_dict(torch.load("test.pt"))
fl_model = fl_model.to(device)
fl_model.eval()


best_dice = 0.0
best_jaccard = 0
optim = torch.optim.Adam(model.parameters(),lr=args.lr)


model.train()
for epoch in range(args.epochs):
    #get_dice_score(dev_loader, model)
    #get_accuracy(dev_loader, model)
    total_loss = 0.0
    for batch_idx, (whole_data, whole_label) in enumerate(train_loader):
        if (len(whole_data) > 0):
            try:
                image_loss = 0.0
                slice_depth = 256 // args.shard
                for shard in range(args.shard):
                    data, target = whole_data[:,:,shard*slice_depth:(shard+1)*slice_depth].to(device), \
                                   whole_label[:,shard*slice_depth:(shard+1)*slice_depth].to(device)

                    logsoftmax_output_z = model(data)
                    loss = nn.NLLLoss(reduce=False)(logsoftmax_output_z, target.long())
                    #loss = loss.float().mean()
                    loss = (loss.float()*(args.augmentation*(target>0)+1).float()).mean()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    total_loss += loss.item()
                    image_loss += loss.item()
                print("Batch : "+str(batch_idx) + " loss : "+str(image_loss))
            except:
                print("except occurs")

        if (batch_idx + 1) % args.log_interval == 0:

            print("Epoch : "+str(epoch)+" Batch: "+str(batch_idx))
            model.eval()
            train_loss = total_loss / train_loader.__len__()
            print("total loss : "+str(train_loss))
            dev_loss = get_loss(dev_loader, model)
            print("dev loss : "+str(dev_loss))
            writer.add_scalar('Train/Loss', train_loss, epoch*args.num_train+batch_idx+1)
            writer.add_scalar('Dev/Loss', dev_loss, epoch*args.num_train+batch_idx+1)
            #train_acc = get_accuracy(train_loader, model)
            #print("Training accuracy : " + str(train_acc))
            dev_dice = get_dice_score(dev_loader, model)
            print("Dev dice score : " + str(dev_dice))
            writer.add_scalar('Dev/Dice', dev_dice, epoch*args.num_train+batch_idx+1)
            torch.save(model, save_model_filename + str(epoch*args.num_train+batch_idx+1) + '.pt')
            #dev_acc = get_accuracy(dev_loader, model)
            #print("Dev accuracy : " + str(dev_acc))

            model.train()

print("Done")
