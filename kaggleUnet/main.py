import os
import random
import time
import cv2
import shutil
import argparse
from multiprocessing.dummy import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from skimage.morphology import binary_opening, disk, label
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary


import torchvision.transforms as transforms
from dataprepare import get_data
from dataprepare import load_labels


#torch.backends.cudnn.enabled = False

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=1,
                     groups=groups,
                     stride=1)

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels,
                                  out_channels,
                                  kernel_size=2,
                                  stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 merge_mode='concat',
                 up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels,
                                self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2*self.out_channels,
                                 self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)

        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=1, depth=5,
                 start_filts=4, up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x

'''
class MyCustomDataset(Dataset):
    def __init__(self, type):
        if(type == 'Train'):
            self.image = np.concatenate((total_image[0::10, :, :, :], total_image[1::10, :, :, :], total_image[2::10, :, :, :],
                                         total_image[3::10, :, :, :], total_image[4::10, :, :, :], total_image[5::10, :, :, :],
                                         total_image[6::10, :, :, :], total_image[7::10, :, :, :], total_image[8::10, :, :, :]))
            self.label = np.concatenate((total_label[0::10, :, :], total_label[1::10, :, :], total_label[2::10, :, :],
                                         total_label[3::10, :, :], total_label[4::10, :, :], total_label[5::10, :, :],
                                         total_label[6::10, :, :], total_label[7::10, :, :], total_label[8::10, :, :]))
            print(self.image.shape)
            print(self.label.shape)
        else:
            self.image = total_image[9::10, :, :, :]
            self.label = total_label[9::10, :, :]
            print(self.image.shape)
            print(self.label.shape)
    def __len__(self):
        return len(self.image)
    def __getitem__(self, idx):
        image = self.image[idx]
        mask = self.label[idx]
        return (image, mask)
'''

class MyCustomDataset(Dataset):
    def __init__(self, type, dev_heart):
        if(dev_heart == 0):
            from_num = 0
        else:
            from_num = heart_index[dev_heart-1][0]
        to_num = heart_index[dev_heart][0]
        if(type == 'Train'):
            self.image = np.concatenate((total_image[:from_num,:,:,:],total_image[to_num:,:,:,:]))
            self.label = np.concatenate((total_label[:from_num,:,:],total_label[to_num:,:,:]))
            print(self.image.shape)
            print(self.label.shape)
        else:
            self.image = total_image[from_num:to_num, :, :, :]
            self.label = total_label[from_num:to_num, :, :]
            print(self.image.shape)
            print(self.label.shape)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        mask = self.label[idx]
        return (image, mask)

def get_loss(dl, model):
    loss = 0
    for X, y in dl:
        X, y = Variable(X).cuda(), Variable(y).cuda()
        output = model(X)
        loss += F.cross_entropy(output, y.long()).data[0]
    loss = loss / len(dl)
    return loss


def one_hot(x, classes):
    #print(x.shape)
    #print(x.dtype)
    length = len(x)
    x_one_hot = np.zeros((classes, length))
    x_one_hot[x, np.arange(length)] = 1
    return x_one_hot


'''
label: ground truth label matrix or tensor
target: predicted label matrix or tensor
classes: number of classes in the label
'''


def dice_score(label, target, classes):
    smooth = 1.

    label_cols = one_hot(label.flatten(), classes)
    target_cols = one_hot(target.flatten(), classes)

    intersection = np.sum((label_cols * target_cols), axis=1)  # len = classes
    normalization = np.sum((label_cols + target_cols), axis=1)  # len = classes

    #print(intersection)
    #print(normalization)

    return ((2. * intersection + smooth).sum() /
            (normalization + smooth).sum())


'''
label: predicted label matrix or tensor
target: predicted label matrix or tensor
classes: number of classes in the label
*WARNING: label and target must be of the same dimension. 
'''


def jaccard_index(label, target):
    label_flat = label.flatten()
    target_flat = target.flatten()
    length = len(label_flat)
    assert (length == len(target_flat))

    union = (label_flat != target_flat).astype(int).sum() + length
    intersection = (label_flat == target_flat).astype(int).sum()

    return intersection / union


def get_accuracy(dl, model):

    total_num = 0
    correct_num = 0

    for X, y in dl:
        X = Variable(X).cuda()
        output = model(X).cpu()
        #print(output.shape)
        #print(y.shape)
        #print(y.type())
        #print(np.argmax(output.data.numpy()).dtype)
        correct_num += (np.argmax(output.data.numpy(),axis=1) == y.data.numpy().astype("int64")).sum().item()
        total_num += y.shape[0]*y.shape[1]*y.shape[2]

    return correct_num/total_num

def get_dice_score(dl, model):

    #batch_num = 0
    score = 0
    #img_sm = cv2.resize(img, (height, depth), interpolation=cv2.INTER_NEAREST)

    for X, y in dl:
        X = Variable(X).cuda()
        output = model(X).cpu()
        #print(output.shape)
        predicted = np.argmax(output.data.numpy().astype("int64"),axis=1)
        #print(predicted.shape)

        predicted_origin = [0]*predicted.shape[0]
        for idx in range(len(predicted)):
            img = predicted[idx, :, :]
            img_sm = cv2.resize(img, (label_original[heart_index[dev_heart][1]].shape[2], label_original[heart_index[dev_heart][1]].shape[1]), interpolation=cv2.INTER_NEAREST)
            predicted_origin[idx] = img_sm
        predicted_origin = np.array(predicted_origin)

        #print(predicted_origin.shape)
        ground_truth = label_original[heart_index[dev_heart][1]].astype("int64")
        #print(ground_truth.shape)
        #score = dice_score(y.data.numpy().astype("int64"),np.argmax(output.data.numpy().astype("int64"),axis=1), 3)
        score = dice_score(ground_truth,predicted_origin, 3)
        #print(batch_num)
        #batch_num += 1

    return score

def get_jaccard_score(dl, model):

    score = 0
    for X, y in dl:
        X = Variable(X).cuda()
        output = model(X).cpu()
        #print(output.shape)
        predicted = np.argmax(output.data.numpy().astype("int64"),axis=1)
        #print(predicted.shape)

        predicted_origin = [0]*predicted.shape[0]
        for idx in range(len(predicted)):
            img = predicted[idx, :, :]
            img_sm = cv2.resize(img, (label_original[heart_index[dev_heart][1]].shape[2], label_original[heart_index[dev_heart][1]].shape[1]), interpolation=cv2.INTER_NEAREST)
            predicted_origin[idx] = img_sm
        predicted_origin = np.array(predicted_origin)

        #print(predicted_origin.shape)
        ground_truth = label_original[heart_index[dev_heart][1]].astype("int64")
        #print(ground_truth.shape)
        #score = dice_score(y.data.numpy().astype("int64"),np.argmax(output.data.numpy().astype("int64"),axis=1), 3)
        score = jaccard_index(ground_truth,predicted_origin)

    return score

parser = argparse.ArgumentParser(description='UNET Implementation')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--figuresize', type=int, default=240, metavar='N',
                    help='size that we use for the model')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many epoches between logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--test-model', type=str, default='', metavar='N',
                    help='If test-model has a name, do not do training, just testing on dev and train set')
parser.add_argument('--load-model', type=str, default='', metavar='N',
                    help='If load-model has a name, use pretrained model')
args = parser.parse_args()

label_original = load_labels()
total_image, total_label, heart_index = get_data(args.figuresize)

dev_heart = 0
total_number_of_2Dfigure = 1497
timeStr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
os.mkdir(timeStr + "model")

while(dev_heart < 10):

    print("We are using heart "+str(heart_index[dev_heart][1]))
    train_loader = torch.utils.data.DataLoader(MyCustomDataset('Train', dev_heart), batch_size=args.batch_size, shuffle=True)
    if (dev_heart == 0):
        dev_loader = torch.utils.data.DataLoader(MyCustomDataset('Dev', dev_heart), batch_size=heart_index[0][0], shuffle=False)
    else:
        dev_loader = torch.utils.data.DataLoader(MyCustomDataset('Dev', dev_heart), batch_size=heart_index[dev_heart][0]-heart_index[dev_heart-1][0], shuffle=False)

    model = UNet(3, merge_mode='concat')
    summary(model, input_size=(1, args.figuresize, args.figuresize))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    model.train()

    best_dice = 0
    best_jaccard = 0
    optim = torch.optim.Adam(model.parameters(),lr=args.lr)

    os.mkdir(timeStr + "model/dice"+str(heart_index[dev_heart][1]))
    os.mkdir(timeStr + "model/jaccard"+str(heart_index[dev_heart][1]))

    for epoch in range(args.epochs):

        for batch_idx, (data, label) in enumerate(train_loader):

            data, target = data.to(device), label.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target.long())

            optim.zero_grad()
            loss.backward()
            optim.step()

        if (epoch + 1) % args.log_interval == 0:

            print("Epoch : "+str(epoch))
            model.eval()

            train_loss = get_loss(train_loader, model)
            print(train_loss)
            train_acc = get_accuracy(train_loader, model)
            dev_dice = get_dice_score(dev_loader, model)
            dev_jaccard = get_jaccard_score(dev_loader, model)
            dev_acc = get_accuracy(dev_loader, model)
            print("Training accuracy : " + str(train_acc))
            print("Dev dice score : " + str(dev_dice))
            print("Dev jaccard score : " + str(dev_jaccard))
            print("Dev accuracy : " + str(dev_acc))
            if(train_acc < 0.01):
                print("Bad initialization")
                exit(0)
            if(args.save_model and (dev_dice > best_dice)):
                torch.save(model.state_dict(), timeStr + "model/dice"+str(heart_index[dev_heart][1])+"/" + str(epoch) + ":" + str(dev_dice) + ".pt")
                best_dice = dev_dice
            if(args.save_model and (dev_jaccard > best_jaccard)):
                torch.save(model.state_dict(), timeStr + "model/jaccard"+str(heart_index[dev_heart][1])+"/" + str(epoch) + ":" + str(dev_jaccard) + ".pt")
                best_jaccard = dev_jaccard

            model.train()

    print("Done")

    dev_heart += 1