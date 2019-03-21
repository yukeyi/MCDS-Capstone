import os
import random
import time
import cv2
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

import torchvision.transforms as transforms
from dataprepare3D import get_data
from dataprepare3D import load_labels


'''
def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim), )


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)  # -> [1, 4, 128, 128, 128]
        pool_1 = self.pool_1(down_1)  # -> [1, 4, 64, 64, 64]

        down_2 = self.down_2(pool_1)  # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_2)  # -> [1, 8, 32, 32, 32]

        down_3 = self.down_3(pool_2)  # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_3)  # -> [1, 16, 16, 16, 16]

        down_4 = self.down_4(pool_3)  # -> [1, 32, 16, 16, 16]
        pool_4 = self.pool_4(down_4)  # -> [1, 32, 8, 8, 8]

        down_5 = self.down_5(pool_4)  # -> [1, 64, 8, 8, 8]
        pool_5 = self.pool_5(down_5)  # -> [1, 64, 4, 4, 4]

        # Bridge
        bridge = self.bridge(pool_5)  # -> [1, 128, 4, 4, 4]

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, down_5], dim=1)  # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1)  # -> [1, 64, 8, 8, 8]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4], dim=1)  # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]

        trans_5 = self.trans_5(up_4)  # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1], dim=1)  # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]

        # Output
        out = self.out(up_5)  # -> [1, 3, 128, 128, 128]
        return out

'''
'''

class Modified3DUNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter = 2, dropout = 0.0):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=dropout)
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        #self.upscale = nn.ConvTranspose3d(in_channels,
        #                          out_channels,
        #                          kernel_size=2,
        #                          stride=2)
        self.softmax = nn.Softmax(dim=1)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter*2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter*4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter*2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter*4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)


    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upscale(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upscale(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        seg_layer = out
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
        #out = out.view(-1, self.n_classes)
        out = self.softmax(out)
        return out, seg_layer
'''

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
        self.out_lovasz = conv_block_4_3d(self.num_filter, out_dim, nn.Softmax())
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
        if(args.use_lovasz):
            out = self.out_lovasz(up_3)
        else:
            out = self.out(up_3)

        return out

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
            self.image = np.concatenate((total_image[:from_num,:,:,:,:],total_image[to_num:,:,:,:,:]))
            self.label = np.concatenate((total_label[:from_num,:,:,:],total_label[to_num:,:,:,:]))
            print(self.image.shape)
            print(self.label.shape)
        else:
            self.image = total_image[from_num:to_num, :, :, :, :]
            self.label = total_label[from_num:to_num, :, :, :]
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
    if (args.use_lovasz):
        for X, y in dl:
            softmax_output_z = model(data)
            vprobas, vlabels = flatten_probas(softmax_output_z, target.long())
            loss += lovasz_softmax_flat(vprobas, vlabels).item()
        loss = loss / len(dl)
        return loss
    else:
        for X, y in dl:
            X, y = Variable(X).cuda(), Variable(y).cuda()
            output = model(X)
            loss += nn.NLLLoss()(output, y.long()).item()
        loss = loss / len(dl)
        return loss


'''
main_class: the class that you want to predict as one, must be a single value
'''


def binary_vector(x, main_class):
    length = len(x)
    binary = np.zeros(length)
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
        #print("label_binary")
        #print(label_binary)
        target_binary = binary_vector(target.flatten(), cls)
        #print("target_binary")
        #print(target_binary)
        intersection = np.sum(label_binary * target_binary)
        normalization = np.sum(label_binary + target_binary)
        score = ((2. * intersection + smooth).sum() /
                 (normalization + smooth).sum())
        scores.append(score)
        #print("--------------------")
    return scores

''' # old version 
def one_hot(x, classes):
    #print(x.shape)
    #print(x.dtype)
    length = len(x)
    x_one_hot = np.zeros((classes, length))
    x_one_hot[x, np.arange(length)] = 1
    return x_one_hot
'''

'''
label: ground truth label matrix or tensor
target: predicted label matrix or tensor
classes: number of classes in the label
'''

''' # old version
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

'''
label: predicted label matrix or tensor
target: predicted label matrix or tensor
classes: number of classes in the label
*WARNING: label and target must be of the same dimension. 
'''

''' # old version 
def jaccard_index(label, target):
    label_flat = label.flatten()
    target_flat = target.flatten()
    length = len(label_flat)
    assert (length == len(target_flat))

    union = (label_flat != target_flat).astype(int).sum() + length
    intersection = (label_flat == target_flat).astype(int).sum()

    return intersection / union
'''

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def isnan(x):
    return x != x

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n



def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, D, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)  # B * D * H * W, C = P, C
    labels = labels.view(-1)
    return probas, labels


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
        total_num += y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]
    return correct_num/total_num

def get_dice_score(dl, model):

    #batch_num = 0
    score = 0
    #img_sm = cv2.resize(img, (height, depth), interpolation=cv2.INTER_NEAREST)


    for X, y in dl:
        X = Variable(X).cuda()
        output = model(X).cpu()
        #print(output.shape)
        predicted = np.argmax(output.data.numpy(),axis=1)
        predicted.resize((predicted.shape[0]*predicted.shape[1],predicted.shape[2],predicted.shape[3]))
        #print(predicted.shape)

        predicted_origin = [0]*predicted.shape[0]
        for idx in range(len(predicted)):
            img = predicted[idx, :, :]
            img_sm = cv2.resize(img, (label_original[heart_index[dev_heart][1]].shape[2], label_original[heart_index[dev_heart][1]].shape[1]), interpolation=cv2.INTER_NEAREST)
            predicted_origin[idx] = img_sm

        predicted_origin = np.array(predicted_origin)
        predicted_origin2 = np.zeros((label_original[heart_index[dev_heart][1]].shape[0], label_original[heart_index[dev_heart][1]].shape[1], label_original[heart_index[dev_heart][1]].shape[2]))
        for idx in range(label_original[heart_index[dev_heart][1]].shape[1]):
            img = predicted_origin[:, idx, :]
            # shape 2 and shape 0 has confuse, need to check again
            img_sm = cv2.resize(img, (label_original[heart_index[dev_heart][1]].shape[2], label_original[heart_index[dev_heart][1]].shape[0]), interpolation=cv2.INTER_NEAREST)
            predicted_origin2[:, idx, :] = img_sm

        ground_truth = label_original[heart_index[dev_heart][1]].astype("int64")
        score = binary_dice_score(ground_truth,predicted_origin2.astype("int64"), [1,2])
    return score


def binary_jaccard_index(label, target, classes):
    scores = []
    assert (len(label.flatten()) == len(target.flatten()))
    for cls in classes:
        label_binary = binary_vector(label.flatten(), cls)
        target_binary = binary_vector(target.flatten(), cls)
        length = len(label_binary)

        union = (label_binary != target_binary).astype(int).sum() + length
        intersection = (label_binary == target_binary).astype(int).sum()

        scores.append(intersection / union)
    return scores


def get_jaccard_score(dl, model):

    #batch_num = 0
    score = 0
    #img_sm = cv2.resize(img, (height, depth), interpolation=cv2.INTER_NEAREST)


    for X, y in dl:
        X = Variable(X).cuda()
        output = model(X).cpu()
        #print(output.shape)
        predicted = np.argmax(output.data.numpy(),axis=1)
        predicted.resize((predicted.shape[0]*predicted.shape[1],predicted.shape[2],predicted.shape[3]))
        #print(predicted.shape)

        predicted_origin = [0]*predicted.shape[0]
        for idx in range(len(predicted)):
            img = predicted[idx, :, :]
            img_sm = cv2.resize(img, (label_original[heart_index[dev_heart][1]].shape[2], label_original[heart_index[dev_heart][1]].shape[1]), interpolation=cv2.INTER_NEAREST)
            predicted_origin[idx] = img_sm

        predicted_origin = np.array(predicted_origin)
        predicted_origin2 = np.zeros((label_original[heart_index[dev_heart][1]].shape[0], label_original[heart_index[dev_heart][1]].shape[1], label_original[heart_index[dev_heart][1]].shape[2]))
        for idx in range(label_original[heart_index[dev_heart][1]].shape[1]):
            img = predicted_origin[:, idx, :]
            # shape 2 and shape 0 has confuse, need to check again
            img_sm = cv2.resize(img, (label_original[heart_index[dev_heart][1]].shape[2], label_original[heart_index[dev_heart][1]].shape[0]), interpolation=cv2.INTER_NEAREST)
            predicted_origin2[:, idx, :] = img_sm

        ground_truth = label_original[heart_index[dev_heart][1]].astype("int64")
        score = jaccard_index(ground_truth,predicted_origin2.astype("int64"))

    return score

parser = argparse.ArgumentParser(description='UNET Implementation')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--slices-depth', type=int, default=8, metavar='N',
                    help='depth of each slides (default: 8)')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--figuresize1', type=int, default=200, metavar='N',
                    help='size1 that we use for the model')
parser.add_argument('--figuresize2', type=int, default=160, metavar='N',
                    help='size2 that we use for the model')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--channel-base', type=int, default=8, metavar='CB',
                    help='number of channel for first convolution (default: 8)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many epoches between logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--use-lovasz', action='store_true', default=False,
                    help='Whether use lovasz cross-entropy')
parser.add_argument('--test-model', type=str, default='', metavar='N',
                    help='If test-model has a name, do not do training, just testing on dev and train set')
parser.add_argument('--load-model', type=str, default=None, metavar='N',
                    help='If load-model has a name, use pretrained model')
args = parser.parse_args()

label_original = load_labels()
total_image, total_label, heart_index = get_data(args.slices_depth,args.figuresize1,args.figuresize2)

dev_heart = 7

timeStr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
os.mkdir(timeStr + "model")

while(dev_heart < 10):

    print("We are using heart "+str(heart_index[dev_heart][1]))
    train_loader = torch.utils.data.DataLoader(MyCustomDataset('Train', dev_heart), batch_size=args.batch_size, shuffle=True)
    if (dev_heart == 0):
        dev_loader = torch.utils.data.DataLoader(MyCustomDataset('Dev', dev_heart), batch_size=heart_index[0][0], shuffle=False)
    else:
        dev_loader = torch.utils.data.DataLoader(MyCustomDataset('Dev', dev_heart), batch_size=heart_index[dev_heart][0]-heart_index[dev_heart-1][0], shuffle=False)

    model = UnetGenerator_3d(1, 3, args.channel_base)
    if(args.load_model is not None):
        exist_dict = torch.load(args.load_model)
        total_dict = model.state_dict()
        for k, v in exist_dict.items():
            total_dict[k] = v
        model.load_state_dict(total_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    summary(model, input_size=(1, args.slices_depth, args.figuresize1, args.figuresize2))
    model.train()

    best_dice = [0.0,0.0]
    best_jaccard = 0
    optim = torch.optim.Adam(model.parameters(),lr=args.lr)

    os.mkdir(timeStr + "model/dice"+str(heart_index[dev_heart][1]))
    os.mkdir(timeStr + "model/jaccard"+str(heart_index[dev_heart][1]))

    #dev_acc = get_accuracy(dev_loader, model)
    #dev_dice = get_dice_score(dev_loader, model)
    for epoch in range(args.epochs):

        for batch_idx, (data, label) in enumerate(train_loader):

            data, target = data.to(device), label.to(device)
            if (args.use_lovasz):
                softmax_output_z = model(data)
                vprobas, vlabels = flatten_probas(softmax_output_z, target.long())
                loss = lovasz_softmax_flat(vprobas, vlabels)
            else:
                logsoftmax_output_z = model(data)
                loss = nn.NLLLoss()(logsoftmax_output_z, target.long())

            optim.zero_grad()
            loss.backward()
            optim.step()

        if (epoch + 1) % args.log_interval == 0:

            print("Epoch : "+str(epoch))
            model.eval()

            train_loss = get_loss(train_loader, model)
            print(train_loss)
            train_acc = get_accuracy(train_loader, model)
            print("Training accuracy : " + str(train_acc))
            dev_dice = get_dice_score(dev_loader, model)
            print("Dev dice score : " + str(dev_dice))
            #dev_jaccard = get_jaccard_score(dev_loader, model)
            #print("Dev jaccard score : " + str(dev_jaccard))
            dev_acc = get_accuracy(dev_loader, model)
            print("Dev accuracy : " + str(dev_acc))
            #if(train_acc < 0.01):
            #    print("Bad initialization")
            #    exit(0)
            if(args.save_model):
                if(dev_dice[0] > best_dice[0] or dev_dice[1] > best_dice[1]):
                    print("Best model found")
                    torch.save(model.state_dict(), timeStr + "model/dice"+str(heart_index[dev_heart][1])+"/" + str(epoch) + ":" + str(dev_dice) + ".pt")
                if(dev_dice[0] > best_dice[0]):
                    best_dice[0] = dev_dice[0]
                if(dev_dice[1] > best_dice[1]):
                    best_dice[1] = dev_dice[1]

            #if(args.save_model and (dev_jaccard > best_jaccard)):
            #    torch.save(model.state_dict(), timeStr + "model/jaccard"+str(heart_index[dev_heart][1])+"/" + str(epoch) + ":" + str(dev_jaccard) + ".pt")
            #    best_jaccard = dev_jaccard

            model.train()

    print("Done")

    dev_heart += 1