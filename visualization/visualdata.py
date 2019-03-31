import numpy as np
import os
import nibabel as nib
import skimage.io as io
from matplotlib import pyplot as plt


# show image
def print_image(img_arr):
    for i in range(100,101):
        io.imshow(img_arr[i,:,:])
        plt.show()


def load_training_data():

    print("loading training img 1")
    train1_img_arr = []
    for pat in range(10):
        img=nib.load('Training 1/training_axial_full_pat'+str(pat)+'.nii')
        train1_img_arr.append(np.swapaxes(np.array(img.get_fdata()),0,2))
    print("Done")

    print("loading training label 1")
    train1_label_arr = []
    for pat in range(10):
        label=nib.load('Label 1/training_axial_full_pat'+str(pat)+'-label.nii')
        train1_label_arr.append(np.swapaxes(np.array(label.get_fdata()),0,2))
        #print_image(train1_label_arr[pat])
    print("Done")


    print("loading training img 2")
    train2_img_arr = []
    for pat in range(10):
        img=nib.load('Training 2/training_axial_crop_pat'+str(pat)+'.nii')
        train2_img_arr.append(np.swapaxes(np.array(img.get_fdata()),0,2))
        #print_image(train2_img_arr[pat])
    print("Done")

    print("loading training label 2")
    train2_label_arr = []
    for pat in range(10):
        label=nib.load('Label 2/training_axial_crop_pat'+str(pat)+'-label.nii')
        train2_label_arr.append(np.swapaxes(np.array(label.get_fdata()),0,2))
        #print_image(train2_label_arr[pat])
    print("Done")


    print("loading training img 3")
    train3_img_arr = []
    for pat in range(10):
        img=nib.load('Training 3/training_sa_crop_pat'+str(pat)+'.nii')
        train3_img_arr.append(np.swapaxes(np.array(img.get_fdata()),0,2))
        #print_image(train3_img_arr[pat])
    print("Done")

    print("loading training label 3")
    train3_label_arr = []
    for pat in range(10):
        label=nib.load('Label 3/training_sa_crop_pat'+str(pat)+'-label.nii')
        train3_label_arr.append(np.swapaxes(np.array(label.get_fdata()),0,2))
        #print_image(train3_label_arr[pat])
    print("Done")

    return train1_img_arr, train1_label_arr, train2_img_arr, train2_label_arr, train3_img_arr, train3_label_arr


if __name__=="__main__":
    train1_img_arr, train1_label_arr, train2_img_arr, train2_label_arr, train3_img_arr, train3_label_arr = load_training_data()