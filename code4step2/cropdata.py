import cv2
import numpy as np
import SimpleITK as sitk
import os
import torch
import copy

# change parameters here
base_image_path = "new_data/image/"
base_label_path = "new_data/label/"
base_dev_image_path = "new_data/dev_image/"
base_dev_label_path = "new_data/dev_label/"


def get_cropped_image_array(label_array, image_array):
    left = -1
    right = -1
    up = -1
    down = -1
    for i in range(label_array.shape[1]):
        j = label_array.shape[1] - i - 1
        # haven't encountered any labels
        if left == -1:
            if (label_array[:, i, :]==1).sum() != 0:
                left = i
        if right == -1:
            if (label_array[:, j, :]==1).sum() != 0:
                right = j
    for i in range(label_array.shape[2]):
        j = label_array.shape[2] - i - 1
        # haven't encountered any labels
        if up == -1:
            if (label_array[:, :, i]==1).sum() != 0:
                up = i
        if down == -1:
            if (label_array[:, :, j]==1).sum() != 0:
                down = j
    left = 0 if left - 20 < 0 else left - 20
    right = label_array.shape[1] - 1 if right + 20 > label_array.shape[1] - 1 else right + 20
    up = 0 if up - 20 < 0 else up - 20
    down = label_array.shape[2] - 1 if down + 20 > label_array.shape[2] - 1 else down + 20


    return (label_array[:, left:(right + 1), up:(down + 1)], image_array[:, left:(right + 1), up:(down + 1)], left, right, up, down)


for file in os.listdir(base_dev_label_path):
    if (file[0:3] != 'DET'):
        continue
    image_path = base_dev_image_path + file[:-10] + ".nii"
    label_path = base_dev_label_path + file
    crop_image_path = "crop" + image_path[3:]
    crop_label_path = "crop" + label_path[3:]
    heart = sitk.ReadImage(image_path)
    heartArray = sitk.GetArrayFromImage(heart)
    label = sitk.ReadImage(label_path)
    labelArray = sitk.GetArrayFromImage(label)
    labelArray, heartArray, _, _, _, _ = get_cropped_image_array(labelArray, heartArray)

    sitk.WriteImage(sitk.GetImageFromArray(labelArray), crop_label_path)
    sitk.WriteImage(sitk.GetImageFromArray(heartArray), crop_image_path)


for file in os.listdir(base_label_path):
    if (file[0:3] != 'DET'):
        continue
    image_path = base_image_path + file[:-10] + ".nii"
    label_path = base_label_path + file
    crop_image_path = "crop" + image_path[3:]
    crop_label_path = "crop" + label_path[3:]
    heart = sitk.ReadImage(image_path)
    heartArray = sitk.GetArrayFromImage(heart)
    label = sitk.ReadImage(label_path)
    labelArray = sitk.GetArrayFromImage(label)
    labelArray, heartArray, _, _, _, _ = get_cropped_image_array(labelArray, heartArray)

    sitk.WriteImage(sitk.GetImageFromArray(labelArray), crop_label_path)
    sitk.WriteImage(sitk.GetImageFromArray(heartArray), crop_image_path)