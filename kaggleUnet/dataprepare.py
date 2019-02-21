import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

#change parameters here
base_image_path = "../Training2/"
base_label_path = "../Label2/"
depth = 240
height = 240
width = 240
use_resize_2 = False

def data_prepare(path, is_label_data):

    heart = sitk.ReadImage(path)
    heartArray = sitk.GetArrayFromImage(heart)
    #print(heartArray.shape)

    # resize the image
    img_stack_sm = np.zeros((len(heartArray), height, depth))

    for idx in range(len(heartArray)):
        img = heartArray[idx, :, :]
        if is_label_data:
            img_sm = cv2.resize(img, (height, depth), interpolation=cv2.INTER_NEAREST)
        else:
            img_sm = cv2.resize(img, (height, depth), interpolation=cv2.INTER_CUBIC)
        img_stack_sm[idx, :, :] = img_sm

    if(use_resize_2):
        img_stack_sm2 = np.zeros((width, height, depth))

        for idx in range(height):
            img = img_stack_sm[:, idx, :]
            if is_label_data:
                img_sm = cv2.resize(img, (width, depth), interpolation=cv2.INTER_NEAREST)

            else:
                img_sm = cv2.resize(img, (width, depth), interpolation=cv2.INTER_CUBIC)
            img_stack_sm2[:, idx, :] = img_sm
        img_stack_sm = img_stack_sm2

    #print(img_stack_sm.shape)

    return img_stack_sm.tolist()

def get_data():
    image = []
    label = []
    for file in os.listdir(base_image_path):
        if(file[0] == '.'):
            continue
        image_path = base_image_path+file
        label_path = base_label_path+file[:-4]+"-label.nii"
        print(image_path)
        print(label_path)

        image += (data_prepare(image_path, False))
        label += (data_prepare(label_path, True))
    
    return np.expand_dims(np.array(image), axis=1).astype(np.float32), np.array(label)

if __name__=="__main__":
    image, label = get_data()
