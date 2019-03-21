import cv2
import numpy as np
import SimpleITK as sitk
import os

# change parameters here
base_image_path = "../Training2/"
base_label_path = "../Label2/"
depth = 240
height = 240
width = 240
slices = 8
use_resize_2 = True


def data_prepare(path, is_label_data):
    heart = sitk.ReadImage(path)
    heartArray = sitk.GetArrayFromImage(heart)
    # print(heartArray.shape)

    # resize the image
    img_stack_sm = np.zeros((len(heartArray), height, depth))
    width = ((heartArray.shape[0]+slices-1)//slices)*slices

    for idx in range(len(heartArray)):
        img = heartArray[idx, :, :]
        if is_label_data:
            img_sm = cv2.resize(img, (depth, height), interpolation=cv2.INTER_NEAREST)
        else:
            img_sm = cv2.resize(img, (depth, height), interpolation=cv2.INTER_CUBIC)
        img_stack_sm[idx, :, :] = img_sm

    if (use_resize_2):
        img_stack_sm2 = np.zeros((width, height, depth))

        for idx in range(height):
            img = img_stack_sm[:, idx, :]
            if is_label_data:
                img_sm = cv2.resize(img, (depth, width), interpolation=cv2.INTER_NEAREST)

            else:
                img_sm = cv2.resize(img, (depth, width), interpolation=cv2.INTER_CUBIC)
            img_stack_sm2[:, idx, :] = img_sm
        img_stack_sm = img_stack_sm2

    # print(img_stack_sm.shape)
    img_stack_sm.resize((img_stack_sm.shape[0]//slices,slices,img_stack_sm.shape[1],img_stack_sm.shape[2]))
    return img_stack_sm.tolist()


def get_data(mini_dim, dim1, dim2):
    global depth
    global width
    global height
    global slices

    slices = mini_dim
    height = dim1
    depth = dim2

    image = []
    label = []
    heart_index = []

    for file in os.listdir(base_image_path):
        if (file[0] == '.'):
            continue
        image_path = base_image_path + file
        label_path = base_label_path + file[:-4] + "-label.nii"
        print(image_path)
        print(label_path)

        image += np.expand_dims(np.array(data_prepare(image_path, False)), axis=1).tolist()
        label += data_prepare(label_path, True)
        heart_index.append((len(label), int(image_path[-5])))

    return np.array(image).astype(np.float32), np.array(label), heart_index


def load_labels():
    labelshape = [0] * 10
    for file in os.listdir(base_label_path):
        if (file[0] == '.'):
            continue
        print(file)
        label = sitk.ReadImage(base_label_path + file)
        labelArray = sitk.GetArrayFromImage(label)
        labelshape[int(file[-11])] = labelArray
    return labelshape


if __name__ == "__main__":
    label_original = load_labels()
    dim2 = 0
    dim3 = 0
    maxdim2 = 0
    maxdim3 = 0
    for i in range(10):
        dim2 += len(label_original[i][0])
        dim3 += len(label_original[i][0][0])
        if len(label_original[i][0]) > maxdim2:
            maxdim2 = len(label_original[i][0])
        if len(label_original[i][0][0]) > maxdim3:
            maxdim3 = len(label_original[i][0][0])
    dim2 /= 10
    dim3 /= 10

    image, label, index = get_data(8,200,160)
    a = 1