import cv2
import numpy as np
import SimpleITK as sitk
import os
import copy

# change parameters here
base_image_path = "new_data/image/"
base_label_path = "new_data/label/"
base_dev_image_path = "new_data/dev_image/"
base_dev_label_path = "new_data/dev_label/"
depth = 240
height = 240
width = 240
slices = 8
use_resize_2 = True


def data_prepare(path, is_label_data):
    heart = sitk.ReadImage(path)
    heartArray = sitk.GetArrayFromImage(heart)
    if is_label_data:
        original_hearArray = copy.deepcopy(heartArray)
    # print(heartArray.shape)

    # resize the image
    img_stack_sm = np.zeros((len(heartArray), height, depth))
    #width = ((heartArray.shape[0]+slices-1)//slices)*slices
    width = 32

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
    img_stack_sm.resize((1,img_stack_sm.shape[0],img_stack_sm.shape[1],img_stack_sm.shape[2]))

    if is_label_data:
        return img_stack_sm.tolist(), original_hearArray
    else:
        return img_stack_sm.tolist()


def get_data(dim1, dim2):
    global depth
    global width
    global height

    height = dim1
    depth = dim2

    image = []
    label = []
    original_label = []
    #heart_index = []


    file_num = 0
    for file in os.listdir(base_label_path):
        if (file[0:3] != 'DET'):
            continue
        image_path = base_image_path + file[:-10] + ".nii"
        label_path = base_label_path + file

        #print(image_path)
        #print(label_path)
        if(file_num % 10 == 0):
            print(file_num)
        file_num += 1

        image += np.expand_dims(np.array(data_prepare(image_path, False)), axis=1).tolist()
        resized_label, true_label = data_prepare(label_path, True)
        label += resized_label
        original_label.append(true_label)
        if(file_num > 50):
            break
        #heart_index.append((len(label), int(image_path[-5])))

    print("Total num of training data : " + str(file_num))

    file_num = 0
    for file in os.listdir(base_dev_label_path):
        if (file[0:3] != 'DET'):
            continue
        image_path = base_dev_image_path + file[:-10] + ".nii"
        label_path = base_dev_label_path + file

        #print(image_path)
        #print(label_path)
        if(file_num % 10 == 0):
            print(file_num)
        file_num += 1

        image += np.expand_dims(np.array(data_prepare(image_path, False)), axis=1).tolist()
        resized_label, true_label = data_prepare(label_path, True)
        label += resized_label
        original_label.append(true_label)
        if(file_num > 10):
            break
        #heart_index.append((len(label), int(image_path[-5])))

    print("Total num of deveploment data : " + str(file_num))

    return np.array(image).astype(np.float32), np.array(label), original_label#, heart_index


def load_labels():
    labelshape = []
    for file in os.listdir(base_label_path):
        if (file[0:3] != 'DET'):
            continue
        print(file)
        label = sitk.ReadImage(base_label_path + file)
        labelArray = sitk.GetArrayFromImage(label)
        labelshape.append(labelArray)
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

    image, label, index = get_data(200,160)
    a = 1