from skimage import color
from skimage import io
import SimpleITK as sitk
import sys, os
import numpy as np

"""
    Usage: python preprocess.py DIR/TO/DATA
    DIR/TO/DATA: /pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/
"""

# directory where the data will be saved
SAVE_DIR = "save_test_series/"
data_dir = "30/image_folders/"


# helper function for sorting filenames
def sortFilenames(dic, key):
    files = dic[key]
    order = []
    for f in files:
        f = f[:-4]
        f = int(f.split('_')[-2][2:])
        order.append(f)
    sorted_files = []
    root = files[0].split('_')[0] + '_SA'
    bottom = '_' + files[0].split('_')[-1]
    for num in sorted(order):
        f = root + str(num) + bottom
        sorted_files.append(f)
    return sorted_files


# create a dictionary of series
# key: series name, ex: SAxx or LAxx
# value: the sorted filenames for that series
def createDictionary(directory):
    series_dict = {}
    for root, directories, filenames in os.walk(directory):
        for i in filenames:
            if i.endswith('.dcm') and i[11] == 'S':
                splitted = i.split("_")
                series_name = splitted[0] + "_" + splitted[2][:-4]
                i = os.path.join(root, i)
                if series_name in series_dict:
                    series_dict[series_name].append(i)
                else:
                    series_dict[series_name] = []
                    series_dict[series_name].append(i)
    for key, value in series_dict.items():
        sorted_value = sortFilenames(series_dict, key)
        series_dict[key] = sorted_value
    return series_dict


def getName(dirname):
    name = dirname.split('/')
    name = name[-1]
    name = ''.join([name.split('_')[0], name.split('_')[2]])
    return name[:-4]


'''
shape: the shape if the image (without label) array shape
filenames: sorted filenames of .png files
'''

def getLabel_sorted(shape,filenames):
    imgLabel = np.zeros(shape)
    cnt = 0
    for file in filenames:
        png_file = file.replace('.dcm', '.png')
        png_file = png_file.replace('image_folders','label_folders')
        img = color.rgb2gray(io.imread(png_file))
        imgLabel[cnt,:,:]  = img
        cnt = cnt + 1

    imgLabel = np.where(imgLabel!=0,1,imgLabel)
    return imgLabel

def get_label(IMG,sortedFilenames, shape):

    spacing = IMG.GetSpacing()
    origin = IMG.GetOrigin()
    direction = IMG.GetDirection()

    ImgLabel = getLabel_sorted(shape, sortedFilenames)
    IMG_LABEL = sitk.GetImageFromArray(ImgLabel)
    IMG_LABEL.SetSpacing(spacing)
    IMG_LABEL.SetDirection(direction)
    IMG_LABEL.SetOrigin(origin)
    return IMG_LABEL

def process(file_path):

    global error_list

    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_dir + file_path)
    nb_series = len(series_IDs)
    if(nb_series != 1):
        print("Find a bug, " + file_path + " number of series is " + str(nb_series))
        error_list.append(file_path)
        return
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_dir + file_path, series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    print(image3D.GetSize())
    image3D.GetSpacing()

    #label_name = [item[:-3]+"png" for item in series_file_names]
    label = get_label(image3D, series_file_names, sitk.GetArrayFromImage(image3D).shape)

    sitk.WriteImage(image3D, SAVE_DIR + file_path + '.nii')
    sitk.WriteImage(label, SAVE_DIR + file_path + '_label.nii')

#process("auto_recog")
error_list = []
#os.mkdir(SAVE_DIR)
for root, directories, filenames in os.walk(data_dir):
    for d in directories:
        print("----------------- " + d + " -----------------")
        process(d)

print(error_list)
