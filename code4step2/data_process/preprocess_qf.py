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
SAVE_DIR = "new_data/"


# helper function for sorting filenames
def sortFilenames(dic, key):
    files = dic[key]
    order = []
    for f in files:
        f = f[:-4]
        f = int(f.split('ph')[-1])
        order.append(f)
    sorted_files = []
    root = files[0].split('ph')[0] + 'ph'
    for num in sorted(order):
        f = root + str(num) + '.dcm'
        sorted_files.append(f)
    return sorted_files


# create a dictionary of series
# key: series name, ex: SAxx or LAxx
# value: the sorted filenames for that series
def createDictionary(directory):
    series_dict = {}
    for root, directories, filenames in os.walk(directory):
        for i in filenames:
            if i.endswith('.dcm'):
                splitted = i.split("_")
                series_name = splitted[0] + "_" + splitted[1]
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
    name = ''.join(name.split('_')[0:2])
    return name


def getLabel_sorted(IMG, files):
    imgLabel = np.zeros(IMG.shape)
    cnt = 0
    for f in files:
        f = f[:-4] + '.png'
        img = color.rgb2gray(io.imread(f))
        imgLabel[cnt, :, :] = img
        cnt = cnt + 1

    imgLabel = np.where(imgLabel != 0, 1, imgLabel)
    return imgLabel


def processDirectory_sorted(dir):
    print("Reading Dicom directory:", dir)
    reader = sitk.ImageSeriesReader()
    # series_ids = reader.GetGDCMSeriesIDs(dir)
    series_dict = createDictionary(dir)

    # for each series
    for key, value in series_dict.items():
        dicom_names = series_dict[key]
        dicom_names = tuple(dicom_names)
        reader.SetFileNames(dicom_names)

        image = reader.Execute()

        name = getName(dicom_names[0])
        sitk.WriteImage(image, SAVE_DIR + name + ".nii")
        IMG = sitk.GetArrayFromImage(image)

        # set label.nii with image metadata
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        imgLabel = getLabel_sorted(IMG, dicom_names)
        imgLabel2 = sitk.GetImageFromArray(imgLabel)
        imgLabel2.SetSpacing(spacing)
        imgLabel2.SetDirection(direction)
        imgLabel2.SetOrigin(origin)
        sitk.WriteImage(imgLabel2, SAVE_DIR + name + "_label.nii")


if __name__ == '__main__':
    # create the data directory
    try:
        os.mkdir(SAVE_DIR)
    except OSError:
        print("Creation of the directory %s failed" % SAVE_DIR)
    else:
        print("Successfully created the directory %s" % SAVE_DIR)
    # loop through all the subdirectories
    count = 0
    for root, directories, filenames in os.walk(sys.argv[1]):
        for d in directories:
            processDirectory_sorted(os.path.join(root, d))
            count += 1
            if count % 10 == 0:
                print("Processed 10 directories")

# save the correspoding labels
# def getLabels (IMG,files):
#     imgLabel = np.zeros(IMG.shape)
#     cnt = 0
#     # sort the label files
#     order = []
#     for f in files:
#         f = f[:-4]
#         f = int(f.split('ph')[-1])
#         order.append(f)

#     root = files[0].split('ph')[0] + 'ph'
#     for num in sorted(order):
#         f = root+str(num)+'.png'
#         print(f)
#         # convert png to grayscale
#         img = color.rgb2gray(io.imread(f))
#         imgLabel[cnt,:,:]  = img
#         cnt = cnt + 1

#     imgLabel = np.where(imgLabel!=0,1,imgLabel)
#     return imgLabel

# def processDirectory(dir):
#     print( "Reading Dicom directory:", dir )
#     reader = sitk.ImageSeriesReader()
#     series_ids = reader.GetGDCMSeriesIDs(dir)
#     # for each series
#     for i in series_ids:
#         # read the image series
#         dicom_names = reader.GetGDCMSeriesFileNames( dir,i )
#         reader.SetFileNames(dicom_names)


#         image = reader.Execute()

#         name = getName(dicom_names[0])
#         sitk.WriteImage(image, SAVE_DIR+ name+".nii" )
#         IMG = sitk.GetArrayFromImage(image)

#         # set label.nii with image metadata
#         spacing = image.GetSpacing()
#         origin = image.GetOrigin()
#         direction = image.GetDirection()
#         imgLabel = getLabel_new(IMG,dicom_names)
#         imgLabel2 = sitk.GetImageFromArray(imgLabel)
#         imgLabel2.SetSpacing(spacing)
#         imgLabel2.SetDirection(direction)
#         imgLabel2.SetOrigin(origin)
#         sitk.WriteImage(imgLabel2, SAVE_DIR+name+"_label.nii")