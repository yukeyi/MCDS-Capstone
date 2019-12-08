import os
import numpy as np
import SimpleITK as sitk
from torch.utils.data.dataset import Dataset

ROOT_DIR = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/"
#ROOT_DIR = "/Users/yukeyi/Desktop/"
trainFileName = "trainfiles.txt"
testFileName = "testfiles.txt"
os.chdir(ROOT_DIR)

class Image:
    def __init__(self, reg_dir):
        # self.image_list = []
        # self.aseg_list = []
        self.reg_dir = reg_dir
        self.parse_images()
        self.parse_registration()
        # self.make_xarray()

    def parse_images(self):
        images = self.reg_dir.split("-")
        assert (len(images) == 2)
        self.moving_image = sitk.ReadImage(ROOT_DIR + "Brain2NIFI/" + images[1] + "/norm.nii")

    def parse_registration(self):
        param0 = sitk.ReadParameterFile("BrainParameterMapsTuned/" + self.reg_dir + "/TransformParameters.0.txt")
        param1 = sitk.ReadParameterFile("BrainParameterMapsTuned/" + self.reg_dir + "/TransformParameters.1.txt")
        #param2 = sitk.ReadParameterFile("BrainParameterMapsTuned/" + self.reg_dir + "/TransformParameters.2.txt")
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.LogToConsoleOff()
        transformixImageFilter.AddTransformParameterMap(param0)
        transformixImageFilter.AddTransformParameterMap(param1)
        #transformixImageFilter.AddTransformParameterMap(param2)
        self.transformixImageFilter = transformixImageFilter

    def register_points(self, test_file='test.pts'):
        if os.path.exists('outputpoints.txt'):
            os.remove('outputpoints.txt')
        self.transformixImageFilter.SetFixedPointSetFileName(test_file)
        self.transformixImageFilter.SetMovingImage(self.moving_image)
        print(1)
        self.transformixImageFilter.Execute()
        print(2)

class BrainImageDataset(Dataset):
    def __init__(self, dirList, register_pairs, KNN, name_list_KNN):
        self.data = dirList
        self.register_pairs = register_pairs
        self.name_list_KNN = name_list_KNN
        self.KNN = KNN

    def __getitem__(self, index):
        fix = self.data[index]
        #if (os.path.exists("points_data_tuned_hard/" + "".join(fix) + "-" + "".join(self.register_pairs[fix]) + "-points.npy")):
        #    return (np.array([]), np.array([]), fix, self.register_pairs[fix])
        if (self.KNN != 0):
            if (fix not in self.name_list_KNN):
                return (np.array([]), np.array([]), fix, np.array([]))
        fixed_image_array = get_data(ROOT_DIR + "Brain2NIFI/" + fix + "/norm.nii")
        moving = self.register_pairs[fix]
        moving_image_array = get_data(ROOT_DIR + "Brain2NIFI/" + moving + "/norm.nii")
        return (fixed_image_array, moving_image_array, fix, moving)

    def __len__(self):
        return len(self.data)

def get_data(path):
    heart = sitk.ReadImage(path)
    heartArray = np.array([sitk.GetArrayFromImage(heart)]) / 256
    return heartArray

def load_Directory(is_train, register_pairs):
    #read file with train data path
    if is_train:
        file = open(ROOT_DIR + trainFileName, "r")
    #read file with test data path
    else:
        file = open(ROOT_DIR + testFileName, "r")

    dirnames = file.readlines()
    data_directory = [x.strip() for x in dirnames]

    dirList = []
    for i in data_directory:
        if i in register_pairs:
            dirList.append(i)
    return dirList

def load_pairs():
    register_pairs = {}
    for root, directories, filenames in os.walk(ROOT_DIR +"BrainParameterMapsTuned"):
        for pairname in directories:
            images = pairname.split("-")
            assert(len(images)==2)
            register_pairs[images[0]] = images[1]
    return register_pairs