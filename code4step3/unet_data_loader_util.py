import SimpleITK as sitk
import numpy as np
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/"
trainFileName = "trainfiles.txt"
testFileName = "testfiles.txt"

def value2label(x, label_map):
    return label_map[x]

def get_label_map():
    label_list = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13,
                  14, 15, 16, 17, 18, 24, 26, 28, 30,
                  31, 41, 42, 43, 44, 46, 47, 49, 50,
                  51, 52, 53, 54, 58, 60, 62, 63, 72,
                  77, 80, 85, 251, 252, 253, 254, 255]
    label_map = {}
    for i in range(len(label_list)):
        label_map[label_list[i]] = i
    return label_map, label_list

def get_data(path):
    heart = sitk.ReadImage(path)
    heartArray = np.array([sitk.GetArrayFromImage(heart)])
    return heartArray

def load_Directory(is_train, args):
    #read file with train data path
    if is_train:
        file = open(ROOT_DIR + trainFileName, "r")
    #read file with test data path
    else:
        file = open(ROOT_DIR + testFileName, "r")

    dirnames = file.readlines()
    data_directory = [x.strip() for x in dirnames]
    if is_train:
        data_directory = data_directory[:args.num_train]
    else:
        data_directory = data_directory[:args.num_dev]
    return data_directory

class BrainImageDataset(Dataset):
    def __init__(self, dirList, label_list):
        self.label_list = label_list
        self.data = dirList

    def __getitem__(self, index):

        name = self.data[index]
        try:
            image = get_data(ROOT_DIR + "Brain2NIFI/" + name + "/norm.nii").astype("float32") / 256
            target = get_data(ROOT_DIR + "Brain2NIFI/" + name + "/aseg.nii")
        except:
            return (np.array([]), np.array([]))
        # two ways to convert from real number value to label
        #start_time = time.time()
        #temp = target.reshape(-1)
        #label = np.array(list(map(value2label,temp)))
        #label = label.reshape((1,256,256,256))
        #mid_time = time.time()
        label = np.zeros(target.shape)
        for i in range(len(self.label_list)):
            label += ((target == self.label_list[i])*i)
        label = label[0]
        #end_time = time.time()
        #print(mid_time-start_time)
        #print(end_time-start_time)
        '''
        # for get label's index
        label_list = []
        for i in range(256):
            if(((label * 256) == i).sum()>0):
                label_list.append(i)
        print(len(label_list))
        print(label_list)
        '''
        #image = image[:, :256, :32, :32]
        #label = label[:256, :32, :32]
        return (image, label)

    def __len__(self):
        return len(self.data)