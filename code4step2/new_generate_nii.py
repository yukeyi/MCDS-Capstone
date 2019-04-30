import os
import numpy as np
import pandas as pd
import xarray as xr
import pickle as pkl
import SimpleITK as sitk
import numpy as np
import glob
from data_process import DicomToXArray
from data_registration import RegHearts

# need to be the folder which contains all folders of hearts
LOAD_DIR = '/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/brain/sample'

def get_num_ph(heart_id):
    path = LOAD_DIR + heart_id + '/SA/SA1'
    return len(os.listdir(path))//2

# start of script
for heart_id in os.listdir(LOAD_DIR):
    num_ph = get_num_ph(heart_id)
    for t_slice in range(num_ph):
        dcxr = DicomToXArray(LOAD_DIR + heart_id  + '/SA')
        dcxr.generate_3D_nifti(t_slice=t_slice)