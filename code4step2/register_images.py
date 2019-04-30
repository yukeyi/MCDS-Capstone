import os
import shutil
import numpy as np
import SimpleITK as sitk
from data_registration import RegHearts

LOAD_DIR = '/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/brain/total/'

'''
Generator function to get one pair of fixed and moving image at a time
(fixed, moving) are viewed as without order.
(a, b) is the same as (b, a), so (b, a) won't be registered 
'''
def get_pair():
    patient_folders = os.listdir(LOAD_DIR)
    
    for i in range(len(patient_folders)):
        for j in range(i+1, len(patient_folders)):
            fixed = patient_folders[i]
            moving = patient_folders[j]
            yield fixed, moving
'''
Register two images and 
'''
def main():
    error = []
    for fixed_patient, moving_patient in get_pair():    
        reg = RegHearts(LOAD_DIR+fixed_patient+'/SA', LOAD_DIR+moving_patient+'/SA')
        reg.gen_param_map()
        try:
            reg.register_imgs()
        except:
            error += [fixed_patient + ',' + moving_patient]

        # make a new directory for storing transform parameter files wrt each moving patient
        try:
            os.mkdir(LOAD_DIR+fixed_patient+'/'+moving_patient)
        except OSError:
            pass

        origin = 'TransformParameters.0.txt'
        dest = LOAD_DIR+fixed_patient+'/'+moving_patient + '/TransformParameters.0.txt'
        shutil.copy(origin, dest)

        origin = 'TransformParameters.1.txt'
        dest = LOAD_DIR+fixed_patient+'/'+moving_patient + '/TransformParameters.1.txt'
        shutil.copy(origin, dest)
    
    with open('pairs_not_registered.csv', 'w') as f:
    for item in error:
        f.write("%s\n" % item)

if __name__ == '__main__':
    main()
