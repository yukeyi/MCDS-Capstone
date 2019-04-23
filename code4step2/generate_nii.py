import glob
import pydicom
import numpy as np
import SimpleITK as sitk
import os
from skimage import color
from skimage import io

SAVE_IMAGE_DIR = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/image"
SAVE_LABEL_DIR = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/label"
LOAD_DIR = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/DicomFiles"

def getLabel_sorted(shape,filenames):
    imgLabel = np.zeros(shape)
    cnt = 0
    for file in filenames:
        png_file = file.replace('.dcm', '.png')
        img = color.rgb2gray(io.imread(png_file))
        imgLabel[cnt,:,:]  = img
        cnt = cnt + 1

    imgLabel = np.where(imgLabel!=0,1,imgLabel)
    return imgLabel


dirs = ['DET0030901', 'DET0003001', 'DET0005301', 'DET0016101', 'DET0005201', 'DET0014101', 'DET0044601', 'DET0028301', 'DET0003101', 'DET0021701', 'DET0042501', 'DET0004801', 'DET0005801', 'DET0043501', 'DET0004201', 'DET0006301', 'DET0012801', 'DET0024501', 'DET0006101', 'DET0003901', 'EDES_Phases_TrainingDataSets.csv', 'DET0001801', 'DET0004001', 'DET0043701', 'DET0000201', 'DET0001201', 'DET0003301', 'DET0021501', 'DET0005001', 'DET0000801', 'DET0002901', 'DET0007101', 'DET0002801', 'DET0005101', 'DET0014201', 'DET0039301', 'DET0003201', 'DET0042601', 'DET0001301', 'DET0015201', 'DET0004101', 'DET0029001', 'DET0035501', 'DET0006001', 'DET0003801']
dim_mismatch_list = []
spacing_mismatch_list = []
slice_loc_list = []
#for load_heart_folder in os.listdir(LOAD_DIR):
for load_heart_folder in dirs:
    if load_heart_folder == 'EDES_Phases_TrainingDataSets.csv':
        continue
    if load_heart_folder == 'DET0003001':
        continue
    print("load_heart_folder=" + load_heart_folder)
        
    ph_num = 0
    heart_ph_path = LOAD_DIR + '/' + load_heart_folder + '/' + load_heart_folder + '_SA*_ph' + str(ph_num) + '.dcm'
    filenames = glob.glob(heart_ph_path)
    while len(filenames) != 0:
        dsList = []
        imgList = []
        slice_loc = []

        try:
            for fn in filenames:
                slice_loc.append(pydicom.dcmread(fn).SliceLocation)
                dsList.append(pydicom.dcmread(fn))
        except:
            print("slice_loc mismatch!")
            slice_loc_list += [load_heart_folder]
            ph_num+=1
            heart_ph_path = LOAD_DIR + '/' + load_heart_folder + '/' + load_heart_folder + '_SA*_ph' + str(ph_num) + '.dcm'
            filenames = glob.glob(heart_ph_path)
            continue

        sortedFilenames = [x for _,x in sorted(zip(slice_loc, filenames))]

        prev_spacing = None
        for file in sortedFilenames:
            tmp_img = sitk.ReadImage(file)  
            imgList.append(sitk.GetArrayFromImage(tmp_img))
            if prev_spacing is None:
                prev_spacing = tmp_img.GetSpacing()
                continue
            if prev_spacing != tmp_img.GetSpacing():
                print("Spacing mismatch for file " + file + "!")
                spacing_mismatch_list += [file]
            prev_spacing = tmp_img.GetSpacing()

        spacing = tmp_img.GetSpacing()
        origin = tmp_img.GetOrigin()
        direction = tmp_img.GetDirection()
 
        try:
            Img = np.stack(np.array(imgList),axis=0).squeeze(axis=1)
        except:
            print("Dimension mismatch: " + load_heart_folder + '_ph' + str(ph_num) + '.dcm!')
            dim_mismatch_list += [load_heart_folder + '_SA*_ph' + str(ph_num)]
            ph_num+=1
            heart_ph_path = LOAD_DIR + '/' + load_heart_folder + '/' + load_heart_folder + '_SA*_ph' + str(ph_num) + '.dcm'
            filenames = glob.glob(heart_ph_path)
            continue

        IMG = sitk.GetImageFromArray(Img)
        IMG.SetSpacing(spacing)
        IMG.SetDirection(direction)
        IMG.SetOrigin(origin)

        write_image_path = SAVE_IMAGE_DIR + '/' + load_heart_folder + '_ph' + str(ph_num) + '.nii'
        sitk.WriteImage(IMG, write_image_path)

        ImgLabel = getLabel_sorted(Img.shape, sortedFilenames)
        IMG_LABEL = sitk.GetImageFromArray(ImgLabel)
        IMG_LABEL.SetSpacing(spacing)
        IMG_LABEL.SetDirection(direction)
        IMG_LABEL.SetOrigin(origin)

        write_label_path = SAVE_LABEL_DIR + '/' + load_heart_folder + '_ph' + str(ph_num) + '.nii'
        sitk.WriteImage(IMG_LABEL, write_label_path)

        ph_num+=1
        heart_ph_path = LOAD_DIR + '/' + load_heart_folder + '/' + load_heart_folder + '_SA*_ph' + str(ph_num) + '.dcm'
        filenames = glob.glob(heart_ph_path)

print(dim_mismatch_list)
print(spacing_mismatch_list)
print(slice_loc_list)
