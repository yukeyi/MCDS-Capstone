import glob
import pydicom
import numpy as np
import SimpleITK as sitk

dsList = []
imgList = []
slice_loc = []
filenames = glob.glob('data/DET0039401/DET0039401_SA*_ph1.dcm')
for fn in filenames:
    slice_loc.append(pydicom.dcmread(fn).SliceLocation)
    dsList.append(pydicom.dcmread(fn))

sortedFilenames = [x for _, x in sorted(zip(slice_loc, filenames))]
sortedListFn = [x for _, x in sorted(zip(slice_loc, dsList))]

for file in sortedFilenames:
    imgList.append(sitk.GetArrayFromImage(sitk.ReadImage(file)))

Img = np.stack(np.array(imgList),axis=0).squeeze(axis=1)
IMG = sitk.GetImageFromArray(Img)

sitk.WriteImage(IMG, '394_1.nii')