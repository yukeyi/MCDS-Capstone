
from skimage import color
from skimage import io
import SimpleITK as sitk
import sys, os
import numpy as np
import dicom

# save the correspoding labels
def getLabels (IMG,files):
    imgLabel = np.zeros(IMG.shape)
    cnt = 0
    # sort the label files
    order = []
    for f in files:
        f = f[:-4]
        f = int(f.split('ph')[-1])
        order.append(f)

    root = files[0].split('ph')[0] + 'ph'
    for num in sorted(order):
        f = root+str(num)+'.png'
        img = color.rgb2gray(io.imread(f))
        imgLabel[cnt,:,:]  = img
        cnt = cnt + 1

    imgLabel = np.where(imgLabel!=0,1,imgLabel)
    return imgLabel

def getName(dirname):
    name = dirname.split('/')
    name = name[-1]
    name = ''.join(name.split('_')[0:2])
    return name

def processDirectory(dir):
    print( "Reading Dicom directory:", dir )
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dir)
    for i in series_ids:
        dicom_names = reader.GetGDCMSeriesFileNames( dir,i )
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        name = getName(dicom_names[0])
        sitk.WriteImage(image, name+".nii" )
        IMG = sitk.GetArrayFromImage(image)

        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        imgLabel = getLabels(IMG,dicom_names)
        imgLabel2 = sitk.GetImageFromArray(imgLabel)
        imgLabel2.SetSpacing(spacing)
        imgLabel2.SetDirection(direction)
        imgLabel2.SetOrigin(origin)

        sitk.WriteImage(imgLabel2, name+"_label.nii")


for root, directories, filenames in os.walk(sys.argv[1]):
    for d in directories:
        processDirectory(os.path.join(root,d))
    