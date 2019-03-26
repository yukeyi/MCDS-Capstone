
from skimage import color
from skimage import io
import SimpleITK as sitk
import sys, os
import numpy as np


print( "Reading Dicom directory:", sys.argv[1] )
reader = sitk.ImageSeriesReader()

dicom_names = reader.GetGDCMSeriesFileNames( sys.argv[1] )
reader.SetFileNames(dicom_names)

image = reader.Execute()

size = image.GetSize()
print( "Image size:", size[0], size[1], size[2] )

print( "Writing image:", sys.argv[2] )

sitk.WriteImage( image, sys.argv[2]+".nii" )


IMG = sitk.GetArrayFromImage(image)
print(IMG.shape)


imgLabel = np.zeros(IMG.shape)
print(imgLabel.shape)
cnt = 0
files = []
for root, directories, filenames in os.walk(sys.argv[1]):
    for i in filenames:
        if i.endswith('.png'):
            files.append(i)

    for f in sorted(files):
        f = os.path.join(root,f)
        img = color.rgb2gray(io.imread(f))
        imgLabel[cnt,:,:]  = img
        cnt = cnt + 1

spacing = image.GetSpacing()
origin = image.GetOrigin()
direction = image.GetDirection()

imgLabel2 = sitk.GetImageFromArray(imgLabel)
imgLabel2.SetSpacing(spacing)
imgLabel2.SetDirection(direction)
imgLabel2.SetOrigin(origin)

# sitk.WriteImage(image, output_image+"nii")
sitk.WriteImage(imageLabel2, output_image+"_label.nii")