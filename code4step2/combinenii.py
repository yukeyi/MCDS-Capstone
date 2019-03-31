from skimage import color
from skimage import io
import SimpleITK as sitk
import sys, os
import numpy as np

"""
    Usage: python preprocess.py DIR/TO/DATA
    DIR/TO/DATA: /pylon5/ac5616p/yukeyi/MCDS-Capstone/data/new_data/
"""

if __name__ == '__main__':
    #os.mkdir("new_S_image")
    #os.mkdir("new_S_label")
    M = {}
    for root, directories, filenames in os.walk(sys.argv[1]):
        for item in filenames:
            if(item[:3] != 'DET'):
                continue
            if(item[:10] in M):
                M[item[:10]].append(item)
            else:
                M[item[:10]] = [item]
    print(len(M))

    for index in M:
        print(index)
        filenames = M[index]
        if(len(filenames) % 2 == 1):
            print(" fuck1 " + index)
        Lnum = 0
        for i in filenames:
            if(i[10] == 'L'):
                Lnum += 1
        if(Lnum % 2 == 1):
            print(" fuck2 " + index)
        Lnum = Lnum // 2
        Snum = len(filenames)//2 - Lnum

        Limage = []
        Llabel = []
        Simage = []
        Slabel = []
        for i in range(Snum):
            Simage.append(sitk.GetArrayFromImage(sitk.ReadImage(index + "SA" + str(i+1) + ".nii")))
            Slabel.append(sitk.GetArrayFromImage(sitk.ReadImage(index + "SA" + str(i+1) + "_label.nii")))

        try:
            Simage = np.vstack(Simage)
            Slabel = np.vstack(Slabel)
        except:
            print("---- bad data ----")
            for i in Simage:
                print(i.shape)
            continue

        if(Simage.shape != Slabel.shape):
            print(" shape not equal")
        else:
            print(Simage.shape)
        sitk.WriteImage(sitk.GetImageFromArray(Simage), 'new_S_image/' + index + "SA.nii")
        sitk.WriteImage(sitk.GetImageFromArray(Slabel), 'new_S_label/' + index + "SA_label.nii")