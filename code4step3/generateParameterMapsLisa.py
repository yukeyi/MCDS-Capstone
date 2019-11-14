import SimpleITK as sitk
import os
import numpy as np
import shutil
import logging
import time

ROOT_DIR = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/"

def generateParameterMaps(fixed, moving):
    fixedImage = sitk.ReadImage(ROOT_DIR + "Brain2NIFI/" + fixed + "/norm.nii")
    movingImage = sitk.ReadImage(ROOT_DIR + "Brain2NIFI/" + moving + "/norm.nii")

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)

    p_map_vector = sitk.VectorOfParameterMap()
    paff = sitk.GetDefaultParameterMap("rigid")
    pbsp = sitk.GetDefaultParameterMap("bspline")
    paff['DefaultPixelValue'] = ['0']
    pbsp['DefaultPixelValue'] = ['0']
    paff['AutomaticTransformInitialization'] = ['true']
    paff['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
    paff['NumberOfSamplesForExactGradient'] = ['100000']
    pbsp['NumberOfSamplesForExactGradient'] = ['100000']
    paff['NumberOfSpatialSamples'] = ['5000']
    pbsp['NumberOfSpatialSamples'] = ['5000']
    paff['NumberOfHistogramBins'] = ['32', '64', '256', '512']
    paff['MaximumNumberOfIterations'] = ['512']
    pbsp['MaximumNumberOfIterations'] = ['512']
    pbsp['NumberOfResolutions'] = ['4']
    paff['GridSpacingSchedule'] = ['6', '4', '2', '1.000000']
    pbsp['GridSpacingSchedule'] = ['6', '4', '2', '1.0']
    pbsp['FinalGridSpacingInPhysicalUnits'] = ['4', '4', '4']
    pbsp['FinalBSplineInterpolationOrder'] = ['2']

    p_map_vector.append(paff)
    p_map_vector.append(pbsp)

    elastixImageFilter.SetParameterMap(p_map_vector)
    elastixImageFilter.Execute()

    resultImage = elastixImageFilter.GetResultImage()

    if os.path.exists(ROOT_DIR + "BrainParameterMapsTuned/") == False:
        os.mkdir(ROOT_DIR + "BrainParameterMapsTuned/")

    newFolderName = ROOT_DIR + "BrainParameterMapsTuned/" + fixed + "-" + moving
    if (os.path.exists(newFolderName) == False):
        os.mkdir(newFolderName)

    sitk.WriteImage(resultImage, newFolderName + "/result.nii")
    shutil.move("TransformParameters.0.txt", newFolderName + "/TransformParameters.0.txt")
    shutil.move("TransformParameters.1.txt", newFolderName + "/TransformParameters.1.txt")

    outfile = open("processedfiles.txt", 'a')
    outfile.write(fixed + "\n" + moving + "\n")
    outfile.close()

if __name__ == '__main__':
    files = [line.rstrip('\n') for line in open("files1.txt")]
    processed = []
    if os.path.exists("processedfiles.txt"):
        processed = [line.rstrip('\n') for line in open("processedfiles.txt")]

    files = [file for file in files if file not in processed]

    print("Processing %s files..."%len(files))

    # remove TransformParameters files if exists
    if os.path.exists("TransformParameters.0.txt") == True:
        os.remove("TransformParameters.0.txt")
    if os.path.exists("TransformParameters.1.txt") == True:
        os.remove("TransformParameters.1.txt")

    for i in range(0, len(files), 2):
        print("Processing %s and %s"%(files[i], files[i+1]))
        generateParameterMaps(files[i], files[i+1])

