import SimpleITK as sitk

""" load fixed image and moving image (take fixed as standard, convert moving image into results)"""
fixedImage = sitk.ReadImage("DET0001101_SA10_ph10.dcm")
movingImage = sitk.ReadImage("DET0001101_SA11_ph10.dcm")

fixedArray = sitk.GetArrayFromImage(fixedImage)
fixedArray = fixedArray.reshape((fixedArray.shape[1], fixedArray.shape[2]))
fixedImage = sitk.GetImageFromArray(fixedArray)
movingArray = sitk.GetArrayFromImage(movingImage)
movingArray = movingArray.reshape((movingArray.shape[1], movingArray.shape[2]))
movingImage = sitk.GetImageFromArray(movingArray)

""" translation is just linear function """
elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fixedImage)
elastixImageFilter.SetMovingImage(movingImage)
#elastixImageFilter.SetParameterMap(parameterMap)

""" use non-linear parameter map """
p_map_vector = sitk.VectorOfParameterMap()

#translation = sitk.GetDefaultParameterMap('translation')
#translation['RequiredRatioOfValidSamples'] = ("0.1",)
#translation['NumberOfResolutions'] = ("20",)

paff = sitk.GetDefaultParameterMap('affine')
#paff['RequiredRatioOfValidSamples'] = ("0.1",)
#paff['NumberOfResolutions'] = ("20","20","20","20")

pbsp = sitk.GetDefaultParameterMap("bspline")
#pbsp['FinalGridSpacingInPhysicalUnits'] = ("32",)
#pbsp['GridSpacingSchedule'] = ("8","4","2","1",)
#pbsp['RequiredRatioOfValidSamples'] = ("0.1",)

#p_map_vector.append(translation)

# parameter tuning from dalao
paff['NumberOfSamplesForExactGradient'] = ['100000']
pbsp['NumberOfSamplesForExactGradient'] = ['100000']

paff['NumberOfSpatialSamples'] = ['5000']
pbsp['NumberOfSpatialSamples'] = ['5000']
paff['NumberOfHistogramBins'] = ['32', '32', '64', '128']
pbsp['NumberOfHistogramBins'] = ['32', '32', '64', '128']
paff['MaximumNumberOfIterations'] = ['1024']
pbsp['MaximumNumberOfIterations'] = ['1024']

paff['GridSpacingSchedule'] = ['8', '4', '2', '1.000000']
pbsp['GridSpacingSchedule'] = ['8', '4', '2', '1.000000']

pbsp['FinalGridSpacingInPhysicalUnits'] = ['16','16','4']


p_map_vector.append(paff)
p_map_vector.append(pbsp)

elastixImageFilter.SetParameterMap(p_map_vector)
elastixImageFilter.Execute()

""" get result and save it into nii file """
resultImage = elastixImageFilter.GetResultImage()
# sitk.PrintParameterMap(p_map_vector)
sitk.WriteImage(resultImage, "/pylon5/ac5616p/yukeyi/trans4.nii")

