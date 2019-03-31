import SimpleITK as sitk

""" load fixed image and moving image (take fixed as standard, convert moving image into results)"""
fixedImage = sitk.ReadImage("training_axial_crop_pat0.nii")
movingImage = sitk.ReadImage("training_axial_crop_pat1.nii")

""" translation is just linear function """
#parameterMap = sitk.GetDefaultParameterMap('translation')
#parameterMap = sitk.GetDefaultParameterMap("bspline")

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fixedImage)
elastixImageFilter.SetMovingImage(movingImage)
#elastixImageFilter.SetParameterMap(parameterMap)

""" use non-linear parameter map """
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.Execute()

""" get result and save it into nii file """
resultImage = elastixImageFilter.GetResultImage()
#transformParameterMap = elastixImageFilter.GetTransformParameterMap()
sitk.WriteImage(resultImage, "trans.nii")
