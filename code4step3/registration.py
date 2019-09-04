import SimpleITK as sitk
import os


''' The function reads from output file generated by transformation and get the output coordinates '''
def get_output_points(filename='outputpoints.txt'):
    fr = open(filename, 'r')
    res = None
    for line in fr.readlines():
    	# Todo: make sure whether we should use OutputIndexMoving or OutputIndexFixed
    	# modify the following line, seems to fix the bug
        
        # line = line[line.index('OutputIndexFixed = ') + len('OutputIndexFixed = '):]
        # line = line[:line.index('\t')].lstrip('[').rstrip(']')
        
        line = line[line.index('OutputIndexMoving = ') + len('OutputIndexMoving = '):]
        line = line[:line.index('\n')].lstrip('[').rstrip(']')
        array = np.fromstring(line, dtype=int, sep=' ')
        if res is None:
            res = array.reshape(1, 3)
        else:
            res = np.concatenate((res, array.reshape(1, 3)), 0)
    return res


'''
This function generate points from fixedImage with given ranges.
hd must be either 'index' or 'point'
'''
def generate_points(x_range=[], y_range=[], z_range=[], filename='test.pts', hd='index'):
    assert (hd == 'index' or hd == 'point')
    res = np.zeros((len(x_range) * len(y_range) * len(z_range), 3))
    fr = open(filename, 'w')
    fr.write(hd + '\n' + str(len(x_range) * len(y_range) * len(z_range)))
    num_row = 0
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            for k in range(len(z_range)):
                x = np.array([x_range[i], y_range[j], z_range[k]]).astype('float')
                s = np.array2string(x, formatter={'float_kind': lambda x: "%.1f" % x}).lstrip('[').rstrip(']')
                fr.write('\n' + s)
                res[num_row] = x
                num_row += 1
    fr.close()
    return res.astype('int')


# path to execute: /pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2
def generate_patches():
	files = os.listdir("Brain2NIFI/")

	for i in range(0,len(files)):
		for j in range(i+1,len(files)):
			""" load fixed image and moving image (take fixed as standard, convert moving image into results)"""
			print(files[i])
			print(files[j])
			generate_patches_one_pair(files[i], files[j])


# function for generating which points from fixed image, write into file 'test.pts'
def sample_points_from_fixImage(filename=input_file_name):
	# Todo: modify this random generator
	original_points = generate_points([20,21],[10,11],[5,6],filename=input_file_name)
	return original_points


def generate_patches_one_pair(filename1, filename2):
	fixedImage = sitk.ReadImage("Brain2NIFI/"+filename1+"/norm.nii")
	movingImage = sitk.ReadImage("Brain2NIFI/"+filename2+"/norm.nii")

	""" translation is just linear function """
	parameterMap = sitk.GetDefaultParameterMap('translation')
	parameterMap = sitk.GetDefaultParameterMap("bspline")

	elastixImageFilter = sitk.ElastixImageFilter()
	elastixImageFilter.SetFixedImage(fixedImage)
	elastixImageFilter.SetMovingImage(movingImage)
	elastixImageFilter.SetParameterMap(parameterMap)

	""" use non-linear parameter map """
	parameterMapVector = sitk.VectorOfParameterMap()
	parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
	parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
	elastixImageFilter.SetParameterMap(parameterMapVector)
	elastixImageFilter.Execute()

	""" get result and save it into nii file """
	resultImage = elastixImageFilter.GetResultImage()
	#transformParameterMap = elastixImageFilter.GetTransformParameterMap()
	sitk.WriteImage(resultImage, "trans_"+files[i]+"_"+files[j]+".nii")	

	""" start generate patches """
	input_file_name = 'test.pts'
	original_points = sample_points_from_fixImage(input_file_name)

	transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    transformixImageFilter.SetFixedPointSetFileName(input_file_name)
    transformixImageFilter.SetMovingImage(movingImage)
    transformixImageFilter.Execute()

    # Todo: add loop checker to make sure all the transformed index is valid
    transformed_points = get_output_points()


