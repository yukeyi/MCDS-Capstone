import SimpleITK as sitk
import os
import random
import numpy as np
import torch

''' The function reads from output file generated by transformation and get the output coordinates '''
def get_output_points(filename='outputpoints.txt'):
	fr = open(filename, 'r')
	res = None
	for line in fr.readlines():
		# modify the following line, seems to fix the bug

		line = line[line.index('OutputIndexFixed = ') + len('OutputIndexFixed = '):]
		line = line[:line.index('\t')].lstrip('[').rstrip(']')

		# line = line[line.index('OutputIndexMoving = ') + len('OutputIndexMoving = '):]
		# line = line[:line.index('\n')].lstrip('[').rstrip(']')
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
def sample_points_from_fixImage(filename):
	# Todo: modify this random generator
	original_points = generate_points([127,129],[127,129],[127,129],filename)
	return original_points


def generate_patches_one_pair(filename1, filename2):
	fixedImage = sitk.ReadImage("Brain2NIFI/"+filename1+"/norm.nii")
	movingImage = sitk.ReadImage("Brain2NIFI/"+filename2+"/norm.nii")

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
	parameterMapVector.append(sitk.GetDefaultParameterMap("translation"))
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
	# Todo: save for furture use


def get_valid_fixImage_point(image, patch_size = [24,24,24], num_points = 10, zero_ratio = 0.3):
	res = []
	data = sitk.GetArrayFromImage(image)
	sz = data.shape
	boundary = [] #[0begin, 0end, 1begin, 1end, 2begin, 2end]
	for i in range(sz[0]):
		if(a[i].sum() == 0):
			continue
		else:
			print(i)
			boundary.append(i)
			break
	for i in range(sz[0]-1,-1,-1):
		if(a[i].sum() == 0):
			continue
		else:
			print(i)
			boundary.append(i)
			break
	for i in range(sz[1]):
		if(a[:,i,:].sum() == 0):
			continue
		else:
			print(i)
			boundary.append(i)
			break
	for i in range(sz[1]-1,-1,-1):
		if(a[:,i,:].sum() == 0):
			continue
		else:
			print(i)
			boundary.append(i)
			break
	for i in range(sz[2]):
		if(a[:,:,i].sum() == 0):
			continue
		else:
			print(i)
			boundary.append(i)
			break
	for i in range(sz[2]-1,-1,-1):
		if(a[:,:,i].sum() == 0):
			continue
		else:
			print(i)
			boundary.append(i)
			break
	while(len(res) < num_points):
		chosen = [random.randint(boundary[0],boundary[1]), random.randint(boundary[2],boundary[3]), random.randint(boundary[4],boundary[5])]
		patch = data[chosen[0]-patch_size[0]//2:chosen[0]+patch_size[0]//2,chosen[1]-patch_size[1]//2:chosen[1]+patch_size[1]//2,chosen[2]-patch_size[2]//2:chosen[2]+patch_size[2]//2]
		if(patch.sum()/(patch.shape[0]*patch.shape[1]*patch.shape[2]) >= zero_ratio):
			res.append(chosen)
		else:
			print("Done")
	return res


def generate_outputpoints_from_transformation_file(path, movingImage, input_file_name):
	param0 = sitk.ReadParameterFile(path+"TransformParameters.0.txt")
	param1 = sitk.ReadParameterFile(path+"TransformParameters.1.txt")
	param2 = sitk.ReadParameterFile(path+"TransformParameters.2.txt")
	transformixImageFilter = sitk.TransformixImageFilter()
	transformixImageFilter.AddTransformParameterMap(param0)
	transformixImageFilter.AddTransformParameterMap(param1)
	transformixImageFilter.AddTransformParameterMap(param2)
	transformixImageFilter.SetFixedPointSetFileName(input_file_name)
	transformixImageFilter.SetMovingImage(movingImage)
	transformixImageFilter.Execute()


if __name__ == '__main__':
	files = os.listdir("Brain2NIFI/")
	i = 0
	j = 1
	""" load fixed image and moving image (take fixed as standard, convert moving image into results)"""
	print(files[i])
	print(files[j])
	generate_patches_one_pair(files[i], files[j])