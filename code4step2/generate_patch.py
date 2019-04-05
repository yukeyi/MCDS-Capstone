import numpy as np
import SimpleITK as sitk

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


'''
The function reads from output file generated by transformation and get the output coordinates
'''


def get_output_points(filename='outputpoints.txt'):
    fr = open(filename, 'r')

    res = None
    for line in fr.readlines():
        line = line[line.index('OutputIndexFixed = ') + len('OutputIndexFixed = '):]
        line = line[:line.index('\t')].lstrip('[').rstrip(']')
        array = np.fromstring(line, dtype=int, sep=' ')
        if res is None:
            res = array.reshape(1, 3)
        else:
            res = np.concatenate((res, array.reshape(1, 3)), 0)

    return res


'''
This function creates index of imageArray from coordinates
'''


def create_index(points, patch_size):
    i = points[:, 0].reshape(patch_size[0], patch_size[1], patch_size[2])
    j = points[:, 1].reshape(patch_size[0], patch_size[1], patch_size[2])
    k = points[:, 2].reshape(patch_size[0], patch_size[1], patch_size[2])
    return i, j, k


'''
This function creates masks given indices. The mask is the same size as image
'''


def create_mask(shape, i, j, k):
    mask = np.zeros(shape)
    mask[i, j, k] = 1
    return mask


'''
Return two lists of corresponding patches. 
'''


def generate_patches(fixedImage, movingImage, transformParameterMap, filename='test.pts', patch_size=[20, 20, 20],
                     padding=20):
    fixedPatches = []
    movingPatches = []

    x, y, z = 0, 0, 0
    movingImageArray = sitk.GetArrayFromImage(movingImage)
    fixedImageArray = sitk.GetArrayFromImage(fixedImage)
    (x_length, y_length, z_length) = fixedImageArray.shape

    while x < x_length:
        while y < y_length:
            while z < z_length:
                x_range = np.array(range(x, x + patch_size[0]))
                y_range = np.array(range(y, y + patch_size[1]))
                z_range = np.array(range(z, z + patch_size[2]))
                original_points = generate_points(x_range, y_range, z_range)
                i, j, k = create_index(original_points, patch_size)
                original_patch = fixedImageArray * create_mask(fixedImageArray.shape, i, j, k)

                transformixImageFilter = sitk.TransformixImageFilter()
                transformixImageFilter.SetTransformParameterMap(transformParameterMap)
                transformixImageFilter.SetFixedPointSetFileName(filename)
                transformixImageFilter.SetMovingImage(movingImage)
                transformixImageFilter.Execute()

                transformed_points = get_output_points()
                if indexIsValid(transformed_points) == False:
                    continue
                i, j, k = create_index(transformed_points)
                transformed_patch = movingImageArray * create_mask(movingImageArray.shape, i, j, k)

                fixedPatches += [original_patch]
                movingPatches += [transformed_patch]

                z += padding
            y += padding
        z += padding
    return fixedPatches, movingPatches


def indexIsValid(points):
    return (points.flatten() < 0).sum() == 0

generate_patches(fixedImage, movingImage, elastixImageFilter.GetTransformParameterMap())