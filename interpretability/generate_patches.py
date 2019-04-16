from sklearn.feature_extraction import image
from PIL import Image
import cv2
import os
import concurrent.futures
from more_itertools import grouper
import shutil

patch_size = 227
stride = 113
partition = 4


def generate(filename):
	dirName = os.path.join(root_name, filename.split(".jpg")[0])
	try:
		os.mkdir(dirName)
	except:
		shutil.rmtree(dirName)
		os.mkdir(dirName)
	filei = os.path.join(root_name, filename)

	img = cv2.imread(filei)
	img_width = img.shape[0]
	img_height = img.shape[1]
	for x in range(0, img_width - patch_size, stride):
		for y in range(0, img_height - patch_size, stride):
			img_patch = img[x:x+patch_size,y:y+patch_size,:]
			img_patch = Image.fromarray(img_patch, 'RGB')
			img_patch.save(dirName+"/"+filename.split(".jpg")[0]+"_x"+str(x)+"_y"+str(y)+".png")
	print(filename)

def loop_directory(img_dir):
	image_list = []
	global root_name
	root_name = ""
	
	for root, directories, filenames in os.walk(img_dir):
		for filename in filenames:
			if filename.endswith('.jpg'):
				image_list.append(filename)
		if root_name is "":
			root_name = root

	executor = concurrent.futures.ProcessPoolExecutor(partition)
	futures = [executor.submit(generate, filename) for filename in image_list]
	concurrent.futures.wait(futures)
		

if __name__ == '__main__':
	loop_directory("./NYC")
	loop_directory("./PIT")
