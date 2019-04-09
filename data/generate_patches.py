from sklearn.feature_extraction import image
from PIL import Image
import cv2
import os


def Image_generate_patches(img_dir):
	for root, directories, filenames in os.walk(img_dir):
		for filename in filenames:
			if filename.endswith('.jpg'):
				dirName = os.path.join(root, filename.split(".jpg")[0])
				os.mkdir(dirName)
				filei = os.path.join(root, filename)

				img = cv2.imread(filei)
				patch_size = 227
				img_width = img.shape[0]
				img_height = img.shape[1]
				stride = 113
				patches = []
				for x in range(0, img_width - patch_size, stride):
					for y in range(0, img_height - patch_size, stride):
						img_patch = img[x:x+patch_size,y:y+patch_size,:]
						#patches.append(img_patch)
						img_patch = Image.fromarray(img_patch, 'RGB')
						img_patch.save(dirName+"/"+"_x"+str(x)+"_y"+str(y)+".png")
				print(filename)

if __name__ == '__main__':
	Image_generate_patches("./NYC")
	Image_generate_patches("./PIT")