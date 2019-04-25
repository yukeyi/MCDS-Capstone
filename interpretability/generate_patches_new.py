from sklearn.feature_extraction import image
from PIL import Image
import cv2
import os
import concurrent.futures
from more_itertools import grouper
import shutil
import numpy as np

resize = 227
partition = 30

def generate(filename):
    if 'NYC' in filename:
        label = 'NYC/'
    else:
        label = 'PIT/'
    dirName = '/pylon5/ac5616p/faqian/pit_nyc/val_fullSize_patches/'+label
    img_name = filename.split("/")[-1].split(".jpg")[0]

    img = Image.open(filename)
    #divide the image to 8 patches
    img = np.array(img)
    img= img[100:, 100:, :]
    height, length, x = img.shape
    patch_size = int (height * 2 / 3)
    patch_starting_pts = []
    shift_len = int(length - patch_size)
    for i in range(2):
        for j in range(2):
            patch_starting_pts.append([i*shift_len,j*(height-patch_size)])
    img_list = [ img[i:i+patch_size,j:j+patch_size, :] for i, j in patch_starting_pts]
    img_list = [ cv2.resize(img, (resize,resize), interpolation=cv2.INTER_CUBIC) for img in img_list]
    for i in range(4):
        img_patch = img_list[i]
        img_patch = Image.fromarray(img_patch, 'RGB')
        x,y = patch_starting_pts[i]
        print(dirName+img_name+"_x"+str(x)+"_y"+str(y)+"_ps"+str(patch_size)+".jpg")
        img_patch.save(dirName+img_name+"_x"+str(x)+"_y"+str(y)+"_ps"+str(patch_size)+".jpg")
    

def loop_directory(img_dir):
    image_list = []
    global root_name
    root_name = ""

    for root, directories, filenames in os.walk(img_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                image_list.append(root+"/"+filename)
        if root_name is "":
            root_name = root

    executor = concurrent.futures.ProcessPoolExecutor(partition)
    futures = [executor.submit(generate, filename) for filename in image_list]
    concurrent.futures.wait(futures)
        

if __name__ == '__main__':
    print("looping val NYC")
    loop_directory("/pylon5/ac5616p/faqian/pit_nyc/val_fullSize/NYC/")
    print("looping val PIT")
    loop_directory("/pylon5/ac5616p/faqian/pit_nyc/val_fullSize/PIT/")
