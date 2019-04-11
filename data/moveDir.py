import scipy.io
import os
import shutil
import random
 
def move(src, dest):
    shutil.move(src, dest)

NUM_DIR = 20000
random.seed(31)

# walk all images in current directory
def walk(path):
    for root, directories, filenames in os.walk(path):
        count = 0
        for direct in directories:
            if count >= NUM_DIR:
                break
            check = random.randint(0,9)
            path = os.path.join(root,direct)
            print(path)
            # if check is 0 or 1, put into validation set
            if check < 2:
                move(path,"Val/"+path)
            # else, put into train set
            else:
                move(path,"Train/"+path)
            count += 1

# create new directories for each class
try:
    os.mkdir("Train/")
    os.mkdir("Val/")
except OSError:
    print("Creation of the directories failed")
else:
    print("Successfully created the directories")
print("walking NYC")
walk('./NYC/')
print("walking PIT")
walk('./PIT/')


