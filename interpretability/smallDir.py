import scipy.io
import os
import shutil


def move(src, dest):
    shutil.move(src, dest)

# walk all images in current directory
def walk(path,max_num,rt):
    count_nyc = 0
    count_pit = 0
    for root, directories, filenames in os.walk(path):
        if(count_nyc >= max_num and count_pit >= max_num):
            break
        for direct in directories:
            path = os.path.join(root,direct)
            arr = direct.split("_")
            print(path)
            if 'NYC' in path and count_nyc < max_num and len(arr) > 1:
                count_nyc += 1
                move(path,rt+"NYC/"+direct )
            if 'PIT' in path and count_pit < max_num and len(arr) > 1:
                count_pit += 1
                move(path,rt+"PIT/"+direct )
            if(count_nyc >= max_num and count_pit >= max_num):
                break


# create new directories for each class
try:
    os.mkdir("SmallTrain/")
    os.mkdir("SmallTrain/NYC/")
    os.mkdir("SmallTrain/PIT/")
    os.mkdir("SmallVal/")
    os.mkdir("SmallVal/NYC/")
    os.mkdir("SmallVal/PIT/")
except OSError:
    print("Creation of the directories failed")
else:
    print("Successfully created the directories")

print("walking Train")
walk('./Val/',5000,"SmallTrain/")
print("walking Validation")
walk('./Train/',1000,"SmallVal/")



