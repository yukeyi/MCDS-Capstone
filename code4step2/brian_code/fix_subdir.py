import os
import sys
import glob
import shutil

input_dir = "/pylon5/ac5616p/yukeyi/cardiac_cap/data_dir"
output_dir = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/brain/total"

first_folder = glob.glob(input_dir+"/*")
heart_name = [item.split("/")[-1] for item in first_folder]
for item in heart_name:
    os.mkdir(output_dir+"/"+item)
    os.mkdir(output_dir + "/" + item + "/SA")

for item in first_folder:
    print(item)
    second_folder = glob.glob(item+"/SA/*")
    slice_name = [i.split("/")[-1] for i in second_folder]
    for it2 in slice_name:
        print(it2)
        os.mkdir(output_dir + "/" + item.split("/")[-1] + "/SA/" + it2)
        third_folder = glob.glob(item+"/SA/"+it2+"/*")
        for i in third_folder:
            temp = i.split("/")[-1]
            if(temp[0] == "T"):
                new_temp = "DE"+temp
            else:
                new_temp = "DET" + temp
            shutil.copy(i, output_dir + "/" + item.split("/")[-1] + "/SA/" + it2+"/"+new_temp)


