import os
import sys
import glob
import shutil


def make_subdirs(input_dir=None, output_dir=None):
    if not input_dir:
        input_dir = 'data/DET0000201/SA'
    file_list = glob.glob(input_dir+'/*.dcm')
    file_png = glob.glob(input_dir+'/*.png')
    unique_SAs = set([a[-20:].split('_')[1] for a in file_list]+[a[-20:].split('_')[1] for a in file_png])
    unique_SAs = list(filter(lambda x: x[0] == 'S', unique_SAs))
    print(unique_SAs)

    for SA in unique_SAs:
        try:
            os.mkdir(output_dir+'/'+SA)
        except:
           pass
    for f in file_list:
        SA = f[-20:].split('_')[1]
        try:
            shutil.copy(f, output_dir+'/'+SA+'/'+f[-20:].split('/')[-1])
        except:
            pass
    for f in file_png:
        SA = f[-20:].split('_')[1]
        try:
            shutil.copy(f, output_dir+'/'+SA+'/'+f[-20:].split('/')[-1])
        except:
            pass

if __name__ == '__main__':
    os.mkdir(sys.argv[2])
    folder_list = glob.glob(sys.argv[1]+"/*")
    print(folder_list)
    count = 1
    for folder in folder_list:
        folder = folder[-10:]
        if(folder[:3] != "DET"):
            continue
        print(folder + "   " + str(count))
        count += 1
        os.mkdir(sys.argv[2]+"/"+folder)
        os.mkdir(sys.argv[2]+"/"+folder+"/SA")
        make_subdirs(sys.argv[1]+"/"+folder, sys.argv[2]+"/"+folder+"/SA")
    #os.mkdir("data_dir")
    #make_subdirs("DET0000201", "data_dir")
