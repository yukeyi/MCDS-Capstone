
import scipy.io
import os

NYC = [[40.695, -74.028],[40.788, -73.940]]
PIT = [[40.425, -80.035],[40.460, -79.930]]


#load the GPS_Long_Lat_Compass.mat file
mat = scipy.io.loadmat('GPS_Long_Lat_Compass.mat')
la_long = mat['GPS_Compass']
city_dic = {}

# assign city label to every placemark
for i in range(len(la_long)):
    coordinate = la_long[i]
    if(NYC[0][0]<=coordinate[0] and coordinate[0]<=NYC[1][0]
        and NYC[0][1]<=coordinate[1] and coordinate[1]<=NYC[1][1]):
        city_dic[i+1] = "NYC"
    else:
        city_dic[i+1] = "PIT"

# create new directories for each class
try:
    os.mkdir("NYC/")
    os.mkdir("PIT/")
except OSError:
    print("Creation of the directories failed")
else:
    print("Successfully created the directories")

# walk all images in current directory
for root, directories, filenames in os.walk("./pitOrManh/"):
    for filename in filenames:
        print(filename)
        if filename.endswith('.jpg'):
            path = os.path.join(root,filename)
            pair = filename.split("_")
            placemark = int(pair[0])
            if city_dic[placemark] == "NYC":
                os.rename(path,"NYC/"+filename)
            else:
                os.rename(path,"PIT/"+filename)




