import numpy as np
import os

def get_KNN_landmark():
    crop_index = [35, 216, 41, 192, 53, 204]
    points = []
    for i in range(crop_index[0],crop_index[1], 60):
        for j in range(crop_index[2],crop_index[3], 50):
            for k in range(crop_index[4],crop_index[5], 50):
                points.append([i,j,k])
    #print(points)
    return points

def distance(point1, point2):
    dist = 0.0
    for i in range(3):
        dist += (point1[i]-point2[i])*(point1[i]-point2[i])
    return np.sqrt(dist)

def compute_KNN_score(truth, pred):
    dist = 0.0
    num_detect_points = 0
    k = 10
    max_margin = 20
    for key in truth.keys():
        num_detect_points = len(pred[key])
        for i in range(len(pred[key])):
            truth_point = truth[key][compute_position(pred[key][i][0])]
            temp = []
            for j in range(k):
                temp.append(distance(truth_point,pred[key][i][j+1]))
            temp.sort()
            if(temp[0] > max_margin):
                temp[0] = max_margin
            dist += temp[0]
    dist /= (len(truth.keys())*num_detect_points)
    return dist

def compute_position(point):
    x = (point[0]-35)//60
    y = (point[1]-41)//50
    z = (point[2]-53)//50
    return 16*x+4*y+z

points = get_KNN_landmark()

files= os.listdir("ground_truth")

ground_truth = {}
sift = {}
no_sift_final = {}
no_sift_same_step = {}

for file_name in files:
    ground_truth[file_name] = np.load("ground_truth/" + file_name)
    sift[file_name] = np.load("sift/" + file_name[:33]+"_KNN_RES.npy")
    no_sift_final[file_name] = np.load("no_sift_final/" + file_name[:33]+"_KNN_RES.npy")
    no_sift_same_step[file_name] = np.load("no_sift_same_step/" + file_name[:33]+"_KNN_RES.npy")

print("Distance for sift ",compute_KNN_score(ground_truth, sift))
print("Distance for no sift final ",compute_KNN_score(ground_truth, no_sift_final))
print("Distance for no sift same step ",compute_KNN_score(ground_truth, no_sift_same_step))
