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
    k = 1
    max_margin = 10
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
model16_1 = {}
model16_2 = {}
model8_1 = {}
model8_2 = {}
model32_1 = {}
model32_2 = {}
model128_1 = {}
model8_05 = {}
model16_05 = {}
model32_05 = {}
model32_hard = {}
model128_0 = {}
model128_05 = {}


for file_name in files:
    ground_truth[file_name] = np.load("ground_truth/" + file_name)
    model16_1[file_name] = np.load("16_1epoch/" + file_name[:33] + "_KNN_RES.npy")
    model16_2[file_name] = np.load("16_2epoch/" + file_name[:33] + "_KNN_RES.npy")
    model8_1[file_name] = np.load("8_1epoch/" + file_name[:33] + "_KNN_RES.npy")
    model8_2[file_name] = np.load("8_2epoch/" + file_name[:33] + "_KNN_RES.npy")
    model32_1[file_name] = np.load("32_1epoch/" + file_name[:33] + "_KNN_RES.npy")
    model32_2[file_name] = np.load("32_2epoch/" + file_name[:33] + "_KNN_RES.npy")
    model128_1[file_name] = np.load("128_1epoch/" + file_name[:33] + "_KNN_RES.npy")
    model8_05[file_name] = np.load("8_halfepoch/" + file_name[:33] + "_KNN_RES.npy")
    model16_05[file_name] = np.load("16_halfepoch/" + file_name[:33] + "_KNN_RES.npy")
    model32_05[file_name] = np.load("32_halfepoch/" + file_name[:33] + "_KNN_RES.npy")
    model32_hard[file_name] = np.load("32_hard/" + file_name[:33] + "_KNN_RES.npy")
    model128_0[file_name] = np.load("sift_initial/" + file_name[:33] + "_KNN_RES.npy")
    model128_05[file_name] = np.load("128_halfepoch/" + file_name[:33] + "_KNN_RES.npy")

print("Distance for 8_0.5 ",compute_KNN_score(ground_truth, model8_05))
print("Distance for 8_1 ",compute_KNN_score(ground_truth, model8_1))
print("Distance for 8_2 ",compute_KNN_score(ground_truth, model8_2))
print("\n")

print("Distance for 16_0.5 ",compute_KNN_score(ground_truth, model16_05))
print("Distance for 16_1 ",compute_KNN_score(ground_truth, model16_1))
print("Distance for 16_2 ",compute_KNN_score(ground_truth, model16_2))
print("\n")

print("Distance for 32_0.5 ",compute_KNN_score(ground_truth, model32_05))
print("Distance for 32_1 ",compute_KNN_score(ground_truth, model32_1))
print("Distance for 32_2 ",compute_KNN_score(ground_truth, model32_2))
print("Distance for 32_hard ",compute_KNN_score(ground_truth, model32_hard))
print("\n")

print("Distance for 128_sift_init ",compute_KNN_score(ground_truth, model128_0))
print("Distance for 128_0.5 ",compute_KNN_score(ground_truth, model128_05))
print("Distance for 128_1 ",compute_KNN_score(ground_truth, model128_1))
