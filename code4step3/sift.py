import cv2
import copy

crop_index = [25, 225, 28, 204, 48, 208]
crop_size = [200, 176, 160]
crop_half_size = [100, 88, 80]

def transform(point):
    point[0] -= crop_index[0]
    point[1] -= crop_index[2]
    point[2] -= crop_index[4]
    x_shard = point[0] // crop_half_size[0]
    y_shard = point[1] // crop_half_size[1]
    z_shard = point[2] // crop_half_size[2]
    point[0] %= crop_half_size[0]
    point[1] %= crop_half_size[1]
    point[2] %= crop_half_size[2]
    return x_shard*4+y_shard*2+z_shard, point

def get_sift_feature(fixed_image_array, num_points_per_layer = 10):
    points = [[],[],[],[],[],[],[],[]]
    features = [[],[],[],[],[],[],[],[]]
    sift = cv2.xfeatures2d.SIFT_create(num_points_per_layer)
    sum_yz = (fixed_image_array>0).sum((1,2))
    for x in range(256):
        if(sum_yz[x] > 1000):
            kp, des = sift.detectAndCompute((fixed_image_array[x]*256).numpy().astype("uint8"), None)
            try:
                for k,d in zip(kp,des):
                    shard, point = transform(copy.deepcopy([x, round(k.pt[0]), round(k.pt[1])]))
                    points[shard].append(point)
                    features[shard].append(copy.deepcopy(d))
            except:
                continue

    return points, features