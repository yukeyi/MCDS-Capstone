import cv2
import copy

def get_sift_feature(fixed_image_array, num_points_per_layer = 5):
    points = []
    features = []
    sift = cv2.xfeatures2d.SIFT_create(num_points_per_layer)
    sum_yz = (fixed_image_array>0).sum((1,2))
    for x in range(256):
        if(sum_yz[x] > 1000):
            kp, des = sift.detectAndCompute((fixed_image_array[x]*256).numpy().astype("uint8"), None)
            for i in kp:
                points.append(copy.deepcopy([x, round(i.pt[0]), round(i.pt[1])]))
            features += copy.deepcopy(list(des))

    return points, features