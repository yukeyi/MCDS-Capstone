import copy
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchsummary import summary
from feature_learner_model import *
from feature_learner_data_loader_util import *
from sift import get_sift_feature

torch.backends.cudnn.enabled = False



def get_output_points(filename='outputpoints.txt'):
    fr = open(filename, 'r')
    res = None
    for line in fr.readlines():
        line = line[line.index('OutputIndexMoving = ') + len('OutputIndexMoving = '):]
        line = line[:line.index('\n')].lstrip('[').rstrip(']')
        array = np.fromstring(line, dtype=int, sep=' ')
        array = pixel_to_index(array)
        if res is None:
            res = array.reshape(1, 3)
        else:
            res = np.concatenate((res, array.reshape(1, 3)), 0)
    return res

def pixel_to_index(pixel):
    [x,y,z] = pixel
    return np.array([z-1,y-1,x-1])

def index_to_pixel(index):
    [x,y,z] = index
    return [z+1,y+1,x+1]

def find_points(point_list, image):
    # 1) write the point to file
    if os.path.exists('test.pts'):
        os.remove('test.pts')
    fr = open('test.pts', 'w')
    fr.write('index' + '\n' + str(len(point_list)))
    for point in point_list:
        point = index_to_pixel(point)
        fr.write('\n'+str(point[0])+' '+str(point[1])+' '+str(point[2]))
    fr.close()

    # find the corresponding point
    image.register_points()
    transformed_points = get_output_points()
    return transformed_points


def point_redirection(x, y, z, x_shard,y_shard,z_shard):
    #assert (x - x_shard * crop_half_size[0] == x % crop_half_size[0])
    #assert (y - y_shard * crop_half_size[1] == y % crop_half_size[1])
    #assert (z - z_shard * crop_half_size[2] == z % crop_half_size[2])
    x = x % crop_half_size[0]
    y = y % crop_half_size[1]
    z = z % crop_half_size[2]
    return x, y, z



def right_difficulty(negative_point, fixed_point, epoch_id, batch_idx):
    if(epoch_id != 0):
        distanceMargin = input_args.distanceMargin
    else:
        distanceMargin = int(60-(60-input_args.distanceMargin)*batch_idx/input_args.splstep)
    if(abs(negative_point[0]-fixed_point[0]) < distanceMargin
            and abs(negative_point[1]-fixed_point[1]) < distanceMargin
            and abs(negative_point[2]-fixed_point[2]) < distanceMargin):
        return 1
    return 0


def find_postive_negative_points(image, fixed_image_array, moving_image_array, Npoints, epoch_id, batch_idx):

    point_list = []
    negative_point_list = []

    for x_shard in range(2):
        for y_shard in range(2):
            for z_shard in range(2):
                for i in range(Npoints*2//8):
                    while(1):
                        x = random.randint(crop_index[0],crop_index[0]+crop_half_size[0]-1)+crop_half_size[0]*x_shard
                        y = random.randint(crop_index[2],crop_index[2]+crop_half_size[1]-1)+crop_half_size[1]*y_shard
                        z = random.randint(crop_index[4],crop_index[4]+crop_half_size[2]-1)+crop_half_size[2]*z_shard
                        fixed_point = np.array([x,y,z]).astype('int')
                        if(fixed_image_array[0][0][x][y][z] != 0):
                            break
                    #generate negative point
                    while(1):
                        x = random.randint(crop_index[0],crop_index[0]+crop_half_size[0]-1)+crop_half_size[0]*x_shard
                        y = random.randint(crop_index[2],crop_index[2]+crop_half_size[1]-1)+crop_half_size[1]*y_shard
                        z = random.randint(crop_index[4],crop_index[4]+crop_half_size[2]-1)+crop_half_size[2]*z_shard
                        negative_point = np.array([x,y,z]).astype('int')
                        if(right_difficulty(negative_point, fixed_point, epoch_id, batch_idx) and moving_image_array[0][0][x][y][z] != 0):
                            break
                    point_list.append(fixed_point)
                    negative_point_list.append(negative_point)

    positive_point_list = find_points(point_list,image)
    print(positive_point_list.shape)
    point_list = list(np.array(point_list).reshape((8, Npoints * 2 // 8, 3)))
    negative_point_list = list(np.array(negative_point_list).reshape((8, Npoints * 2 // 8, 3)))
    positive_point_list = list(positive_point_list.reshape((8, Npoints * 2 // 8, 3)))

    for i in range(8):
        x_shard = i//4
        y_shard = (i%4)//2
        z_shard = i%2

        cnt = 0
        good_list = []
        for item in positive_point_list[i]:
            if(check_boundary_new(item[0],item[1],item[2], x_shard, y_shard, z_shard)):
                good_list.append(cnt)
                if(len(good_list) == Npoints//8):
                    break
            # Here is a huge bug in previous code
            cnt += 1

        if(len(good_list) != Npoints//8):
            print("only part data generated : "+str(len(good_list)))
            while(len(good_list) < Npoints//8):
                good_list *= 2
            good_list = good_list[:Npoints//8]

        point_list[i] = [point_list[i][index] for index in good_list]
        positive_point_list[i] = [positive_point_list[i][index] for index in good_list]
        negative_point_list[i] = [negative_point_list[i][index] for index in good_list]

    return point_list, positive_point_list, negative_point_list


def check_boundary_new(a,b,c, x_shard, y_shard, z_shard):
	return (a>=crop_index[0]+x_shard*crop_half_size[0] and a<crop_index[0]+crop_half_size[0]+x_shard*crop_half_size[0]) \
           and (b>=crop_index[2]+y_shard*crop_half_size[1] and b<crop_index[2]+crop_half_size[1]+y_shard*crop_half_size[1])\
           and (c>=crop_index[4]+z_shard*crop_half_size[2] and c<crop_index[4]+crop_half_size[2]+z_shard*crop_half_size[2])


class SiftLoss(nn.Module):

    def __init__(self):
        super(SiftLoss, self).__init__()

    def forward(self, image, points, feature):
        loss = 0.0
        for i in range(len(points)):
            loss += (image[0][:,points[i][0],points[i][1],points[i][2]] - torch.tensor(feature[i]/200).float().cuda()).pow(2).sum()
        return loss

class CorrespondenceContrastiveLoss(nn.Module):
    """
    Correspondence Contrastive loss
    Takes feature of pairs of points and a target label == 1 if positive pair and label == 0 otherwise
    """

    def __init__(self, margin, batch):
        super(CorrespondenceContrastiveLoss, self).__init__()
        self.margin = margin
        self.batch = batch

    def forward(self, fix_image_feature, moving_image_feature, fixed_points, positive_points, negative_points,x_shard,y_shard,z_shard):
        global input_args
        loss = 0.0
        cnt = 0
        # positive pairs
        pos_dis = []
        neg_dis = []

        for i in range(self.batch):
            x, y, z = fixed_points[i]
            a, b, c = positive_points[i]
            x, y, z = point_redirection(x, y, z,x_shard,y_shard,z_shard)
            a, b, c = point_redirection(a, b, c,x_shard,y_shard,z_shard)
            if (input_args.check_correspond and i == 0):
                print(x_shard,y_shard,z_shard,"fix: ",x,y,z,"moving: ",a,b,c)
            distance = (fix_image_feature[0][:,x,y,z] - moving_image_feature[0][:,a,b,c]).pow(2).sum()  # squared distance
            #print(torch.sqrt(distance))
            #print("pos "+str(math.sqrt(distance)))
            loss += distance
            pos_dis.append(torch.sqrt(distance).detach().item())
            #print(distance ** 2)
            cnt += 1

        #a = 0.0
        # negative pairs
        #print("start negative pairs")
        for i in range(self.batch):
            x, y, z = fixed_points[i]
            a, b, c = negative_points[i]
            x, y, z = point_redirection(x, y, z,x_shard,y_shard,z_shard)
            a, b, c = point_redirection(a, b, c,x_shard,y_shard,z_shard)
            distance = (fix_image_feature[0][:,x,y,z] - moving_image_feature[0][:,a,b,c]).pow(2).sum()  # squared distance
            #print(((max(0, self.margin-torch.sqrt(distance))) ** 2))
            #a += torch.sqrt(distance).item()
            '''
            if(torch.sqrt(distance).item() == 0.0):
                print("fuck")
                print(fix_image_feature[0][:,x,y,z])
                print(sdf[0][0][x][y][z])
                #continue
            '''
            neg_dis.append(torch.sqrt(distance).detach().item())
            loss += ((max(0, self.margin-torch.sqrt(distance))) ** 2)
            #print(loss)
            cnt += 1

        #print(a/self.batch)
        loss /= (2*cnt)
        loss *= 100
        #print(np.array(pos_dis).mean(), np.array(neg_dis).mean())
        return loss, pos_dis, neg_dis

def find_boundary(fixed_image_array, moving_image_array):
    # for finding the boundary, which is [11, 228, 25, 221, 47, 209]
    # max 100 boundary, is [25, 221, 28, 205, 48, 209]
    # for now we use [25, 224, 28, 203, 48, 207]
    index_range = [255, 0, 255, 0, 255, 0]

    for image in [fixed_image_array, moving_image_array]:
        index = []
        image = image[0][0]
        dim = image.sum(dim=[1, 2])
        for i in range(0,256):
            if(dim[i] != 0):
                index.append(i)
                break
        for i in range(255,-1,-1):
            if(dim[i] != 0):
                index.append(i)
                break
        dim = image.sum(dim=[0, 2])
        for i in range(0,256):
            if(dim[i] != 0):
                index.append(i)
                break
        for i in range(255,-1,-1):
            if(dim[i] != 0):
                index.append(i)
                break
        dim = image.sum(dim=[0, 1])
        for i in range(0,256):
            if(dim[i] != 0):
                index.append(i)
                break
        for i in range(255,-1,-1):
            if(dim[i] != 0):
                index.append(i)
                break
        #print(index)
        old_index_range = copy.deepcopy(index_range)
        index_range[0] = min(index_range[0], index[0])
        index_range[1] = max(index_range[1], index[1])
        index_range[2] = min(index_range[2], index[2])
        index_range[3] = max(index_range[3], index[3])
        index_range[4] = min(index_range[4], index[4])
        index_range[5] = max(index_range[5], index[5])
        if(old_index_range != index_range):
            print(index_range)

def get_KNN_landmark():
    crop_index = [35, 216, 41, 192, 53, 204]
    points = []
    for i in range(crop_index[0],crop_index[1], 60):
        for j in range(crop_index[2],crop_index[3], 50):
            for k in range(crop_index[4],crop_index[5], 50):
                points.append([i,j,k])
    #print(points)
    return points

def train(args, model, device, loader, optimizer):

    model.train()
    if(args.final_channel == 128):
        margin = 2.4
    elif(args.final_channel == 32):
        margin = 2.0
    elif(args.final_channel == 16):
        margin = 1.5
    elif(args.final_channel == 8):
        margin = 0.8
    criterion = CorrespondenceContrastiveLoss(margin, args.batch)
    criterion_sift = SiftLoss()
    timeStr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    os.mkdir(timeStr)
    save_loss_filename = timeStr+"/loss.npy"
    save_dis_filename = timeStr + "/distance.npy"
    save_model_filename = timeStr+"/model"
    loss_history = []
    positive_distance_history = []
    negative_distance_history = []

    for epoch_id in range(10000):
        print("---------------------------------------------------------------------")
        print("Epoch: "+str(epoch_id))
        print("---------------------------------------------------------------------")
        for batch_idx, (fixed_image_array, moving_image_array, fix, moving) in enumerate(loader):
            #print(batch_idx)
            if(args.KNN != 0):
                global name_list_KNN
                global test_points
                if("".join(fix) not in name_list_KNN):
                    continue
                image = Image("".join(fix) + "-" + "".join(moving))
                print("".join(fix) + "-" + "".join(moving))
                KNN_ground_truth = find_points(test_points,image)
                print(KNN_ground_truth)
                print(KNN_ground_truth.shape)
                np.save("".join(fix) + "-" + "".join(moving)+"-KNN-ground-truth.npy",KNN_ground_truth)
                continue

            if(args.sift == 1):
                losses = []
                for input_image in [fixed_image_array, moving_image_array]:
                    points, features = get_sift_feature(input_image[0][0])
                    for x_shard in range(2):
                        for y_shard in range(2):
                            for z_shard in range(2):

                                if(len(points[4 * x_shard + 2 * y_shard + z_shard]) == 0):
                                    continue
                                part_input_image = input_image[:, :,
                                                   crop_index[0]+x_shard * crop_half_size[0]:crop_index[0]+(x_shard + 1) * crop_half_size[0],
                                                   crop_index[2]+y_shard * crop_half_size[1]:crop_index[2]+(y_shard + 1) * crop_half_size[1],
                                                   crop_index[4]+z_shard * crop_half_size[2]:crop_index[4]+(z_shard + 1) * crop_half_size[2]]

                                part_input_image = part_input_image.to(device)

                                optimizer.zero_grad()
                                part_input_feature = model(part_input_image.float())
                                loss = criterion_sift(part_input_feature,
                                                        points[4 * x_shard + 2 * y_shard + z_shard],
                                                        features[4 * x_shard + 2 * y_shard + z_shard])

                                loss.backward()
                                optimizer.step()
                                losses.append(loss.item())
                print("loss ",np.array(losses).mean())
                loss_history.append(np.array(losses).mean())
                if (len(loss_history) % args.loss_save_interval == 0):
                    np.save(save_loss_filename, np.array(loss_history))
                if (len(loss_history) % args.model_save_interval == 0):
                    torch.save(model, save_model_filename + str(batch_idx + epoch_id * 10000) + '.pt')
                continue

            print(batch_idx+epoch_id*10000)
            positive_distance = []
            negative_distance = []

            print("points_data_tuned_spl/"+"".join(fix)+"-"+"".join(moving)+"-points.npy")
            if(epoch_id > 1):
                try:
                    points_data = np.load("points_data_tuned_spl/"+"".join(fix)+"-"+"".join(moving)+"-points.npy")
                except:
                    continue
                point_list = np.array(points_data[0])
                positive_point_list = np.array(points_data[1])
                negative_point_list = np.array(points_data[2])
            else:
                image = Image("".join(fix)+"-"+"".join(moving))
                print("".join(fix)+"-"+"".join(moving))

                point_list, positive_point_list, negative_point_list = \
                    find_postive_negative_points(image, fixed_image_array, moving_image_array, args.Npoints, epoch_id, batch_idx)
                points_data = np.array([point_list, positive_point_list, negative_point_list])
                if(epoch_id == 1):
                    np.save("points_data_tuned_spl/" + "".join(fix) + "-" + "".join(moving) + "-points.npy", points_data)
                point_list = np.array(point_list)
                positive_point_list = np.array(positive_point_list)
                negative_point_list = np.array(negative_point_list)

            #for item in point_list[0]:
            #    assert(fixed_image_array[0][0][item[0]][item[1]][item[2]].item() != 0.0)

            # crop image and triple points here
            fixed_image_array = fixed_image_array[:, :, crop_index[0]:crop_index[1], crop_index[2]:crop_index[3],
                                crop_index[4]:crop_index[5]]
            moving_image_array = moving_image_array[:, :, crop_index[0]:crop_index[1], crop_index[2]:crop_index[3],
                                 crop_index[4]:crop_index[5]]

            point_list[:, :, 0] -= crop_index[0]
            point_list[:, :, 1] -= crop_index[2]
            point_list[:, :, 2] -= crop_index[4]
            positive_point_list[:, :, 0] -= crop_index[0]
            positive_point_list[:, :, 1] -= crop_index[2]
            positive_point_list[:, :, 2] -= crop_index[4]
            negative_point_list[:, :, 0] -= crop_index[0]
            negative_point_list[:, :, 1] -= crop_index[2]
            negative_point_list[:, :, 2] -= crop_index[4]

            '''
            # sanity check
            for item in point_list[0]:
                assert (fixed_image_array[0][0][item[0]][item[1]][item[2]].item() != 0.0)
                assert (item[0] < crop_half_size[0])
                assert (item[1] < crop_half_size[1])
                assert (item[2] < crop_half_size[2])
            for item in negative_point_list[0]:
                assert (moving_image_array[0][0][item[0]][item[1]][item[2]].item() != 0.0)
                assert (item[0] < crop_half_size[0])
                assert (item[1] < crop_half_size[1])
                assert (item[2] < crop_half_size[2])
            for item in positive_point_list[0]:
                assert (item[0] < crop_half_size[0])
                assert (item[1] < crop_half_size[1])
                assert (item[2] < crop_half_size[2])
            '''

            losses = []
            for x_shard in range(2):
                for y_shard in range(2):
                    for z_shard in range(2):

                        mini_batch = 0

                        if (input_args.check_correspond):
                            sitk.WriteImage(sitk.GetImageFromArray(fixed_image_array[0, 0,
                                                x_shard * crop_half_size[0]:(x_shard + 1) * crop_half_size[0],
                                                y_shard * crop_half_size[1]:(y_shard + 1) * crop_half_size[1],
                                                z_shard * crop_half_size[2]:(z_shard + 1) * crop_half_size[2]]),
                                            "".join(fix)+"_"+str(x_shard)+str(y_shard)+str(z_shard)+".nii")
                            sitk.WriteImage(sitk.GetImageFromArray(moving_image_array[0, 0,
                                                x_shard * crop_half_size[0]:(x_shard + 1) * crop_half_size[0],
                                                y_shard * crop_half_size[1]:(y_shard + 1) * crop_half_size[1],
                                                z_shard * crop_half_size[2]:(z_shard + 1) * crop_half_size[2]]),
                                            "".join(moving)+"_"+str(x_shard)+str(y_shard)+str(z_shard)+".nii")

                        while (1):
                            part_fixed_image_array = fixed_image_array[:, :,
                                                x_shard * crop_half_size[0]:(x_shard + 1) * crop_half_size[0],
                                                y_shard * crop_half_size[1]:(y_shard + 1) * crop_half_size[1],
                                                z_shard * crop_half_size[2]:(z_shard + 1) * crop_half_size[2]]
                            part_moving_image_array = moving_image_array[:, :,
                                                 x_shard * crop_half_size[0]:(x_shard + 1) * crop_half_size[0],
                                                 y_shard * crop_half_size[1]:(y_shard + 1) * crop_half_size[1],
                                                 z_shard * crop_half_size[2]:(z_shard + 1) * crop_half_size[2]]

                            part_fixed_image_array, part_moving_image_array = part_fixed_image_array.to(device), part_moving_image_array.to(device)
                            optimizer.zero_grad()

                            fixed_image_feature = model(part_fixed_image_array.float())
                            moving_image_feature = model(part_moving_image_array.float())

                            start_pos = mini_batch * args.batch
                            end_pos = (mini_batch+1) * args.batch
                            loss, pos_dis, neg_dis = criterion(fixed_image_feature, moving_image_feature,
                                             point_list[4*x_shard+2*y_shard+z_shard][start_pos:end_pos],
                                             positive_point_list[4*x_shard+2*y_shard+z_shard][start_pos:end_pos],
                                             negative_point_list[4*x_shard+2*y_shard+z_shard][start_pos:end_pos],
                                                               x_shard,y_shard,z_shard)
                            positive_distance.append(pos_dis)
                            negative_distance.append(neg_dis)

                            loss.backward()
                            optimizer.step()
                            losses.append(loss.item())

                            mini_batch += 1

                            if(mini_batch % args.log_interval == 0):
                                print('Train Batch: '+str(batch_idx+epoch_id*10000) + " Corner : "+str(4*x_shard+2*y_shard+z_shard)+" mini_batch: "+
                                      str(mini_batch)+"  percentage: "+
                                      str(100. * batch_idx / loader.__len__())+"% loss: "+
                                      str(np.array(losses[-1*args.log_interval:]).mean()))

                            if(mini_batch*args.batch*8 == args.Npoints):
                                break

            if (input_args.check_correspond):
                exit()
            print(np.array(losses).mean())
            loss_history.append(np.array(losses).mean())
            #print(positive_distance)
            positive_distance_history.append(np.array(positive_distance).mean())
            positive_distance_history.append(np.array(positive_distance).std())
            negative_distance_history.append(np.array(negative_distance).mean())
            negative_distance_history.append(np.array(negative_distance).std())
            if(len(loss_history) % args.loss_save_interval == 0):
                np.save(save_loss_filename,np.array(loss_history))
                np.save(save_dis_filename, np.array([positive_distance_history,negative_distance_history]))
            if(len(loss_history) % args.model_save_interval == 0):
                torch.save(model, save_model_filename+str(batch_idx+epoch_id*10000)+'.pt')

        if(args.KNN != 0):
            exit()


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--test-model', type=str, default='', metavar='N',
                    help='If test-model has a name, load pre-trained model')
parser.add_argument('--predict-model', type=str, default='', metavar='N',
                    help='If predict-model has a name, do not do training, just give result on dev and test set')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.00001)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='LR',
                    help='weight decay')
parser.add_argument('--margin', type=float, default=2.4, metavar='LR',
                    help='margin')
parser.add_argument('--distanceMargin', type=float, default=10, metavar='LR',
                    help='distanceMargin')
#parser.add_argument('--hardMode', type=int, default=0, metavar='LR',
#                    help='If hard Mode is set as 1, we only use hard negative example,'
#                         ' or we only use easy negative example')
parser.add_argument('--splstep', type=int, default=1350, metavar='LR',
                    help='number of examples for self pace learning')
parser.add_argument('--sift', type=int, default=0, metavar='LR',
                    help='If sift set as 1, we train the feature learner network using sift feature')
parser.add_argument('--epoch', type=int, default=1, metavar='LR',
                    help='epoch')
parser.add_argument('--check-correspond', type=int, default=0, metavar='LR',
                    help='check whether points feed into loss function is correct')
parser.add_argument('--Npoints', type=int, default=10000, metavar='LR',
                    help='number of points for each image')
parser.add_argument('--batch', type=int, default=250, metavar='LR',
                    help='batch size of each update')
parser.add_argument('--log_interval', type=int, default=5, metavar='LR',
                    help='log_interval')
parser.add_argument('--loss_save_interval', type=int, default=1, metavar='LR',
                    help='loss_save_interval')
parser.add_argument('--model_save_interval', type=int, default=10, metavar='LR',
                    help='model_save_interval')
parser.add_argument('--cubic_size', type=int, default=256, metavar='LR',
                    help='cubic_size')
parser.add_argument('--final_channel', type=int, default=8, metavar='LR',
                    help='final_channel')
parser.add_argument('--load-model', type=str, default=None, metavar='N',
                    help='If load-model has a name, use pretrained model')
parser.add_argument('--KNN', type=int, default=0, metavar='N',
                    help='if KNN is not 0, we generate KNN matching for each image, K is set')

input_args = parser.parse_args()
crop_index = [25, 225, 28, 204, 48, 208]
crop_size = [200, 176, 160]
crop_half_size = [100, 88, 80]

if(input_args.KNN>0):
    test_points = get_KNN_landmark()
    name_list_KNN = ['090425_FY89SB_FS','100311_RD78TU_FS','090613_YJ67CK_FS','100330_JC86VH_FS','090927_QF82NU_FS',
                     '120820_BD75XH_FS','100401_RH93ZU_FS','100709_GH46GU_FS','090314_KK88XB_FS','110210_FK52JU_FS']

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using device: "+str(device))

#store all pairs of registration
register_pairs = load_pairs()

if (input_args.final_channel == 128):
    channels = [32,32,32,32,32,128]
elif (input_args.final_channel == 32):
    channels = [56,32,32,32,32,32]
elif (input_args.final_channel == 16):
    channels = [60,32,32,32,16,16]
elif (input_args.final_channel == 8):
    channels = [64,32,32,16,16,8]
model = featureLearner(channels).to(device)
if(input_args.load_model is not None):
    print("Load model : "+input_args.load_model)
    model = torch.load(input_args.load_model).to(device)

#summary(model, input_size=(1, 100, 88, 80))
print(model)
optimizer = optim.Adam(model.parameters(), lr=input_args.lr, betas=(0.9, 0.99), weight_decay=input_args.wd)

if(input_args.KNN>0):
    train_dataset = BrainImageDataset(load_Directory(True, register_pairs), register_pairs, input_args.KNN, name_list_KNN)
else:
    train_dataset = BrainImageDataset(load_Directory(True, register_pairs), register_pairs, input_args.KNN, None)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1, shuffle=True)

train(input_args, model, device, train_loader, optimizer)
