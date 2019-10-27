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
# explore parameter margin: for 6 layer CNN use 0.5, for dense net use 6.0

torch.backends.cudnn.enabled = False



def get_output_points(filename='outputpoints.txt'):
    fr = open(filename, 'r')
    res = None
    for line in fr.readlines():
        # Todo: make sure whether we should use OutputIndexMoving or OutputIndexFixed
        # modify the following line, seems to fix the bug

        line = line[line.index('OutputIndexMoving = ') + len('OutputIndexMoving = '):]
        line = line[:line.index('\n')].lstrip('[').rstrip(']')
        array = np.fromstring(line, dtype=int, sep=' ')
        if res is None:
            res = array.reshape(1, 3)
        else:
            res = np.concatenate((res, array.reshape(1, 3)), 0)
    return res


def find_points(point_list, image):
    # 1) write the point to file
    if os.path.exists('test.pts'):
        os.remove('test.pts')
    fr = open('test.pts', 'w')
    fr.write('index' + '\n' + str(len(point_list)))
    for point in point_list:
        fr.write('\n'+str(point[0])+' '+str(point[1])+' '+str(point[2]))
    fr.close()

    # find the corresponding point
    image.register_points()
    transformed_points = get_output_points()
    return transformed_points


def point_redirection(x, y, z):
    # Todo: fix this
    x = x % crop_half_size[0]
    y = y % crop_half_size[1]
    z = z % crop_half_size[2]
    return x, y, z



def right_difficulty(negative_point, fixed_point):
    if(abs(negative_point[0]-fixed_point[0]) < input_args.distanceMargin
            and abs(negative_point[1]-fixed_point[1]) < input_args.distanceMargin
            and abs(negative_point[2]-fixed_point[2]) < input_args.distanceMargin):
        return (input_args.hardMode == 1)
    return (input_args.hardMode == 0)


def find_postive_negative_points(image, fixed_image_array, moving_image_array, Npoints):

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
                        if(right_difficulty(negative_point, fixed_point) and moving_image_array[0][0][x][y][z] != 0):
                            break
                    point_list.append(fixed_point)
                    negative_point_list.append(negative_point)

    positive_point_list = find_points(point_list,image)
    #print(positive_point_list.shape)
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
                cnt += 1
        point_list[i] = [point_list[i][index] for index in good_list]
        positive_point_list[i] = [positive_point_list[i][index] for index in good_list]
        negative_point_list[i] = [negative_point_list[i][index] for index in good_list]

        if(len(good_list) != Npoints//8):
            print("only part data generated : "+str(len(good_list)))

    return point_list, positive_point_list, negative_point_list


def check_boundary_new(a,b,c, x_shard, y_shard, z_shard):
    # Todo: fix that
	return (a>=crop_index[0]+x_shard*crop_half_size[0] and a<crop_index[0]+crop_half_size[0]+x_shard*crop_half_size[0]) \
           and (b>=crop_index[2]+y_shard*crop_half_size[1] and b<crop_index[2]+crop_half_size[1]+y_shard*crop_half_size[1])\
           and (c>=crop_index[4]+z_shard*crop_half_size[2] and c<crop_index[4]+crop_half_size[2]+z_shard*crop_half_size[2])


class CorrespondenceContrastiveLoss(nn.Module):
    """
    Correspondence Contrastive loss
    Takes feature of pairs of points and a target label == 1 if positive pair and label == 0 otherwise
    """

    def __init__(self, margin, batch):
        super(CorrespondenceContrastiveLoss, self).__init__()
        self.margin = margin
        self.batch = batch

    def forward(self, fix_image_feature, moving_image_feature, fixed_points, positive_points, negative_points):
        loss = 0
        cnt = 0
        # positive pairs
        pos_dis = []
        neg_dis = []

        for i in range(self.batch):
            x, y, z = fixed_points[i]
            a, b, c = positive_points[i]
            x, y, z = point_redirection(x, y, z)
            a, b, c = point_redirection(a, b, c)
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
            x, y, z = point_redirection(x, y, z)
            a, b, c = point_redirection(a, b, c)
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
        #exit()
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


def train(args, model, device, loader, optimizer):

    model.train()
    criterion = CorrespondenceContrastiveLoss(args.margin, args.batch)
    timeStr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    os.mkdir(timeStr)
    save_loss_filename = timeStr+"/loss.npy"
    save_dis_filename = timeStr + "/distance.npy"
    save_model_filename = timeStr+"/model"
    loss_history = []
    positive_distance_history = []
    negative_distance_history = []

    for epoch_idx, (fixed_image_array, moving_image_array, fix, moving) in enumerate(loader):
        #print(fix, type(fix))
        #print(moving, type(moving))
        # if we only want to generate points
        print(epoch_idx)
        positive_distance = []
        negative_distance = []

        print("points_data/"+"".join(fix)+"-"+"".join(moving)+"-points.npy")
        if(os.path.exists("points_data/"+"".join(fix)+"-"+"".join(moving)+"-points.npy")):
            try:
                points_data = np.load("points_data/"+"".join(fix)+"-"+"".join(moving)+"-points.npy")
            except:
                continue
            point_list = np.array(points_data[0])
            positive_point_list = np.array(points_data[1])
            negative_point_list = np.array(points_data[2])
        else:
            image = Image("".join(fix)+"-"+"".join(moving))
            print("".join(fix)+"-"+"".join(moving))
            point_list, positive_point_list, negative_point_list = \
                find_postive_negative_points(image, fixed_image_array, moving_image_array, args.Npoints)
            points_data = np.array([point_list, positive_point_list, negative_point_list])
            np.save("points_data/" + "".join(fix) + "-" + "".join(moving) + "-points.npy", points_data)
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

        #for item in point_list[0]:
        #    assert (fixed_image_array[0][0][item[0]][item[1]][item[2]].item() != 0.0)
        #    assert (item[0] < crop_half_size[0])
        #    assert (item[1] < crop_half_size[1])
        #    assert (item[2] < crop_half_size[2])

        losses = []
        for x_shard in range(2):
            for y_shard in range(2):
                for z_shard in range(2):

                    mini_batch = 0

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
                                         negative_point_list[4*x_shard+2*y_shard+z_shard][start_pos:end_pos])
                        positive_distance.append(pos_dis)
                        negative_distance.append(neg_dis)

                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())

                        mini_batch += 1

                        if(mini_batch % args.log_interval == 0):
                            print('Train Epoch: '+str(epoch_idx) + " Corner : "+str(4*x_shard+2*y_shard+z_shard)+" mini_batch: "+
                                  str(mini_batch)+"  percentage: "+
                                  str(100. * epoch_idx / loader.__len__())+"% loss: "+
                                  str(np.array(losses[-1*args.log_interval:]).mean()))

                        if(mini_batch*args.batch*8 == args.Npoints):
                            break

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
            torch.save(model, save_model_filename+str(epoch_idx)+'.pt')



parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--test-model', type=str, default='', metavar='N',
                    help='If test-model has a name, load pre-trained model')
parser.add_argument('--predict-model', type=str, default='', metavar='N',
                    help='If predict-model has a name, do not do training, just give result on dev and test set')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.00001)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='LR',
                    help='weight decay')
parser.add_argument('--margin', type=float, default=0.5, metavar='LR',
                    help='margin')
parser.add_argument('--distanceMargin', type=float, default=20, metavar='LR',
                    help='distanceMargin')
parser.add_argument('--hardMode', type=int, default=0, metavar='LR',
                    help='If hard Mode is set as 1, we only use hard negative example,'
                         ' or we only use easy negative example')
parser.add_argument('--epoch', type=int, default=1, metavar='LR',
                    help='epoch')
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

input_args = parser.parse_args()
crop_index = [25, 225, 28, 204, 48, 208]
crop_size = [200, 176, 160]
crop_half_size = [100, 88, 80]

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using device: "+str(device))

#store all pairs of registration
register_pairs = load_pairs()

model = featureLearner().to(device)
summary(model, input_size=(1, 100, 88, 80))
print(model)
optimizer = optim.Adam(model.parameters(), lr=input_args.lr, betas=(0.9, 0.99), weight_decay=input_args.wd)

train_dataset = BrainImageDataset(load_Directory(True, register_pairs), register_pairs)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1, shuffle=True)

train(input_args, model, device, train_loader, optimizer)
#model.save(0)

