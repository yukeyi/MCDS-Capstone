import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import copy
from tensorboardX import SummaryWriter
from unet_model import UnetGenerator_3d
from unet_data_loader_util import get_data, load_Directory, get_label_map
from feature_learner_model import featureLearner, featureLearner_old


try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

def load_pairs():
    if(args.KNN != 0):
        register_pairs = {}
        register_pairs['090425_FY89SB_FS'] = '091006_YN89DU_FS'
        register_pairs['100311_RD78TU_FS'] = '101214_KZ78TH_FS'
        register_pairs['090613_YJ67CK_FS'] = '100810_NX39XU_FS'
        register_pairs['100330_JC86VH_FS'] = '140923_BS78VP_FS'
        register_pairs['090927_QF82NU_FS'] = '090922_FZ57YU_FS'
        register_pairs['120820_BD75XH_FS'] = '111013_HS48TU_FS'
        register_pairs['100401_RH93ZU_FS'] = '101212_QV37DU_FS'
        register_pairs['100709_GH46GU_FS'] = '100907_HY25UU_FS'
        register_pairs['090314_KK88XB_FS'] = '110124_JJ63PU_FS'
        register_pairs['110210_FK52JU_FS'] = '110912_FN66YK_FS'
    else:
        register_pairs = {}
        for root, directories, filenames in os.walk(ROOT_DIR +"BrainParameterMaps"):
            for pairname in directories:
                images = pairname.split("-")
                assert(len(images)==2)
                register_pairs[images[0]] = images[1]
    return register_pairs

class BrainImageDataset(Dataset):
    def __init__(self, dirList):
        self.data = dirList

    def __getitem__(self, index):

        global fix_name
        global fix_feature
        global fix_image
        global batch_id

        if(args.KNN == 0):
            name = self.data[index]

        # following code does not exactly make sense, but since we do not need to retrieve all the pairs, it looks OK for now.
        if(args.KNN > 0):
            if(batch_id % 2 == 1):
                while(1):
                    name = self.data[index]
                    if(name in register_pairs):
                        break
                    else:
                        #print("miss "+name)
                        index += 1
                        if(index >= len(self.data)):
                            index = 0
            else:
                name = register_pairs[fix_name]
                register_pairs.pop(fix_name)
        print(name)

        try:
            image = get_data(ROOT_DIR + "Brain2NIFI/" + name + "/norm.nii").astype("float32") / 256
            target = get_data(ROOT_DIR + "Brain2NIFI/" + name + "/aseg.nii")
        except:
            return (np.array([]), np.array([]))

        label = np.zeros(target.shape)
        for i in range(len(label_list)):
            label += ((target == label_list[i])*i)
        label = label[0]
        if(args.KNN > 0 and batch_id % 2 == 1):
            fix_image = copy.deepcopy(image)
        torch.backends.cudnn.enabled = False
        image = torch.tensor(abstract_feature(image)) # this operation cannot be put in 16G machine
        #image = abstract_feature(image)
        torch.backends.cudnn.enabled = True
        if(args.KNN > 0):
            if(batch_id % 2 == 1):
                fix_name = name
                fix_feature = copy.deepcopy(image.cpu().numpy())
            else:
                KNN_res = []
                cnt = 1
                for point in test_points:
                    #print(cnt)
                    cnt += 1
                    if(fix_image[:,point[0],point[1],point[2]].sum() == 0):
                        continue
                    point_res = find_neighbor(image.cpu().numpy(), point, fix_feature[:,point[0],point[1],point[2]])
                    KNN_res.append(point_res)
                print("already saved the KNN res")
                np.save(timeStr+"/"+fix_name+"-"+name+"_KNN_RES",KNN_res)
        return (image, label)

    def __len__(self):
        return len(self.data)


def abstract_feature(image):
    #print(image.shape)

    feature = np.zeros((128,256,256,256))
    with torch.no_grad():
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    #print(x,y,z)
                    feature[:,x*128:x*128+128,y*128:y*128+128,z*128:z*128+128] = \
                        copy.deepcopy(fl_model(torch.tensor([image[:,x*128:x*128+128,y*128:y*128+128,z*128:z*128+128]]).to(device))[0].cpu())

    #print(feature.shape)
    return feature

# original point is put at the front of result
def find_neighbor(image, point, feature):
    diff = image.transpose(1,2,3,0)-feature
    diff = (diff*diff).sum(axis=3).reshape(-1)
    points = np.argpartition(diff,args.KNN)[:args.KNN]
    res = []
    res.append(point)
    for item in points:
        res.append([item//65536,(item%65536)//256,item%256])
        #assert(item == (item//65536)*256*256+((item%65536)//256)*256+item%256)
    del diff
    del points
    return res


def get_loss(dl, model):

    model.eval()
    total_loss = 0.0
    for batch_idx, (whole_data, whole_label) in enumerate(dl):
        slice_depth = 256 // args.shard
        for shard in range(args.shard):
            data, target = whole_data[:,:,shard*slice_depth:(shard+1)*slice_depth].to(device), \
                           whole_label[:,shard*slice_depth:(shard+1)*slice_depth].to(device)
            torch.cuda.empty_cache()
            logsoftmax_output_z = model(data.cuda().float())
            loss = nn.NLLLoss(reduce=False)(logsoftmax_output_z, target.long())
            #loss = loss.float().mean()
            loss = (loss.float() * (args.augmentation * (target > 0) + 1).float()).mean()
            total_loss += loss.item()
            print(shard)

    model.train()

    return total_loss / len(dl)


'''
main_class: the class that you want to predict as one, must be a single value
'''


def binary_vector(x, main_class):
    length = len(x)
    binary = main_class

    return (binary == x).astype(int)


'''
classes: a list of labels that we want for binary comparison
e.g. [1, 2] will return a list of two scores. The first index
is the score of regarding 1 as 1 and 2 as 0. The second is the
score of regarding 2 as 1 and 1 as 0. 

*WARNING: label and target must be of the same dimension. 
'''


def binary_dice_score(label, target, classes):
    scores = []
    smooth = 1

    for cls in classes:
        label_binary = binary_vector(label.flatten(), cls)
        target_binary = binary_vector(target.flatten(), cls)
        intersection = np.sum(label_binary * target_binary)
        normalization = np.sum(label_binary + target_binary)
        score = ((2. * intersection + smooth).sum() /
                 (normalization + smooth).sum())
        scores.append(score)
    return scores

def get_dice_score(dl, model):

    score = []
    slice_depth = 256 // args.shard
    for whole_data, y in dl:
        predicted = np.zeros(y.shape)
        for shard in range(args.shard):
            torch.cuda.empty_cache()
            X = whole_data[:,:,shard*slice_depth:(shard+1)*slice_depth].to(device)
            X = Variable(X).to(device).float()
            output = model(X).cpu()
            output = np.argmax(output.data.numpy(),axis=1)
            predicted[:,shard*slice_depth:(shard+1)*slice_depth] = output
        #predicted.resize((predicted.shape[0]*predicted.shape[1],predicted.shape[2],predicted.shape[3]))

        score.append(binary_dice_score(np.array(y),predicted.astype("int64"), list(np.arange(0,46))))

    return np.mean(score)


def get_KNN_landmark():
    crop_index = [35, 216, 41, 192, 53, 204]
    points = []
    for i in range(crop_index[0],crop_index[1], 60):
        for j in range(crop_index[2],crop_index[3], 50):
            for k in range(crop_index[4],crop_index[5], 50):
                points.append([i,j,k])
    #print(points)
    return points


parser = argparse.ArgumentParser(description='UNET Implementation')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--classes', type=int, default=46, metavar='N',
                    help='total classes of task')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--shard', type=int, default=4, metavar='N',
                    help='split how many shards for one image')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
        help='number of epochs to train (default: 20)')
parser.add_argument('--num_train', type=int, default=3000, metavar='N',
                    help='number of data for training')
parser.add_argument('--num_dev', type=int, default=5, metavar='N',
                    help='number of data for evaluation')
parser.add_argument('--KNN', type=int, default=0, metavar='N',
                    help='if KNN is not 0, we generate KNN matching for each image, K is set')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--augmentation', type=float, default=10.0, metavar='LR',
                    help='weight for lebeled object')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--channel-base', type=int, default=8, metavar='CB',
                    help='number of channel for first convolution (default: 8)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many epoches between logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--test-model', type=str, default=None, metavar='N',
                    help='If test-model has a name, do not do training, just testing on dev and train set')
parser.add_argument('--load-model', type=str, default='2019-11-13-14-08-12/model1450.pt', metavar='N',
                    help='If load-model has a name, use pretrained model')
parser.add_argument('--embedding-model', type=str, default='2019-11-11-12-14-57/model559.pt', metavar='N',
                    help='pretrained feature learning model')
#parser.add_argument('--embedding-model', type=str, default='2019-11-11-12-14-57/model19.pt', metavar='N',
#                    help='pretrained feature learning model')
#parser.add_argument('--embedding-model', type=str, default='2019-11-11-12-58-10/model1039.pt', metavar='N',
#                    help='pretrained feature learning model')
args = parser.parse_args()

ROOT_DIR = "/pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/"
#ROOT_DIR = "/Users/yukeyi/Desktop/"
trainFileName = "trainfiles.txt"
testFileName = "testfiles.txt"
label_map, label_list = get_label_map()

#store all pairs of registration
register_pairs = load_pairs()

timeStr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
os.mkdir(timeStr)
writer = SummaryWriter(timeStr+'/log')
save_model_filename = timeStr + "/model"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = BrainImageDataset(load_Directory(True, args))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
dev_dataset = BrainImageDataset(load_Directory(False, args))
dev_loader = torch.utils.data.DataLoader(dev_dataset,batch_size=args.test_batch_size, shuffle=False)

model = UnetGenerator_3d(128, args.classes, args.channel_base)
if(args.load_model is not None):
    print("Load model : "+args.load_model)
    model = torch.load(args.load_model)
print(device)
model = model.to(device)
print(model)

if(device == torch.device('cpu')):
    fl_model = torch.load(args.embedding_model,map_location='cpu')
else:
    fl_model = torch.load(args.embedding_model)
fl_model = fl_model.to(device)
fl_model.eval()


best_dice = 0.0
best_jaccard = 0
optim = torch.optim.Adam(model.parameters(),lr=args.lr)

if(args.KNN>0):
    test_points = get_KNN_landmark()
    fix_name = ""
    fix_position = []
    fix_feature = []
    fix_image = []
    batch_id = 1

model.train()
for epoch in range(args.epochs):
    #get_dice_score(dev_loader, model)
    #get_accuracy(dev_loader, model)
    total_loss = 0.0
    for batch_idx, (whole_data, whole_label) in enumerate(train_loader):
        if(args.KNN != 0):
            batch_id = batch_idx
            print(batch_idx)
            continue
        if (len(whole_data) > 0):
            try:
                image_loss = 0.0
                slice_depth = 256 // args.shard
                for shard in range(args.shard):
                    data, target = whole_data[:,:,shard*slice_depth:(shard+1)*slice_depth].to(device), \
                                   whole_label[:,shard*slice_depth:(shard+1)*slice_depth].to(device)

                    torch.cuda.empty_cache()
                    logsoftmax_output_z = model(data.cuda().float())
                    loss = nn.NLLLoss(reduce=False)(logsoftmax_output_z, target.long())
                    #loss = loss.float().mean()
                    loss = (loss.float()*(args.augmentation*(target>0)+1).float()).mean()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    total_loss += loss.item()
                    image_loss += loss.item()
                print("Batch : "+str(batch_idx) + " loss : "+str(image_loss))
            except:
                print("except occurs")

        if (batch_idx + 1) % args.log_interval == 0:

            print("Epoch : "+str(epoch)+" Batch: "+str(batch_idx))
            model.eval()
            train_loss = total_loss / args.log_interval
            total_loss = 0.0
            print("total loss : "+str(train_loss))
            #dev_loss = get_loss(dev_loader, model)
            #print("dev loss : "+str(dev_loss))
            writer.add_scalar('Train/Loss', train_loss, epoch*args.num_train+batch_idx+1)
            #writer.add_scalar('Dev/Loss', dev_loss, epoch*args.num_train+batch_idx+1)
            dev_dice = get_dice_score(dev_loader, model)
            print("Dev dice score : " + str(dev_dice))
            writer.add_scalar('Dev/Dice', dev_dice, epoch*args.num_train+batch_idx+1)
            torch.save(model, save_model_filename + str(epoch*args.num_train+batch_idx+1) + '.pt')

            model.train()

print("Done")
