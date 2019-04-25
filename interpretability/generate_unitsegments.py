# adapted from https://github.com/metalbubble/cnnvisualizer/blob/master/pytorch_generate_unitsegments.py

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from munch import Munch
from PIL import Image
from torch.autograd import Variable

from PIL import Image
import cv2
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', default='resnet18')
parser.add_argument('--config_path', default= '../training/logs/2019-04-23_18-58-42.755288_resnet18/config.yml')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=4)
args = parser.parse_args()



with open(args.config_path, 'r') as f:
    cfg = Munch.fromYAML(f)


# visualization setup
#this is changed from 64 to 32
num_top = 32                # how many top activated images to extract
threshold_scale = 0.2       # the scale used to segment the feature map. Smaller the segmentation will be tighter.

# dataset setup
data_root = "/pylon5/ac5616p/faqian/pit_nyc/val_fullSize/"
resize_size = (227, 227)
output_dir = 'output'


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

print("=> creating model '{}'".format(cfg.arch.model))
model = models.__dict__[cfg.arch.model](pretrained=cfg.arch.pretrained)

if cfg.arch.model == 'alexnet':
    model.classifier._modules['6'] = nn.Linear(4096, cfg.arch.num_classes)
    features = [
        ('conv1', model.features[0]),
        ('conv2', model.features[3]),
        ('conv3', model.features[6]),
        ('conv4', model.features[8]),
        ('conv5', model.features[10])
    ]
elif cfg.arch.model == 'vgg16':
    model.classifier._modules['6'] = nn.Linear(4096, cfg.arch.num_classes)
    features = [
        ('conv1_2', model.features[2]),
        ('conv2_2', model.features[7]),
        ('conv3_3', model.features[14]),
        ('conv4_3', model.features[21]),
        ('conv5_3', model.features[28])
    ]
elif cfg.arch.model == 'inception_v3':
    model = models.inception_v3(pretrained=cfg.arch.pretrained, transform_input=True)
    model.aux_logits = False
    model.fc = nn.Linear(2048, cfg.arch.num_classes)
    features = [
        ('mixed_5d', model.Mixed_5d),
        ('mixed_6a', model.Mixed_6a),
        ('mixed_6e', model.Mixed_6e),
        ('mixed_7a', model.Mixed_7a),
        ('mixed_7c', model.Mixed_7c)
    ]
elif cfg.arch.model == 'resnet152':
    model.fc = nn.Linear(2048, cfg.arch.num_classes)
    features = [
        ('bn1', model.bn1),
        ('layer1', model.layer1),
        ('layer2', model.layer2),
        ('layer3', model.layer3),
        ('layer4', model.layer4)
    ]
elif cfg.arch.model == 'resnet18':
    model.fc = nn.Linear(512, cfg.arch.num_classes)
    features = [
        ('bn1', model.bn1),
        ('layer1', model.layer1),
        ('layer2', model.layer2),
        ('layer3', model.layer3),
        ('layer4', model.layer4)
    ]
else:
    raise Exception

if cfg.arch.model.startswith('alexnet') or cfg.arch.model.startswith('vgg'):
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
else:
    model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True


resume_path = cfg.training.resume.replace(cfg.training.resume[-16:-8], '{:08}'.format(args.epoch))
resume_path = os.path.join('../training', resume_path)
if os.path.isfile(resume_path):
    print("=> loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(resume_path))
    print('try replacing the resume relative path with the full path')
    print('')
    raise Exception


features_blobs = []
def hook_feature(module, input, output):
    # hook the feature extractor
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

for _, module in features:
    module.register_forward_hook(hook_feature)


# change to our dataset
class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        resize = 227
        if (len(self.file_list[index].split("val_fullSize/"))>1):
            image_name = self.file_list[index].split("val_fullSize/")[1]
            img = Image.open(self.file_list[index])
        else:
            image_name = self.file_list[index]
            img = Image.open(data_root + self.file_list[index])
        img = np.array(img)
        img= img[100:, 100:, :]
        height, length, x = img.shape
        #print("hi")
        patch_size = int (height * 2 / 3)
        patch_starting_pts = []
        #print("hi")
        shift_len = int(length - patch_size)
        for i in range(2):
            for j in range(2):
                patch_starting_pts.append([i*shift_len,j*(height-patch_size)])
        img_list = [ img[i:i+patch_size,j:j+patch_size, :] for i, j in patch_starting_pts]
        img_list = [ cv2.resize(img, (resize,resize), interpolation=cv2.INTER_CUBIC) for img in img_list]
        img = np.vstack(img_list)
        img = torchvision.transforms.ToTensor()(img)
        label = self.target_list[index]
        return img, label, patch_starting_pts, patch_size,image_name


#given root directory parse data
def parse_data(datadir, max_count):
    img_list = []
    label_list = []
    NYC_count = 0
    PIT_count = 0
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                if "NYC" in filei:
                    if NYC_count < max_count:
                        img_list.append(filei)
                        label_list.append(0)
                        NYC_count += 1
                else:
                    if PIT_count < max_count:
                        img_list.append(filei)
                        label_list.append(1)
                        PIT_count += 1
    print('{}\t\t{}\n'.format('#Images', len(img_list)))
    return img_list, label_list


def get_loader(mode="val",img_list_sorted = None):
    loader = None
    if mode == "val":
        data_path = "/pylon5/ac5616p/faqian/pit_nyc/train_fullSize/"
        shuffle = False
        img_list, label_list = parse_data(data_path,float('inf'))
        if img_list_sorted is not None:
            dataset = ImageDataset(img_list_sorted, label_list)
        else:
            dataset = ImageDataset(img_list, label_list)
        loader = DataLoader(dataset, shuffle=shuffle, batch_size=32, drop_last=False)
    return loader,dataset

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

#change to list of validation path
image_list = []
for root, directories, filenames in os.walk(data_root):
    for file in filenames:
        if '.jpg' in file:
            path = os.path.join(root,file)
            image_list.append(path)

val_transforms = []
if cfg.arch.model == 'inception_v3':
    val_transforms.append(transforms.Scale(299))


# dataset = DDSM(data_root, image_list, transforms.Compose(val_transforms + [
#     transforms.ToTensor(),
#     normalize,
# ]))
# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=args.batch_size, shuffle=False,
#     num_workers=args.workers, pin_memory=True)
data_loader,dataset = get_loader("val")

# extract the max value activation for each image
imglist_results = []
imglist_info = []
print(len(features))
maxfeatures = [None] * len(features)
num_batches = len(data_loader)
with torch.no_grad():
    for batch_idx, (input, label, patch_starting_pts, patch_size) in enumerate(data_loader):
        del features_blobs[:]
        print('%d / %d' % (batch_idx+1, num_batches))
        input = input.cuda()
        for j in range(4):
            input = input_list[:,:,j*227:j*227+227,:]
            logit = model.forward(input)
        #imglist_results = imglist_results + list(paths)
        if maxfeatures[0] is None:
            # initialize the feature variable
            for i, feat_batch in enumerate(features_blobs):
                size_features = (len(dataset), feat_batch.shape[1])
                maxfeatures[i] = np.zeros(size_features)
        start_idx = batch_idx*args.batch_size
        end_idx = min((batch_idx+1)*args.batch_size, len(dataset))
        for i, feat_batch in enumerate(features_blobs):
            maxfeatures[i][start_idx:end_idx] = np.max(np.max(feat_batch,3),2)


# generate the unit visualization
for layerID, (name, layer) in enumerate(features):
    print(name)
    if (name!='layer4'):
        continue
    num_units = maxfeatures[layerID].shape[1]
    print(num_units)
    imglist_sorted = []
    # load the top activated image list into one list
    print("start sorting units")
    for unitID in range(num_units):
        activations_unit = np.squeeze(maxfeatures[layerID][:, unitID])
        idx_sorted = np.argsort(activations_unit)[::-1]

        for item in idx_sorted[:num_top]:
            print(tuple(dataset[item][1:]))
            imglist_sorted += [tuple(dataset[item][1:])] 
    print("end sorting units")
    print(imglist_sorted)
    # data loader for the top activated images
    # dataset_top = DDSM(data_root, imglist_sorted, transforms.Compose(val_transforms + [
    #     transforms.ToTensor(),
    #     normalize,
    # ]))
    # data_loader_top = torch.utils.data.DataLoader(
    #     dataset_top, batch_size=num_top, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)
    data_loader_top, dataset_top = get_loader("val", imglist_sorted)
    print("data loaded")
    with torch.no_grad():
        for unitID, (input, paths) in enumerate(data_loader_top):
            del features_blobs[:]
            print('%d / %d' % (unitID+1, num_units))
            input = input.cuda()
            logit = model.forward(input)
            feature_maps = features_blobs[layerID]
            images_input = input.cpu().numpy()
            max_value = 0
            print("unitID is")
            print(unitID)
            for i in range(num_top):
                feature_map = feature_maps[i][unitID]
                if max_value == 0:
                    max_value = np.max(feature_map)
                feature_map = feature_map / max_value
                mask = np.array(Image.fromarray(feature_map).resize(resize_size, resample=Image.BILINEAR))
                alpha = 0.2
                mask[mask < threshold_scale] = alpha # binarize the mask
                mask[mask > threshold_scale] = 1.0

                img = Image.open(os.path.join(data_root, paths[i]))
                img = img.resize(resize_size, resample=Image.BILINEAR)
                img = np.asarray(img, dtype=np.float32)
                img_mask = np.multiply(img, mask[:,:, np.newaxis])
                img_mask = np.uint8(img_mask)
                suffix = os.path.basename(list(paths)[i])
                layer_unit_dir = os.path.join(output_dir, 'images', args.experiment_name, name, 'unit_{:04}'.format(unitID + 1))
                if not os.path.exists(layer_unit_dir):
                    os.makedirs(layer_unit_dir)
                out_img_name = os.path.join(layer_unit_dir, '{:04}_{}'.format(i + 1, suffix))
                Image.fromarray(img_mask).save(out_img_name)
