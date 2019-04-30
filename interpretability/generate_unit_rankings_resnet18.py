# determine the units most frequently 'influential' to classification decisions

import argparse
import os
import pickle

import numpy as np
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from munch import Munch
from torch.autograd import Variable
from tqdm import tqdm as tqdm
from PIL import Image

import models.resnet
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2


# pytorch 0.2 and torchvision 0.1.9
import torchvision
#assert '0.1.9' in torchvision.__file__

# add condition when using resnet18
def surgery(model, arch, num_classes):
    if arch == 'inception_v3' or arch == 'resnet152':
        model.module.fc.cpu()
        state_dict = model.state_dict()
        state_dict['module.fc.weight'] = state_dict['module.fc.weight'].view(num_classes, 2048, 1, 1)
        model.module.fc = nn.Conv2d(2048, num_classes, kernel_size=(1, 1))
        model.load_state_dict(state_dict)
        model.module.fc.cuda()
    elif arch == 'resnet18':
        model.module.fc.cpu()
        state_dict = model.state_dict()
        state_dict['module.fc.weight'] = state_dict['module.fc.weight'].view(num_classes, 512, 1, 1)
        model.module.fc = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        model.load_state_dict(state_dict)
        model.module.fc.cuda()
    else:
        raise Exception

# change to our dataset
class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        resize = 227
        img = Image.open(self.file_list[index])
        #divide the image to 8 patches
        img = np.array(img)
        img= img[100:, 100:, :]
        height, length, x = img.shape
        patch_size = int (height * 2 / 3)
        patch_starting_pts = []
        shift_len = int(length - patch_size)
        for i in range(2):
            for j in range(2):
                patch_starting_pts.append([i*shift_len,j*(height-patch_size)])
        img_list = [ img[i:i+patch_size,j:j+patch_size, :] for i, j in patch_starting_pts]
        img_list = [ cv2.resize(img, (resize,resize), interpolation=cv2.INTER_CUBIC) for img in img_list]
        #img_list = [ torch.from_numpy(img) for img in img_list]
        #img = torchvision.transforms.ToTensor()(img)
        #img = [ torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img) for img in img_list ] 
        img_list = np.vstack(img_list)
        img_list = torchvision.transforms.ToTensor()(img_list)
        label = self.target_list[index]

        return img_list, label, patch_starting_pts, patch_size

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
def get_loader(mode="train"):
    loader = None
    if mode == "train":
        data_path = "/pylon5/ac5616p/baij/DeepMiner/train_fullSize/"
        shuffle = True
        img_list, label_list = parse_data(data_path, float('inf'))
        dataset = ImageDataset(img_list, label_list)
        loader = DataLoader(dataset, shuffle=shuffle, batch_size=256, drop_last=False)
    if mode == "val":
        data_path = "/pylon5/ac5616p/baij/DeepMiner/val_fullSize/"
        shuffle = False
        img_list, label_list = parse_data(data_path, float('inf'))
        dataset = ImageDataset(img_list, label_list)
        loader = DataLoader(dataset, shuffle=shuffle, batch_size=256, drop_last=False)

    return loader,dataset



def main(args):
    with open(args.config_path, 'r') as f:
        cfg = Munch.fromYAML(f)

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    print("=> creating model '{}'".format(cfg.arch.model))
    if cfg.arch.model == 'inception_v3':
        model = models.inception.inception_v3(use_avgpool=False, transform_input=True)
        model.aux_logits = False
        model.fc = nn.Linear(2048, cfg.arch.num_classes)
        features_layer = model.Mixed_7c
    elif cfg.arch.model == 'resnet152':
        model = models.resnet.resnet152(use_avgpool=False)
        model.fc = nn.Linear(2048, cfg.arch.num_classes)
        features_layer = model.layer4
    elif cfg.arch.model =='resnet18':
        model = models.resnet.resnet18(use_avgpool=False)
        model.fc = nn.Linear(512, cfg.arch.num_classes)
        features_layer = model.layer4
    else:
        raise Exception

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    resume_path = cfg.training.resume.replace(cfg.training.resume[-16:-8], '{:08}'.format(args.epoch))
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))

    # convert fc to conv
    surgery(model, cfg.arch.model, cfg.arch.num_classes)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    patch_size = 299 if cfg.arch.model == 'inception_v3' else 227
    # val_dataset = DDSM(args.raw_image_dir, args.raw_image_list_path, 'val', patch_size, transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize,
    # ]))
    val_dataloader, val_dataset =  get_loader("val")

    # extract features and max activations
    features = []
    def feature_hook(module, input, output):
        features.extend(output.data.cpu().numpy())
    features_layer._forward_hooks.clear()
    features_layer.register_forward_hook(feature_hook)
    prob_maps = []
    max_class_probs = []
    with torch.no_grad():
        for input_list, label, patch_starting_pts, patch_size in tqdm(val_dataset):
            for j in range(4):
                image = input_list[:,j*227:j*227+227,:]
                input = image.unsqueeze(0)
                input = input.cuda()
                output = model(input)
                output = output.transpose(1, 3).contiguous()
                size = output.size()[:3]
                output = output.view(-1, output.size(3))
                prob = nn.Softmax(dim=1)(output)
                prob = prob.view(size[0], size[1], size[2], -1)
                prob = prob.transpose(1, 3)
                prob = prob.cpu().numpy()
                prob_map = prob[0]
                prob_maps.append(prob_map)

    # save final fc layer weights
    params = list(model.parameters())
    weight_softmax = params[-2].data.cpu().numpy().squeeze(3).squeeze(2)

    # rank the units by influence
    max_activations = np.array([feature_map.max(axis=(1, 2)) for feature_map in features])
    max_activations = np.expand_dims(max_activations, 1)
    weighted_max_activations = max_activations * weight_softmax
    unit_indices = np.argsort(-weighted_max_activations, axis=2)
    all_unit_indices_and_counts = []
    for class_index in range(cfg.arch.num_classes):
        num_top_units = 8
        unit_indices_and_counts = zip(*np.unique(unit_indices[:, class_index, :num_top_units].ravel(), return_counts=True))
        unit_indices_and_counts = sorted(unit_indices_and_counts,key=lambda x: -x[1])
        all_unit_indices_and_counts.append(unit_indices_and_counts)    

    # save rankings to file
    unit_rankings_dir = os.path.join(args.output_dir, 'unit_rankings', cfg.training.experiment_name, args.final_layer_name)
    if not os.path.exists(unit_rankings_dir):
        os.makedirs(unit_rankings_dir)
    with open(os.path.join(unit_rankings_dir, 'rankings.pkl'), 'wb') as f:
        pickle.dump(all_unit_indices_and_counts, f)

    # print some statistics
    for class_index in range(cfg.arch.num_classes):
        print('class index: {}'.format(class_index))
        # which units show up in the top num_top_units all the time?
        # note: unit_id == unit_index + 1
        num_top_units = 8
        unit_indices_and_counts = zip(*np.unique(unit_indices[:, class_index, :num_top_units].ravel(), return_counts=True))
        unit_indices_and_counts = sorted(unit_indices_and_counts, key=lambda x: -x[1])
        #unit_indices_and_counts.sort(key=lambda x: -x[1])

        # if we annotate the num_units_annotated top units, what percent of
        # the top num_top_units units on all val images will be annotated?
        num_units_annotated = 20
        print(unit_indices_and_counts[:num_units_annotated])
        annotated_count = sum(x[1] for x in unit_indices_and_counts[:num_units_annotated])
        unannotated_count = sum(x[1] for x in unit_indices_and_counts[num_units_annotated:])
        assert annotated_count + unannotated_count == num_top_units * len(features)
        print('percent annotated: {:.2f}%'.format(100.0 * annotated_count / (annotated_count + unannotated_count)))
        print('')


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default='../training/logs/2019-04-23_18-51-25.936372_resnet18/config.yml')
parser.add_argument('--epoch', type=int, default=22)
parser.add_argument('--final_layer_name', default='layer4')
parser.add_argument('--raw_image_dir', default='../data/ddsm_raw')
parser.add_argument('--raw_image_list_path', default='../data/ddsm_raw_image_lists/val.txt')
parser.add_argument('--output_dir', default='output_fullSize/')
args = parser.parse_args()
main(args)
