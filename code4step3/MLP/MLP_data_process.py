import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import random
import argparse
import time
import os

def check_label_count(label):
    for i in range(46):
        print("number of pixels with label: "+str(i)+"  "+str((label == i).sum()))

def sample_data(max_size):
    #embedding = np.load("MLP_image.npy").T
    embedding = np.load("MLP_image.npy")
    label = np.load("MLP_label.npy")

    sampled_embedding = []
    sampled_label = []
    label_count = [0]*46

    temp = list(zip(embedding, label))
    random.shuffle(temp)
    embedding, label = zip(*temp)
    embedding = np.array(embedding)
    label = np.array(label)
    #np.save("MLP_image.npy", embedding)
    #np.save("MLP_label.npy", label)
    #exit()

    for i in range(len(label)):
        if(label_count[int(label[i])] >= max_size):
            continue
        label_count[int(label[i])] += 1
        sampled_embedding.append(embedding[i])
        sampled_label.append(int(label[i]))


    print(label_count)
    sampled_embedding = np.array(sampled_embedding)
    np.save("MLP_sampled_image.npy", sampled_embedding)
    sampled_label = np.array(sampled_label)
    np.save("MLP_sampled_label.npy", sampled_label)


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 512)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):

    global train_loss

    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # data = data.view(-1, 40)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss.append(np.array(losses).mean())
    print("Epoch "+str(epoch)+"  Training loss: "+str(np.array(losses).mean()))


def test(args, model, device, test_loader, train, epoch=0):

    global dev_loss
    global dev_acc
    global dev_dice

    model.eval()
    test_loss = 0
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            predictions += list(pred.numpy().reshape(-1))
            ground_truth += list(target.numpy())
            #correct += pred.eq(target.view_as(pred)).sum().item()
            #print(len(ground_truth))

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    correct = (predictions == ground_truth).sum()
    dice_array = []
    for i in range(46):
        if((predictions == i).sum()+(ground_truth == i).sum()==0):
            continue
        TP = ((predictions == i) * (ground_truth == i)).sum()
        dice_array.append(2*TP/((predictions == i).sum()+(ground_truth == i).sum()))
    dice_array = np.array(dice_array)

    test_loss /= len(test_loader.dataset)

    if(train):
        prefix = "On training set:  "
    else:
        prefix = "On validation set:  "

    print(prefix + "Epoch "+str(epoch)+", Average loss: "+str(test_loss)+" Accuracy: "+
          str(100. * correct / len(test_loader.dataset))+" Dice: "+str(dice_array.mean()))

    dev_loss.append(test_loss)
    dev_acc.append(100. * correct / len(test_loader.dataset))
    dev_dice.append(dice_array.mean())
    return dice_array.mean()

class MyCustomDataset(Dataset):

    def __init__(self, args, train):
        if(args.test or train == False):
            self.embedding = np.load("MLP_dev_image.npy").astype('float32')
            self.label = np.load("MLP_dev_label.npy").astype('int64')
        else:
            if(args.use_sample_data):
                self.embedding = np.load("MLP_sampled_image.npy").astype('float32')
                self.label = np.load("MLP_sampled_label.npy").astype('int64')
                #if(train == True):
                #    self.embedding = np.load("MLP_sampled_image.npy")[:-5000].astype('float32')
                #    self.label = np.load("MLP_sampled_label.npy")[:-5000].astype('int64')
                #else:
                #    self.embedding = np.load("MLP_sampled_image.npy")[-5000:].astype('float32')
                #    self.label = np.load("MLP_sampled_label.npy")[-5000:].astype('int64')
            else:
                self.embedding = np.load("MLP_image.npy").astype('float32')
                self.label = np.load("MLP_label.npy").astype('int64')
                #if(train == True):
                #    self.embedding = np.load("MLP_image.npy")[:-50000].astype('float32')
                #    self.label = np.load("MLP_label.npy")[:-50000].astype('int64')
                #else:
                #    self.embedding = np.load("MLP_image.npy")[-50000:].astype('float32')
                #    self.label = np.load("MLP_label.npy")[-50000:].astype('int64')

    def __getitem__(self, index):
        return (self.embedding[index],self.label[index])

    def __len__(self):
        return len(self.label)


parser = argparse.ArgumentParser(description='PyTorch MLP')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=50000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many epoch to save the model')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--use-sample-data', type=int, default=0, metavar='N',
                    help='if set as 1, we use sampled data, otherwise use entire data')
parser.add_argument('--test', type=int, default=0, metavar='N',
                    help='if set as 1, we test on another brain')
parser.add_argument('--load-model', type=str, default='2019-12-02-18-32-29/model_6:0.35018829135226054.pt', metavar='N',
                    help='If load-model has a name, use pretrained model')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

print("Using device: "+str(device))

# saved data
train_loss = []
dev_loss = []
dev_acc = []
dev_dice = []
best_dice = 0.0

timeStr = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
os.mkdir(timeStr)

model = Net(32,46).to(device)
if (len(args.load_model) > 1):
    model.load_state_dict(torch.load(args.load_model))
if(args.test):
    dev_loader = torch.utils.data.DataLoader(MyCustomDataset(args, train=False), batch_size=args.test_batch_size,
                                             shuffle=False)
    test(args, model, device, dev_loader, train=True, epoch=0)
    exit()

train_loader = torch.utils.data.DataLoader(MyCustomDataset(args, train=True), batch_size=args.batch_size, shuffle=True)

dev_loader = torch.utils.data.DataLoader(MyCustomDataset(args, train=False), batch_size=args.test_batch_size, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

#test(args, model, device, dev_loader, train=False, epoch=0)

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    dice = test(args, model, device, dev_loader, train=False, epoch=epoch)
    #test(model, device, train_loader, train=True)
    if(dice >= best_dice):
        best_dice = dice
        torch.save(model.state_dict(), timeStr + "/model_" + str(epoch) + ":" + str(dice) + ".pt")
    np.save(timeStr + "/train_loss.npy", np.array(train_loss))
    np.save(timeStr + "/dev_loss.npy", np.array(dev_loss))
    np.save(timeStr + "/dev_acc.npy", np.array(dev_acc))
    np.save(timeStr + "/dev_dice.npy", np.array(dev_dice))
