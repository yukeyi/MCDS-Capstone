import torch
import torch.nn as nn
from model_util import conv_block_3d_feature_leaner


class featureLearner(nn.Module):
    def __init__(self):
        super(featureLearner, self).__init__()

        self.in_dim = 1
        self.mid1_dim = 60
        self.mid2_dim = 32
        self.mid3_dim = 32
        self.mid4_dim = 32
        self.mid5_dim = 16
        self.out_dim = 16
        #act_fn = nn.LeakyReLU()
        act_fn = nn.ReLU()

        print("\n------Initiating Network------\n")

        self.cnn1 = conv_block_3d_feature_leaner(self.in_dim, self.mid1_dim, act_fn, 1)
        self.cnn2 = conv_block_3d_feature_leaner(self.mid1_dim, self.mid2_dim, act_fn, 1)
        self.cnn3 = conv_block_3d_feature_leaner(self.mid2_dim, self.mid3_dim, act_fn, 2)
        self.cnn4 = conv_block_3d_feature_leaner(self.mid3_dim, self.mid4_dim, act_fn, 2)
        self.cnn5 = conv_block_3d_feature_leaner(self.mid4_dim, self.mid5_dim, act_fn, 4)
        self.cnn6 = conv_block_3d_feature_leaner(self.mid5_dim, self.out_dim, act_fn, 8, True)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if (isinstance(m, nn.Conv3d)):
            nn.init.kaiming_normal(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        out = self.cnn6(x)
        #get_gpu_info(2)
        return out

    def save(self,epoch):
        torch.save(self.state_dict(),"featureLearner"+'_'+str(epoch)+'.pt')



class featureLearner_old(nn.Module):
    def __init__(self):
        super(featureLearner_old, self).__init__()

        self.in_dim = 1
        self.mid1_dim = 32
        self.mid2_dim = 32
        self.mid3_dim = 16
        self.out_dim = 8
        act_fn = nn.ReLU()

        print("\n------Initiating Network------\n")

        self.cnn1 = conv_block_3d_feature_leaner(self.in_dim, self.mid1_dim, act_fn, 1)
        self.cnn2 = conv_block_3d_feature_leaner(self.mid1_dim, self.mid2_dim, act_fn, 2)
        self.cnn3 = conv_block_3d_feature_leaner(self.mid2_dim, self.mid3_dim, act_fn, 4)
        self.cnn4 = conv_block_3d_feature_leaner(self.mid3_dim, self.out_dim, act_fn, 8, True)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if (isinstance(m, nn.Conv3d)):
            # todo: change it to kaiming initialization
            nn.init.kaiming_normal(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        out = self.cnn4(x)
        return out

    def save(self,epoch):
        torch.save(self.state_dict(),"featureLearner"+'_'+str(epoch)+'.pt')