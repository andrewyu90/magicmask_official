import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import math

class Identify_Feature_Feeding_Block(nn.Module):
    def __init__(self, output_dim, z_id_size=512):
        super(Identify_Feature_Feeding_Block, self).__init__()
        self.output_dim = output_dim
        self.z_id_size = z_id_size
        self.fc = nn.Linear(self.z_id_size, self.output_dim)
    def forward(self, z_id):
        out = self.fc(z_id)
        out = torch.unsqueeze(out,2)
        out = torch.unsqueeze(out,3)
        first_1028 = out[:,0:int(self.output_dim/2),:,:]
        second_1024 = out[:,int(self.output_dim/2):self.output_dim,:,:]
        return first_1028,second_1024

class Operation_Unit(nn.Module):
    def __init__(self, channel, identity_feature_out_dim,identity_feature_in_dim=512,act_out=True):
        super(Operation_Unit, self).__init__()
        self.channel = channel
        self.act_out = act_out
        self.id_feature_out_dim = identity_feature_out_dim
        self.id_feature_in_dim = identity_feature_in_dim
        self.Conv1 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=0)
        self.activation = nn.ReLU()
        self.IFF = Identify_Feature_Feeding_Block(self.id_feature_out_dim,z_id_size=self.id_feature_in_dim)

    def forward(self, input_feature, id_feature):
        idf0_0, idf0_1 = self.IFF(id_feature)
        x0 = F.pad(input_feature, (1, 1, 1, 1, 0, 0), mode='reflect')
        x0 = self.Conv1(x0)
        x0 = torch.sub(x0,torch.mean(x0, (2, 3),keepdim=True))

        x1 = torch.mul(x0, x0)
        x1 = torch.mean(x1, (2, 3),keepdim=True)
        x1 = torch.add(x1, 9.99999993922529e-9)
        x1 = torch.sqrt(x1)
        x1 = torch.div(1.0,x1)
        x2 = torch.mul(x0, x1)
        x2 = torch.mul(idf0_0, x2)
        x2 = torch.add(x2, idf0_1)
        if self.act_out==True:
            return self.activation(x2)
        else:
            return x2

class Feature_Fusion_Block(nn.Module):
    def __init__(self, channel, identity_feature_out_dim,identity_feature_in_dim):
        super(Feature_Fusion_Block, self).__init__()

        self.channel = channel
        self.identity_feature_out_dim = identity_feature_out_dim
        self.identity_feature_in_dim = identity_feature_in_dim

        self.OP1 = Operation_Unit(self.channel, self.identity_feature_out_dim,identity_feature_in_dim=self.identity_feature_in_dim)
        self.OP2 = Operation_Unit(self.channel, self.identity_feature_out_dim,identity_feature_in_dim=self.identity_feature_in_dim,act_out=False)


    def forward(self, input_feature, id_feature):
        x = self.OP1(input_feature,id_feature)
        x = self.OP2(x, id_feature)
        return input_feature + x


class Open_Encoder(nn.Module):
    def __init__(self,id_feature_dim):
        super(Open_Encoder, self).__init__()
        self.id_feature_dim = id_feature_dim
        self.Encoder_channel = [3,128, 256, 512, 1024]
        self.Encoder_kernel_size = [7,3,3,3]
        self.pading_scale = [0,1,1,1]
        self.stride_scale = [1,1,2,2]
        self.Encoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
                nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i+1], kernel_size=self.Encoder_kernel_size[i], stride=self.stride_scale[i], padding=self.pading_scale[i]),
                nn.LeakyReLU(0.2)
            )for i in range(4)})

        #self.FFB1 = Feature_Fusion_Block(1024,2048)
        
        self.fusion_module = nn.ModuleDict(
            {f'fusion_layer_{i}' : Feature_Fusion_Block(1024,2048,512) 
        for i in range(6)})


    def forward(self,x,id_feature):
        x = F.pad(x, (3, 3, 3, 3, 0, 0), mode='reflect')
        for i in range(4):
            x = self.Encoder[f'layer_{i}'](x)
        #x = self.FFB1(input_feature=x,id_feature=id_feature)
        
        for i in range(6):
            x = self.fusion_module[f'fusion_layer_{i}'](x,id_feature)
        return x


class Sesame_Generator(nn.Module):
    def __init__(self, in_channels=1023, out_channels=3):
        super(Sesame_Generator, self).__init__()

        self.in_channel = in_channels
        self.out_channel = out_channels

        # self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.Upsample = nn.Upsample(scale_factor=2,align_corners=False,mode='bilinear')

        self.Conv1 = nn.Conv2d(self.in_channel,512, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(512,256, kernel_size=3, stride=1, padding=1)
        self.Conv3 = nn.Conv2d(256,128, kernel_size=3, stride=1, padding=1)
        self.Conv4 = nn.Conv2d(128, self.out_channel, kernel_size=7, stride=1, padding=0)

        self.Activation_LeakyRelu =  nn.LeakyReLU(0.2)
        self.Activation_Tanh =  nn.Tanh()
        

    def forward(self,visual_feature,geometric_feature=None,out_latent=True):
        list_of_latent= []
        x = self.Upsample(visual_feature)
        #pdb.set_trace()
        if geometric_feature!=None:
            x = self.Activation_LeakyRelu(torch.add(self.Conv1(x),geometric_feature))
        else:
            x = self.Activation_LeakyRelu(self.Conv1(x))
        list_of_latent.append(x)
        x = self.Upsample(x)
        x = self.Activation_LeakyRelu(self.Conv2(x))
        list_of_latent.append(x)
        x = self.Activation_LeakyRelu(self.Conv3(x))
        list_of_latent.append(x)
        x = F.pad(x, (3, 3, 3, 3, 0, 0), mode='reflect')
        x = self.Conv4(x)
        x = self.Activation_Tanh(x)
        x = torch.add(x,1.0)
        x = torch.div(x,2.0)
        if out_latent==True:
            return x, list_of_latent
        return x


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance



class Discriminator(nn.Module):
    def __init__(self, img_size=256, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        num_domains = 1
        blocks = []
        blocks += [nn.Conv2d(6, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x,lm_images):
        #out = self.main(x)
        #pdb.set_trace()
        x = torch.cat((x,lm_images), dim=-3)
        for block in self.main:
            x = block(x)
        out = x
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out

if __name__ == '__main__':
    #Test 'Open!' Encoder
    Encoder = Open_Encoder(512)
    Generator = Sesame_Generator(1024,3)
    
    dummy_target = torch.rand(1,3,128,128)
    dummy_id_feature = torch.rand(1, 512)
    print(dummy_target.shape)
    print(dummy_id_feature.shape)
    encoder_output = Encoder(dummy_target,dummy_id_feature)

    
    print(encoder_output.shape)
    #Test Sesame! Generator
    output = Generator(encoder_output)
    print(output.shape)
    print('Done!')
