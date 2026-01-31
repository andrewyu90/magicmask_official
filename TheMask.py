import torch
import torch.nn.functional as F
import pdb
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.models import resnet101
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch import nn
from Model.Neural_Visionary_module import Open_Encoder, Sesame_Generator, Discriminator
from Model.Neural_Geometric_module import Generator, Interpolate
from models.arcface_resnet import resnet50,resnet_face18 
#from Model.MultiScaleDiscriminator import MultiscaleDiscriminator
from Model.iresnet import iresnet100
import numpy as np
#from core.wing import FAN

#from Model.loss import GANLoss, AEI_Loss
batch_size = 1
ema_path = 'emp.npy'

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        x = x - self.mean
        x = x / self.std
        return x

class ResNet_ID_encoder(pl.LightningModule):
    def __init__(self,ema_path=None):
        super(ResNet_ID_encoder, self).__init__()
        self.resnet = resnet50()
        #pdb.set_trace()
        if ema_path==None:
            self.ema = nn.Linear(512, 512).cuda()
            self.ema_init = False
        else:
            self.ema = torch.from_numpy(np.load(ema_path)).cuda()
            self.ema_init = True
            
        
    def forward(self, inputs):
        x = self.resnet(inputs)
        if self.ema_init==False:
            x = self.ema(x)
        else:
            x = torch.matmul(x, self.ema)
        out = torch.div(x,torch.linalg.norm(x,dim=1,keepdim=True))
        return out


class Visual_Representation_Module(pl.LightningModule):
    def __init__(self,source_dim):
        super(Visual_Representation_Module, self).__init__()
        self.source_dim = source_dim
        
        self.E = Open_Encoder(self.source_dim)
        self.G = Sesame_Generator(1024,3)

  
    def forward(self, target, source, geometric_feature=None):
        # source = np.dot(source, self.ema)
        # source /= np.linalg.norm(source)
        feature_map = self.E(target,source)
        if geometric_feature!=None:
            output,latent_list = self.G(feature_map,geometric_feature=geometric_feature)
        else:
            output,latent_list  = self.G(feature_map,geometric_feature=None)
        return output,feature_map,latent_list


class The_Mask(pl.LightningModule):
    def __init__(self,nerual_visionariy_module,nerual_geometry_module,id_encoder,embedding_transfer_dim = [512,128]):
        super(The_Mask, self).__init__()
        self.NV_module = nerual_visionariy_module.cuda()
        self.NG_module = nerual_geometry_module.cuda()
        self.ID_encoder = id_encoder.cuda()
        self.embedding_transfer_dim_input = embedding_transfer_dim[0]
        self.embedding_transfer_dim_out = embedding_transfer_dim[1]
        self.embedding_transfer_net = nn.Linear(self.embedding_transfer_dim_input,self.embedding_transfer_dim_out).cuda()

        self.discriminator = None
        self.train_adv=False

        self.transform_for_face =  transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    ])
        self.transform_for_nvm =  transform = transforms.Compose([
                                        transforms.Resize([112, 112]),
                                        transforms.ToTensor()
                                    ])

    def initialisation(self):
        self.NG_module.decode = nn.Sequential(*list(self.NG_module.decode.children())[:-3])
        self.NG_module.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2),
            Interpolate(scale=2, mode='nearest'),
            nn.Conv2d(512, 512, 1, 1, 0)).cuda()

    def Prepareing_adversrial_learning(self):
        self.discriminator = Discriminator(img_size=128, max_conv_dim=512).cuda()
        self.train_adv = True
    
        #Make connection
        #using For loop, check name and if there is some specific keyword, the value will be zero.

    def set_grads_and_opts_4all(self,config,param_list_out=False):
        # parameter on and off and
        # Freeze NVM and ID_encoder.
        nv_layer_idx = 0
        nvm_tunning_params_list = []
        id_transfer_net_tunning_params_list = []
        ngm_tunning_params_list = []

        nvm_parm_list = list(self.NV_module.parameters())
        nvm_generator_parm_list = list(self.NV_module.G.parameters())
        
        ngm_parm_list = list(self.NG_module.parameters())
        id_enc_parm_list = list(self.ID_encoder.parameters())
        emb_transfer_net_parm_list = list(self.embedding_transfer_net.parameters())

        for param in nvm_parm_list:
            param.requires_grad = False
            #nvm_tunning_params_list.append(param)
            
        for param in id_enc_parm_list:
            param.requires_grad = False
            #id_transfer_net_tunning_params_list.append(param)

        
        for param in nvm_generator_parm_list:
            param.requires_grad = True
            #pdb.set_trace()
            nvm_tunning_params_list.append(param)

        
        for param in emb_transfer_net_parm_list:
            param.requires_grad = True
            id_transfer_net_tunning_params_list.append(param)
            
        for param in ngm_parm_list:
            param.requires_grad = True
            ngm_tunning_params_list.append(param)

       
        self.TheMask_opt = torch.optim.Adam([{'params': nvm_tunning_params_list},{'params': id_transfer_net_tunning_params_list},{'params': ngm_tunning_params_list}],lr=config['lr'],betas=[config['beta1'], config['beta2']],weight_decay=config['weight_decay'])

        if self.train_adv:
            print('Adversarial learning Opt is set')
            discriminator_parm_list = list(self.discriminator.parameters())
            for param in discriminator_parm_list:
                param.requires_grad = True
            self.Dis_opt = torch.optim.Adam([{'params': discriminator_parm_list}],lr=config['lr'],betas=[config['beta1'], config['beta2']],weight_decay=config['weight_decay'])
        else:
            print('Adversarial learning is ignored')

    def set_grads_and_opts(self,config,param_list_out=False):
        # parameter on and off and
        # Freeze NVM and ID_encoder.
        nv_layer_idx = 0
        nvm_tunning_params_list = []
        id_transfer_net_tunning_params_list = []
        ngm_tunning_params_list = []


        nvm_parm_list = list(self.NV_module.parameters())
        nvm_generator_parm_list = list(self.NV_module.G.parameters())
        
        ngm_parm_list = list(self.NG_module.parameters())
        id_enc_parm_list = list(self.ID_encoder.parameters())
        emb_transfer_net_parm_list = list(self.embedding_transfer_net.parameters())

        for param in nvm_parm_list:
            param.requires_grad = False
        for param in id_enc_parm_list:
            param.requires_grad = False

        
        for param in nvm_generator_parm_list:
            param.requires_grad = True
            nvm_tunning_params_list.append(param)

        
        for param in emb_transfer_net_parm_list:
            param.requires_grad = True
            id_transfer_net_tunning_params_list.append(param)
            
        for param in ngm_parm_list:
            param.requires_grad = True
            ngm_tunning_params_list.append(param)

        
        self.TheMask_opt = torch.optim.Adam([{'params': nvm_tunning_params_list},{'params': id_transfer_net_tunning_params_list},{'params': ngm_tunning_params_list}],lr=config['lr'],betas=[config['beta1'], config['beta2']],weight_decay=config['weight_decay'])

        if self.train_adv:
            print('Adversarial learning Opt is set')
            discriminator_parm_list = list(self.discriminator.parameters())
            for param in discriminator_parm_list:
                param.requires_grad = True
            self.Dis_opt = torch.optim.Adam([{'params': discriminator_parm_list}],lr=config['lr'],betas=[config['beta1'], config['beta2']],weight_decay=config['weight_decay'])
        else:
            print('Adversarial learning is ignored')

    
    def _set_zero_grads(self):
        self.TheMask_opt.zero_grad()
        if self.train_adv:
            self.Dis_opt.zero_grad()

    def save(self,fname,step):
        print('Saving checkpoint into %s...' % fname)

        PATH_NV = fname+'_nvm_%d'%(step)+'.pt'
        PATH_NG = fname+'_ngm_%d'%(step)+'.pt'
        PATH_ETN = fname+'_etn_%d'%(step)+'.pt'
        PATH_IDC = fname+'_idc_%d'%(step)+'.pt'
        PATH_DIS = fname+'_dis_%d'%(step)+'.pt'

        PATH_NV_last = fname+'_nvm_last'+'.pt'
        PATH_NG_last = fname+'_ngm_last'+'.pt'
        PATH_ETN_last = fname+'_etn_last'+'.pt'
        PATH_IDC_last = fname+'_idc_last'+'.pt'
        PATH_DIS_last = fname+'_dis_last'+'.pt'
        
        #Save the mask module 
        torch.save(self.NV_module.state_dict(), PATH_NV)
        torch.save(self.NG_module.state_dict(), PATH_NG)
        torch.save(self.embedding_transfer_net.state_dict(), PATH_ETN)
        torch.save(self.ID_encoder.state_dict(), PATH_IDC)
        #Save discriminator
        torch.save(self.discriminator.state_dict(), PATH_DIS)

        #Save the mask module 
        torch.save(self.NV_module.state_dict(), PATH_NV_last)
        torch.save(self.NG_module.state_dict(), PATH_NG_last)
        torch.save(self.embedding_transfer_net.state_dict(), PATH_ETN_last)
        torch.save(self.ID_encoder.state_dict(), PATH_IDC_last)
        #Save discriminator
        torch.save(self.discriminator.state_dict(), PATH_DIS_last)
        
    def load(self, fname,step=None):
        print('Loading checkpoint from %s...' % fname)
        if step==None:
            PATH_NV = fname+'__nvm_last'+'.pt'
            PATH_NG = fname+'__ngm_last'+'.pt'
            PATH_ETN = fname+'__etn_last'+'.pt'
            PATH_IDC = fname+'__idc_last'+'.pt'
            PATH_DIS = fname+'__dis_last'+'.pt'
        else:
            PATH_NV = fname+'_nvm_%d'%(step)+'.pt'
            PATH_NG = fname+'_ngm_%d'%(step)+'.pt'
            PATH_ETN = fname+'_etn_%d'%(step)+'.pt'
            PATH_IDC = fname+'_idc_%d'%(step)+'.pt'
            PATH_DIS = fname+'_dis_%d'%(step)+'.pt'
            
        #Save the mask module 
        self.NV_module.load_state_dict(torch.load(PATH_NV))
        self.NG_module.load_state_dict(torch.load(PATH_NG))
        self.embedding_transfer_net.load_state_dict(torch.load(PATH_ETN))
        self.ID_encoder.load_state_dict(torch.load(PATH_IDC))
        #Save discriminator
        self.discriminator.load_state_dict(torch.load(PATH_DIS))
        

    def forward(self, target,target_depth,target_lm, source_id_img, nv_latent_out = False):
        # source = np.dot(source, self.ema)
        # source /= np.linalg.norm(source)
        #pdb.set_trace()
        source_code = self.ID_encoder(source_id_img)
        embedded128D_source_code = self.embedding_transfer_net(source_code)
        temp_latents = self.NG_module(target_depth,target_lm,embedded128D_source_code)
        output, nv_latent,latent_list = self.NV_module(target,source_code,geometric_feature=temp_latents)
        if nv_latent_out==True:  
            return output, nv_latent, latent_list
        else:
            return output, latent_list


def build_models(config=None):
    Neural_Visionary_Module = Visual_Representation_Module(512).cuda()
    Neural_Geometric_Module = Generator(style_dim=128).cuda()


    ID_encoder = ResNet_ID_encoder(ema_path)
    module_dict = torch.load('./arcface_w600k_r50_pytorch.pt', map_location=torch.device('cuda'))
    ID_encoder.resnet.load_state_dict(module_dict)
    
    TheMask = The_Mask(Neural_Visionary_Module,Neural_Geometric_Module,ID_encoder)
    TheMask.initialisation()
    #TheMask.initialisation()
    TheMask.Prepareing_adversrial_learning()
    
    #TheMask.fan = FAN(fname_pretrained='./wing.ckpt').cuda().eval()
    return TheMask

