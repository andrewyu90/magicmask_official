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
from Model.Neural_Visionary_module import Open_Encoder, Sesame_Generator
from Model.Neural_Geometric_module import Generator
#from Model.MultiScaleDiscriminator import MultiscaleDiscriminator
from Model.iresnet import iresnet100
import numpy as np

#from Model.loss import GANLoss, AEI_Loss
batch_size = 1

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        x = x - self.mean
        x = x / self.std
        return x

class MagicMask(pl.LightningModule):
    def __init__(self,source_dim,ema_path):
        super(MagicMask, self).__init__()
        self.source_dim = source_dim
        self.ema = np.load(ema_path)

        #self.G = ADDGenerator(hp.arcface.vector_size)
        self.E = Open_Encoder(self.source_dim)
        self.G = Sesame_Generator(1024,3)
        #self.D = MultiscaleDiscriminator(3)

        #self.Loss_GAN = GANLoss()
        #self.Loss_E_G = AEI_Loss()

    def IDL_initialise(self, id_latent):
        id_latent = np.dot(id_latent, self.ema)
        id_latent /= np.linalg.norm(id_latent)
        return id_latent


    def forward(self, target, source, geometric_feature=None):
        # source = np.dot(source, self.ema)
        # source /= np.linalg.norm(source)
        feature_map = self.E(target,source)
        if geometric_feature!=None:
            output = self.G(feature_map,geometric_feature=geometric_feature)
        else:
            output = self.G(feature_map,geometric_feature=None)
        return output


class MMNet_with_IDEncoder(pl.LightningModule):
    def __init__(self, hp):
        super(MMNet_with_IDEncoder, self).__init__()
        self.hp = hp
        self.ema = np.load(hp.arcface.ema_path)
        #self.G = ADDGenerator(hp.arcface.vector_size)
        self.E = Open_Encoder(hp.arcface.vector_size)
        self.G = Sesame_Generator(1024,3)
        #self.D = MultiscaleDiscriminator(3)


        #Identity Encoder (Output is 512)
        self.Z_e = iresnet100(pretrained=False, fp16=False)
        self.Z_e.load_state_dict(torch.load(hp.arcface.chkpt_path, map_location='cpu'))
        self.Z_e = nn.Sequential(Normalize(0.5, 0.5), self.Z_e)
        #self.Loss_GAN = GANLoss()
        #self.Loss_E_G = AEI_Loss()

    def forward(self, target_img, source_img):
        #Extract latent featurer
        z_id = self.Z_e(F.interpolate(source_img, size=112, mode='bilinear'))
        z_id = np.dot(z_id, self.ema)
        z_id /= np.linalg.norm(z_id)

        resized_target_img = F.interpolate(target_img, size=128, mode='bilinear')
        #z_id = F.normalize(z_id)
        z_id = z_id.detach()
        feature_map = self.E(resized_target_img,z_id)
        output = self.G(feature_map)
        return output




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                        help="path of configuration yaml file")
    parser.add_argument("--target_image", type=str, default="examples/target.png",
                        help="path of preprocessed target face image")
    parser.add_argument("--source_image", type=str, default="examples/source.png",
                        help="path of preprocessed source face image")
    parser.add_argument("--output_path", type=str, default="output.png",
                        help="path of output image")
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    model = MagicMask(hp)
    model.eval()

    filepath = "MMNet_dummy_framework.onnx"
    target_x = torch.randn(batch_size, 3, 128, 128, requires_grad=True)
    source_x = torch.randn(batch_size, 512, requires_grad=True)
    #output = model.forward(target_x, source_img)
    
    torch.onnx.export(model,               # model being run
                  (target_x,source_x),                         # model input (or a tuple for multiple inputs)
                  filepath,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['target','source'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
