import pdb
# import copy
from copy import copy, deepcopy
import math
from typing import Optional
from munch import Munch
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn as nn
import torch.nn.functional as F
# visT
import logging
import torch.nn as nn
from fastai.vision import *

import torchvision.transforms as transforms
from torch.cuda.amp import autocast
import torchvision.transforms.functional as Func

resize = transforms.Resize(256)
L1loss = nn.L1Loss()
L2loss = nn.MSELoss()
Cossim = nn.CosineSimilarity(dim=1)

def Identity_loss(source_identity_code,swapped_identity_code):
    #Contrastive learning between the swapped face and source face under a latent feature space    
    loss = 1.0 - Cossim(source_identity_code, swapped_identity_code)
    return loss

def Attribute_loss(target_img,swapped_img,mask):
    loss = None
    #pdb.set_trace()
    loss = L1loss(target_img*mask,swapped_img*mask)
    return loss



def Reconstruction_loss(img1, img2,dist='l1'):
    if dist=='l2':
        loss = L2loss(img1,img2)
    elif dist=='l1':
        loss = L1loss(img1,img2)
    else:
        print('Recon loss parameter error. Only l2, and l1 are possible')
        exit()
    return loss

def Perceptual_Attribute_loss(target_latent_list, swapped_latent_list, mask):
    loss = 0.0
    #pdb.set_trace()
    for _t_latent, _s_latent in zip(target_latent_list, swapped_latent_list):
        _b,_d,_w,_h = _t_latent.size()
        _scaled_mask = F.interpolate(mask, size=(_w,_h), mode='bilinear', align_corners=False)
        loss += Reconstruction_loss(_t_latent*_scaled_mask, _s_latent*_scaled_mask,dist='l2')
    return loss


def Adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def R1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def Loss_for_discriminator(model, config, source_img, target_img,  source_id_img, target_id_img, source_img_lm, target_img_lm, source_img_mask, targert_img_mask, source_img_depth=None, target_img_depth=None):
    """
    source_img : source_img
    target_img : target_img
    source_img_lm: source_landmark
    target_img_ml: target_landmark
    source_img_mask: source_mask
    target_img_mask: target_mask
    """
    source_img.requires_grad_()
    target_img.requires_grad_()
    out_source_real = model.discriminator(source_img,source_img_lm)
    out_target_real = model.discriminator(target_img,target_img_lm)
    loss_real_source = Adv_loss(out_source_real, 1)
    loss_real_target = Adv_loss(out_target_real, 1)
    loss_reg_source = R1_reg(loss_real_source, source_img)
    loss_reg_target = R1_reg(loss_real_target, target_img)
    loss_real = loss_real_source + loss_real_target
    loss_reg = loss_reg_source + loss_reg_target
    
    #target,target_depth,target_lm,source_img=None, source_code=None):
    
  
    # The first source_img_lm should be revised to a depth image, this code is just for a test
    target2source_img,_ = model(target_img, target_img_depth, target_img_lm, source_id_img=source_id_img) 
    source2target_img,_ = model(source_img, source_img_depth, source_img_lm, source_id_img=target_id_img)

    out_s2t = model.discriminator(source2target_img, source_img_lm)  # source_imgb， b's identity
    out_t2s = model.discriminator(target2source_img, target_img_lm)  # target_imga， a's identity

    loss_fake_t2s = Adv_loss(out_t2s, 0)
    loss_fake_s2t = Adv_loss(out_s2t, 0)
    loss_fake = loss_fake_t2s + loss_fake_s2t
    loss = loss_real + loss_fake + config['lambda_reg'] * loss_reg
    #loss = loss_real + loss_fake 
    return loss, Munch(real=loss_real.item(),fake=loss_fake.item(),reg=loss_reg.item(),total_loss=loss.item())


def Masked_Loss_for_the_mask(model, config, source_img, target_img, source_id_img, target_id_img, source_img_lm, target_img_lm,
                      source_img_mask, target_img_mask, source_img_depth=None, target_img_depth=None):
    target2target_img,t2t_latent_list = model(target_img, target_img_depth, target_img_lm,
                              source_id_img=target_id_img)  # Swapped face : Target to Sourcd
    source2source_img,s2s_latent_list  = model(source_img, source_img_depth, source_img_lm,
                              source_id_img=source_id_img)  # Swapped face: Source to Target
    # print(target2target_img[0,:,64,64])
    target2source_img, t2s_latent_list  = model(target_img, target_img_depth, target_img_lm,
                              source_id_img=source_id_img)  # Swapped face : Target to Sourcd
    source2target_img, s2t_latent_list  = model(source_img, source_img_depth, source_img_lm,
                              source_id_img=target_id_img)  # Swapped face: Source to Target

    #pdb.set_trace()
    with torch.no_grad():
        source_img_mask128D = F.interpolate(source_img_mask, size=(112, 112), mode='bilinear', align_corners=False)
        target_img_mask128D = F.interpolate(target_img_mask, size=(112, 112), mode='bilinear', align_corners=False)
        source_code = model.ID_encoder(source_id_img*(1-source_img_mask128D))
        target_code = model.ID_encoder(target_id_img*(1-target_img_mask128D))

    target2source_code = model.ID_encoder(
        F.interpolate(target2source_img, size=(112, 112), mode='bilinear', align_corners=False)*(1-target_img_mask128D))
    source2target_code = model.ID_encoder(
        F.interpolate(source2target_img, size=(112, 112), mode='bilinear', align_corners=False)*(1-source_img_mask128D))
    # pdb.set_trace()
    loss_id_source = Identity_loss(source_code, target2source_code)
    loss_id_target = Identity_loss(target_code, source2target_code)

    source_code128 = model.embedding_transfer_net(source_code)
    target_code128 = model.embedding_transfer_net(target_code)
    t2s_code128 = model.embedding_transfer_net(target2source_code)
    s2t_code128 = model.embedding_transfer_net(source2target_code)

    loss_id_source128 = Identity_loss(source_code128, t2s_code128)
    loss_id_target128 = Identity_loss(target_code128, s2t_code128)

    loss_id_source = loss_id_source.mean()
    loss_id_target = loss_id_target.mean()

    loss_id_source128 = loss_id_source128.mean()
    loss_id_target128 = loss_id_target128.mean()

    loss_id = loss_id_source + loss_id_target + loss_id_source128 + loss_id_target128

    out_s2t = model.discriminator(source2target_img, source_img_lm)  # target_imga, a's identity
    out_t2s = model.discriminator(target2source_img, target_img_lm)  # source_imgb, b's identity
    loss_fake_s2t = Adv_loss(out_s2t, 1)
    loss_fake_t2s = Adv_loss(out_t2s, 1)

    loss_adv = loss_fake_s2t + loss_fake_t2s

    loss_recon_source_img = Reconstruction_loss(source2source_img, source_img, dist='l1')
    loss_recon_target_img = Reconstruction_loss(target2target_img, target_img, dist='l1')

    loss_recon = loss_recon_source_img + loss_recon_target_img

    loss_att_source = Attribute_loss(source_img, source2target_img, source_img_mask)
    loss_att_target = Attribute_loss(target_img, target2source_img, target_img_mask)

    loss_attp_source = Perceptual_Attribute_loss(t2t_latent_list, t2s_latent_list, target_img_mask)
    loss_attp_target = Perceptual_Attribute_loss(s2s_latent_list, s2t_latent_list, source_img_mask)

    loss_att = loss_att_source + loss_att_target+ loss_attp_source+ loss_attp_target

    loss = config['lambda_adv'] * loss_adv + config['lambda_id'] * loss_id + config['lambda_recon'] * loss_recon + \
           config['lambda_att_face'] * loss_att
    # loss =  config['lambda_recon']* loss_id + loss_recon + config['lambda_att_face'] * loss_att
    return loss, Munch(id=loss_id.item(),
                       recon=loss_recon.item(),
                       att=loss_att.item(),
                       total_loss=loss.item())
    # return loss, Munch(adv=loss_adv.item(),
    #                    id=loss_id.item(),
    #                    recon=loss_recon.item(),
    #                    att=loss_att.item(),
    #                    total_loss=loss.item())


def Loss_for_the_mask(model, config, source_img, target_img,  source_id_img, target_id_img, source_img_lm, target_img_lm, source_img_mask, target_img_mask, source_img_depth=None, target_img_depth=None):
    
    target2target_img  = model(target_img, target_img_depth, target_img_lm, source_id_img=target_id_img)  # Swapped face : Target to Sourcd
    source2source_img  = model(source_img, source_img_depth, source_img_lm, source_id_img=source_id_img)  # Swapped face: Source to Target
    #print(target2target_img[0,:,64,64])
    target2source_img  = model(target_img, target_img_depth, target_img_lm, source_id_img=source_id_img)   # Swapped face : Target to Sourcd
    source2target_img  = model(source_img, source_img_depth, source_img_lm, source_id_img=target_id_img)  # Swapped face: Source to Target
    
    with torch.no_grad():
        source_code = model.ID_encoder(source_id_img)
        target_code = model.ID_encoder(target_id_img)
        
    target2source_code = model.ID_encoder(F.interpolate(target2source_img, size=(112, 112), mode='bilinear', align_corners=False))
    source2target_code = model.ID_encoder(F.interpolate(source2target_img, size=(112, 112), mode='bilinear', align_corners=False))
    #pdb.set_trace()
    loss_id_source = Identity_loss(source_code, target2source_code)
    loss_id_target= Identity_loss(target_code, source2target_code)

    source_code128 = model.embedding_transfer_net(source_code)
    target_code128 = model.embedding_transfer_net(target_code)
    t2s_code128 = model.embedding_transfer_net(target2source_code)
    s2t_code128 = model.embedding_transfer_net(source2target_code)

    loss_id_source128 = Identity_loss(source_code128, t2s_code128)
    loss_id_target128 = Identity_loss(target_code128, s2t_code128)

    
    loss_id_source = loss_id_source.mean()
    loss_id_target = loss_id_target.mean()

    loss_id_source128 = loss_id_source128.mean()
    loss_id_target128 = loss_id_target128.mean()
    
    loss_id = loss_id_source + loss_id_target + loss_id_source128 + loss_id_target128

    out_s2t = model.discriminator(source2target_img, source_img_lm) #target_imga, a's identity
    out_t2s = model.discriminator(target2source_img, target_img_lm) #source_imgb, b's identity
    loss_fake_s2t = Adv_loss(out_s2t, 1)
    loss_fake_t2s = Adv_loss(out_t2s, 1)
    
    loss_adv = loss_fake_s2t + loss_fake_t2s


    loss_recon_source_img = Reconstruction_loss(source2source_img,source_img,dist='l1')
    loss_recon_target_img = Reconstruction_loss(target2target_img,target_img,dist='l1')
    
    loss_recon = loss_recon_source_img + loss_recon_target_img

    loss_att_swapped_source = Attribute_loss(source_img, source2target_img, source_img_mask)
    loss_att_swapped_target = Attribute_loss(target_img, target2source_img, target_img_mask)

    loss_att_quality_source = Attribute_loss(source_img, source2source_img, source_img_mask)
    loss_att_quality_target = Attribute_loss(target_img, target2target_img, target_img_mask)

    loss_att   = loss_att_swapped_source + loss_att_swapped_target


    loss = config['lambda_adv']*loss_adv + config['lambda_id']* loss_id + config['lambda_recon']*loss_recon + config['lambda_att_face'] * loss_att
    #loss =  config['lambda_recon']* loss_id + loss_recon + config['lambda_att_face'] * loss_att
    return loss, Munch(id=loss_id.item(),
                       recon=loss_recon.item(),
                       att=loss_att.item(),
                       total_loss=loss.item())
    # return loss, Munch(adv=loss_adv.item(),
    #                    id=loss_id.item(),
    #                    recon=loss_recon.item(),
    #                    att=loss_att.item(),
    #                    total_loss=loss.item())


if __name__ == '__main__':
    source_img = np.load('./tmp/source_img.npy')
    target_img = np.load('./tmp/target_img.npy')
    swapped_img = np.load('./tmp/swapped_img.npy')
    
    source_identity_latents = np.load('./tmp/source_identity_code.npy')
    swapped_identity_latents = np.load('./tmp/swapped_identity_code.npy')
    #pdb.set_trace()

    #masks = np.load('./mask.npy')