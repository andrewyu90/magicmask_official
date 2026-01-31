import pdb

import cv2
import argparse
import face_alignment
import torch
import argparse
import cv2
import numpy as np
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from decalib.deca import DECA
from decalib.datasets import datasets_demo
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import detectors
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from core.data_loader import InputFetcher_TMP
import os
import pdb
#----------------------------------

def crop_face_with_landmarks(frame, landmarks):
    x_min = int(np.min(landmarks[:, 0]))
    x_max = int(np.max(landmarks[:, 0]))
    y_min = int(np.min(landmarks[:, 1]))
    y_max = int(np.max(landmarks[:, 1]))
    margin = 20  # 增加一些邊距
    x_min = max(0, x_min - margin)
    x_max = min(frame.shape[1], x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(frame.shape[0], y_max + margin)
    return frame[y_min:y_max, x_min:x_max]


def fuse_shape(deca, face_detector, src_codedict, ref_img):
    testdata = datasets_demo.TestData(face_detector, [ref_img], iscrop=True, face_detector='fan', sample_step=10)
    device = 'cuda'
    i=0
    depth_image_list = []
    lm_image_list = []
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = 'pytorch3d'
    deca_cfg.model.extract_tex = True
    img = testdata[i]['image'].to(device)[None,...]
    with torch.no_grad():
        ref_codedict = deca.encode(img)
    for i in range(len(src_codedict)):
        with torch.no_grad():
            codedict2 = ref_codedict
            codedict1 = src_codedict[i]
            src_shape = codedict1['shape']

            light_code = codedict1['light']
            tex_code = codedict1['tex']
            detail_code = codedict1['detail']

            ref_shape = codedict2['shape']
            temp = codedict2
            temp['shape'] = src_shape
            temp['light'] = light_code
            temp['tex'] = tex_code
            temp['detail'] = detail_code
            tform = testdata[0]['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2).to(device)
            original_image = testdata[0]['original_image'][None, ...].to(device)
            orig_opdict, orig_visdict = deca.decode(temp, render_orig=True, original_image=original_image, tform=tform)
            orig_visdict['inputs'] = original_image

            lm_image = cv2.resize(util.tensor2image(orig_visdict['landmarks2d'][0]),(256,256))
            depth_image = deca.render.render_depth(orig_opdict['trans_verts']).repeat(1,3,1,1)[0]

            depth_image = depth_image.detach().cpu().numpy()
            depth_image = depth_image*255.
            depth_image = np.maximum(np.minimum(depth_image, 255), 0)
            depth_image = depth_image.transpose(1,2,0)[:,:,[2,1,0]]
            depth_image = Image.fromarray(np.uint8(depth_image))
            lm_image = Image.fromarray(lm_image)
            depth_image_list.append(depth_image)
            lm_image_list.append(lm_image)
    return depth_image_list, lm_image_list

def get_shape(deca, face_detector, ref_img):
    testdata = datasets_demo.TestData(face_detector, [ref_img], iscrop=True, face_detector='fan', sample_step=10)
    device = 'cuda'
    i=0
    depth_image_list = []
    lm_image_list = []
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = 'pytorch3d'
    deca_cfg.model.extract_tex = True
    img = testdata[i]['image'].to(device)[None,...]
    #pdb.set_trace()
    #if len(np.shape(img))!=3:
    #    pdb.set_trace()
    with torch.no_grad():
        ref_codedict = deca.encode(img)
        temp = ref_codedict
        tform = testdata[0]['tform'][None, ...]
        tform = torch.inverse(tform).transpose(1,2).to(device)
        original_image = testdata[0]['original_image'][None, ...].to(device)
        orig_opdict, orig_visdict = deca.decode(temp, render_orig=True, original_image=original_image, tform=tform)
        # orig_visdict['inputs'] = original_image
        # cv2.imwrite('1.png', cv2.resize(util.tensor2image(orig_visdict['landmarks2d'][0]),(256,256)))
        #pdb.set_trace()
        lm_image = cv2.resize(util.tensor2image(orig_visdict['landmarks2d'][0]),(256,256))
        depth_image = deca.render.render_depth(orig_opdict['trans_verts']).repeat(1,3,1,1)[0]

        depth_image = depth_image.detach().cpu().numpy()
        depth_image = depth_image*255.
        depth_image = np.maximum(np.minimum(depth_image, 255), 0)
        depth_image = depth_image.transpose(1,2,0)[:,:,[2,1,0]].astype(np.uint8)
        depth_image = cv2.resize(depth_image,(256,256))
    #pdb.set_trace()
    return depth_image, lm_image #Get depth and land mark image


def disentangle_and_swapping_test(nets, config, inputs, save_dir):
    post_process = config['post_process']
    src, tar, src_id, tar_id, src_lm, tar_lm, src_depth, tar_depth, src_mask, tar_mask = inputs.src, inputs.tar, inputs.src_id, inputs.tar_id, inputs.src_lm, inputs.tar_lm,  inputs.src_depth, inputs.tar_depth,  inputs.src_mask, inputs.tar_mask
    src_parsing, tar_parsing = inputs.src_parsing, inputs.tar_parsing
    src_name, tar_name = inputs.src_name, inputs.tar_name
    src_mask = F.interpolate(src_mask, src.size(2), mode='bilinear', align_corners=True)
    tar_mask = F.interpolate(tar_mask, src.size(2), mode='bilinear', align_corners=True)
    #pdb.set_trace()
    srcid_taratt = nets(tar, tar_depth, tar_lm, source_id_img=tar_id)

    result_first  = save_dir + 'swapped_result_single/'
    result_second = save_dir + 'swapped_result_afterps/'
    result_third  = save_dir + 'swapped_result_all/'
    if not os.path.exists(result_first):
        os.makedirs(result_first)
    if not os.path.exists(result_second):
        os.makedirs(result_second)
    if not os.path.exists(result_third):
        os.makedirs(result_third)
    if post_process:
        #pdb.set_trace()
        src_convex_hull = nets.fan.get_convex_hull(src)
        tar_convex_hull = nets.fan.get_convex_hull(tar)
        temp_src_forehead = src_convex_hull - F.interpolate(src_mask, scale_factor=2, mode='nearest')
        temp_tar_forehead = tar_convex_hull - F.interpolate(tar_mask, scale_factor=2, mode='nearest')
        # to ensure the values of src_forehead and tar_forehead are in [0,1]
        one_tensor  = torch.ones(temp_src_forehead.size()).to(device=temp_src_forehead.device)
        zero_tensor = torch.zeros(temp_src_forehead.size()).to(device=temp_src_forehead.device)
        temp_var = torch.where(temp_src_forehead >= 1.0, one_tensor, temp_src_forehead)
        src_forehead = torch.where(temp_var  <= 0.0, zero_tensor, temp_var)
        temp_var = torch.where(temp_tar_forehead >= 1.0, one_tensor, temp_tar_forehead)
        tar_forehead = torch.where(temp_var <= 0.0, zero_tensor, temp_var)
        tar_hair = get_hair(tar_parsing)
        post_result = postprocess(tar, srcid_taratt, tar_hair, src_forehead, tar_forehead)
    for i in range(len(srcid_taratt)):
        #pdb.set_trace()
        filename = result_first + src_name[i][0:-4]+'_FS_'+ tar_name[i][0:-5]+'.png'
        filename_post = result_second + src_name[i][0:-4] + '_FS_' + tar_name[i][0:-5] + '.png'
        filename_all  = result_third + src_name[i][0:-4] + '_FS_' + tar_name[i][0:-5] + '.png'
        save_image(srcid_taratt[i,:,:,:], 1, filename)
        if post_process:
            save_image(post_result[i, :, :, :], 1, filename_post)
            x_concat = torch.cat([src[i].unsqueeze(0), tar[i].unsqueeze(0),
                                srcid_taratt[i, :, :, :].unsqueeze(0),post_result[i, :, :, :].unsqueeze(0)], dim=0)
            #pdb.set_trace()
            save_image(x_concat, 4, filename_all)   
        else:
            x_concat = torch.cat([src[i].unsqueeze(0), tar[i].unsqueeze(0),
                                srcid_taratt[i, :, :, :].unsqueeze(0)], dim=0)
            save_image(x_concat, 3, filename_all)


@torch.no_grad()
def disentangle_and_swapping_test_mpie(nets, config, inputs, save_dir):
    post_process = config['post_process']
    #pdb.set_trace()
    src, tar, src_id, tar_id, src_lm, tar_lm, src_depth, tar_depth, src_mask, tar_mask = inputs.src, inputs.tar, inputs.src_id, inputs.tar_id, inputs.src_lm, inputs.tar_lm,  inputs.src_depth, inputs.tar_depth,  inputs.src_mask, inputs.tar_mask
    src_name = inputs.src_name
    tar_name = inputs.tar_name
    src_mask = F.interpolate(src_mask, src.size(2), mode='bilinear', align_corners=True)
    tar_mask = F.interpolate(tar_mask, src.size(2), mode='bilinear', align_corners=True)

    #F.interpolate(srcid_taratt, src_id.size(2), mode='bilinear', align_corners=True)
    
    srcid_taratt,_ = nets(tar.to('cuda'), tar_depth.to('cuda'), tar_lm.to('cuda'), source_id_img=src_id.to('cuda'))
    _source_id_latent = nets.ID_encoder(src_id.to('cuda'))
    srcid_taratt_enc = srcid_taratt
    srcid_taratt_enc = (256*srcid_taratt_enc).type(torch.int64)
    srcid_taratt_enc = (srcid_taratt_enc/127.5-1)
    _swapped_id_latent = nets.ID_encoder(F.interpolate(srcid_taratt_enc, src_id.size(2), mode='bilinear', align_corners=True))
    #pdb.set_trace()
    csim = F.cosine_similarity(_source_id_latent, _swapped_id_latent).cpu().numpy()

    #Cosine similarity
    
    result_third  = save_dir
    if not os.path.exists(result_third):
        os.makedirs(result_third)
    for i in range(len(srcid_taratt)):
        #pdb.set_trace()
        #filename = result_first + src_name[i][0:-4]+'_FS_'+ tar_name[i][0:-5]+'.png'
        #filename_post = result_second + src_name[i][0:-4] + '_FS_' + tar_name[i][0:-5] + '.png'
        #pdb.set_trace()
        filename_all  = result_third +'/'+ src_name[i].split('/')[-1].split('.')[0]+ '_FS_' + tar_name[i].split('/')[-1]
        #save_image(srcid_taratt[i,:,:,:], 1, filename)
        #pdb.set_trace()
        x_concat = torch.cat([src[i].unsqueeze(0), tar[i].unsqueeze(0),srcid_taratt[i, :, :, :].cpu().unsqueeze(0)], dim=0)
        save_image(x_concat, 3, filename_all)
        print(filename_all+'           saved')
    return csim


def test_mpie(config,nets_ema, loaders,txt_path,output_dir):
        config = config
        nets_ema = nets_ema
        os.makedirs(config['result_dir'], exist_ok=True)
        #self._load_test_checkpoint(config['test_checkpoint_name'])
        f = open(txt_path, 'r')
        img_num = len(f.readlines())
        f.close()
        total_iters = int(img_num/config['batch_size']) + 1
        save_dir=output_dir
        test_fetcher = InputFetcher_TMP(loaders.src, 'test')
        csims_list = []
        for i in range(0, total_iters):
            inputs = next(test_fetcher)
            csims = disentangle_and_swapping_test_mpie(nets_ema, config, inputs, save_dir)
            csims_list.append(csims)
        return csims_list

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='seg_output.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [255, 255, 255], [0, 0, 0], [255, 255, 255],
                   [255, 255, 255], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    '''
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    '''

    #[255, 85, 0] = face or [0, 85, 255]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.0, vis_parsing_anno_color, 1.0, 0)

    return vis_im


def get_hair(segmentation):
    out = segmentation.mul_(255).int()
    mask_ind_hair = [17]
    with torch.no_grad():
        out_parse = out
        hair = torch.ones((out_parse.shape[0], 1, out_parse.shape[2], out_parse.shape[3])).cuda()
        for pi in mask_ind_hair:
            index = torch.where(out_parse == pi)
            hair[index[0], :, index[2],index[3]] = 0
    return  hair


def postprocess(tar, srcid_taratt, tar_hair, src_forehead, tar_forehead):
    #inner area of tar_hair is  0, inner area of tar_forehead is  1
    smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
    one_tensor = torch.ones(tar_forehead.size()).to(device=tar_forehead.device)
    temp_tar_hair_and_forehead = (1- F.interpolate(tar_hair, scale_factor=2, mode='nearest')) + tar_forehead
    tar_hair_and_forehead = torch.where(temp_tar_hair_and_forehead >= 1.0, one_tensor, temp_tar_hair_and_forehead)
    tar_preserve = tar_hair
    # find whether occlusion exists in source image; if exists, then preserve the hair and forehead of the target image
    for i in range(src_forehead.size(0)):
        src_forehead_i = src_forehead[i,:,:,:]
        src_forehead_i  = src_forehead_i .squeeze_()
        tar_forehead_i = tar_forehead[i,:,:,:]
        tar_forehead_i = tar_forehead_i.squeeze_()
        H1,W1 = torch.nonzero(src_forehead_i).size()
        H2,W2 = torch.nonzero(tar_forehead_i).size()
        if (H1 * W1) / (H2 * W2 + 0.0001) < 0.4 and (H2 * W2) >= 1000:  #
            tar_preserve[i,:,:,:] = 1 - tar_hair_and_forehead[i,:,:,:]
    soft_mask, _ = smooth_mask(tar_preserve)
    result =  srcid_taratt * soft_mask + tar * (1-soft_mask)
    return result



class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold
        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)
        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()
        return x, mask


def denormalize(x):
    if len(np.shape(x))==3:
        out = (x.permute(1,2,0).detach().cpu().numpy()*255).astype(int)
    if len(np.shape(x))==4:
        _num_img= np.shape(x)[0]
        tmp = (x.permute(0,2,3,1).detach().cpu().numpy()*255).astype(int)
        tmp = tmp[:,:,:,[2,1,0]]
        out = np.zeros((128,128*_num_img,3))
        #pdb.set_trace()
        for i in range(_num_img):
            out[:,128*i:128*(i+1),:] = tmp[i]
    return out

def save_image(x, ncol, filename):
    x = denormalize(x)
    cv2.imwrite(filename,x)

