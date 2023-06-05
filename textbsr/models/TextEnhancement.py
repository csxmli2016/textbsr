# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from cnstd import CnStd
import warnings
warnings.filterwarnings('ignore')
##########################################################################################
###############Text Restoration Model revised by xiaoming li
##########################################################################################

class TextRestoration(object):
    def __init__(self, ModelName='RRDB', TextModelPath='../checkpoints/bsrgan_text.pth', device='cuda'):
        self.device = device
        self.ModelName=ModelName
        self.modelText = RRDBNet(scale=4)
        self.modelText.load_state_dict(torch.load(TextModelPath)['params_ema'], strict=True)
        self.modelText.eval()

        for k, v in self.modelText.named_parameters():
            v.requires_grad = False
        self.modelText = self.modelText.to(self.device)
        torch.cuda.empty_cache()
        self.std = CnStd(model_name='db_resnet34',rotated_bbox=True, model_backend='pytorch', box_score_thresh=0.3, min_box_size=10, context=device)
        self.insize = 64


    def handle_texts(self, img, bg=None, sf=4, is_aligned=False):
        ExistText = 0
        height, width = img.shape[:2]
        bg_height, bg_width = bg.shape[:2]
        box_infos = self.std.detect(img)

        # if bg is None:
        #     bg = cv2.resize(img, (width*sf, height*sf))
        full_mask = np.zeros(bg.shape, dtype=np.float32)
        full_img = np.zeros(bg.shape, dtype=np.float32) #+255

        orig_texts, enhanced_texts = [], []


        if not is_aligned:
            for i, box_info in enumerate(box_infos['detected_texts']):
                box = box_info['box'].astype(np.int)# left top, right top, right bottom, left bottom, [width, height]
                std_cropped = box_info['cropped_img']

                h, w = std_cropped.shape[:2]
                score = box_info['score']
                if w < 10 or h < 10:
                    continue
                
                scale_wl = 0.4#0.04
                scale_hl = 0.4
                
                move_w = (box[0][0] + box[2][0]) * (scale_wl) / 2
                move_h = (box[0][1] + box[2][1]) * (scale_hl) / 2

                extend_box = box.copy()

                extend_box[0][0] = extend_box[0][0] * (1+scale_wl) - move_w
                extend_box[0][1] = extend_box[0][1] * (1+scale_hl) - move_h
                extend_box[1][0] = extend_box[1][0] * (1+scale_wl) - move_w
                extend_box[1][1] = extend_box[1][1] * (1+scale_hl) - move_h
                extend_box[2][0] = extend_box[2][0] * (1+scale_wl) - move_w
                extend_box[2][1] = extend_box[2][1] * (1+scale_hl) - move_h
                extend_box[3][0] = extend_box[3][0] * (1+scale_wl) - move_w
                extend_box[3][1] = extend_box[3][1] * (1+scale_hl) - move_h


                if w > h:
                    ref_h = self.insize
                    ref_w = int(ref_h * w / h)
                else:
                    ref_w = self.insize
                    ref_h = int(ref_w * h / w)
                ref_point = np.float32([[0,0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]])
                det_point = np.float32(extend_box)

                matrix = cv2.getPerspectiveTransform(det_point, ref_point)
                inv_matrix = cv2.getPerspectiveTransform(ref_point*sf, det_point*sf)

                cropped_img = cv2.warpPerspective(img, matrix, (ref_w, ref_h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)
                in_img = cropped_img


                LQ = transforms.ToTensor()(in_img)
                LQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ)
                LQ = LQ.unsqueeze(0)
                SQ = self.modelText(LQ.to(self.device))
                SQ = SQ * 0.5 + 0.5
                SQ = SQ.squeeze(0).permute(1, 2, 0) #.flip(2) # RGB->BGR
                SQ = np.clip(SQ.float().cpu().numpy(), 0, 1) * 255.0
                orig_texts.append(in_img)
                enhanced_texts.append(SQ)

                tmp_mask = np.ones(SQ.shape).astype(np.float)*255

                warp_mask = cv2.warpPerspective(tmp_mask, inv_matrix, (bg_width, bg_height), flags=3)
                warp_img = cv2.warpPerspective(SQ, inv_matrix, (bg_width, bg_height), flags=3)
                
                full_img = full_img + warp_img
                full_mask = full_mask + warp_mask

            index = full_mask>0
            full_img[index] = full_img[index]/full_mask[index]

            full_mask = np.clip(full_mask, 0, 1)
            kernel = np.ones((7, 7), dtype=np.uint8)
            full_mask_dilate = cv2.erode(full_mask, kernel, 1)

            full_mask_blur = cv2.GaussianBlur(full_mask_dilate, (3, 3), 0) 
            full_mask_blur = cv2.GaussianBlur(full_mask_blur, (3, 3), 0) 

            img = cv2.convertScaleAbs(bg*(1-full_mask_blur) + full_img*255*full_mask_blur)
        
        else:
            if height > width:
                up_s = self.insize / width
                ds = int(up_s * height)
                in_img = cv2.resize(img, (self.insize,ds))
            else:
                up_s = self.insize / height
                ds = int(up_s * width)
                in_img = cv2.resize(img, (ds, self.insize))
            
            LQ = transforms.ToTensor()(in_img)
            LQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ)
            LQ = LQ.unsqueeze(0)
            
            SQ = self.modelText(LQ.to(self.device))
            SQ = SQ * 0.5 + 0.5
            SQ = SQ.squeeze(0).permute(1, 2, 0)#.flip(2) # RGB->BGR
            SQ = np.clip(SQ.float().cpu().numpy(), 0, 1) * 255.0
            orig_texts.append(in_img[:,:,::-1])
            enhanced_texts.append(SQ)
            
        return img, orig_texts, enhanced_texts

#####################################################
############   ESRGAN   #############################
#####################################################
def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)
def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


if __name__ == '__main__':
    print('Test Text Crop and Alignment')
