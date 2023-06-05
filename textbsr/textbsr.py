
import torch
from torch.hub import download_url_to_file, get_dir
from .models.TextEnhancement import TextRestoration as TextRestoration
from .utils.utils_image import get_image_paths, imread_uint
import cv2 
import numpy as np
import os.path
import torch.nn.functional as F
import time
import argparse
import os.path as osp
from urllib.parse import urlparse

pretrain_model_url = {
    'x4': 'https://github.com/csxmli2016/textbsr/releases/download/0.2.0/bsrgan_text_256.pth',
}

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

def bsr(input_path=None, bg_path=None, output_path=None, aligned=False, save_text=False, device=None):
    if input_path is None:
        exit('input image path is none. Please see our document')
    if output_path is None:
        TIMESTAMP = time.strftime("%m-%d_%H-%M", time.localtime())
        if input_path[-1] == '/' or input_path[-1] == '\\':
            input_path = input_path[:-1]
        output_path = osp.join(input_path+'_'+TIMESTAMP+'_BSRGAN-Text')
    os.makedirs(output_path, exist_ok=True)

    lq_imgs = []
    sq_imgs = []
    lq_imgs = get_image_paths(input_path)
    if len(lq_imgs) ==0:
        exit('No Image in the LR path.')
    if bg_path is not None:
        sq_imgs = get_image_paths(bg_path)
        if len(sq_imgs) != len(lq_imgs):
            exit('The LQ path has {} images, while the SR path has {} ones. Please check whether the two paths are consistent.'.format(len(lq_imgs), len(sq_imgs)))

    scale_factor = 4 # upsample scale factor for the final output, fixed
    if device == None or device == 'gpu':
        use_cuda = torch.cuda.is_available()
    if device == 'cpu':
        use_cuda = False
    device = torch.device('cuda' if use_cuda else 'cpu')
    weight_path = load_file_from_url(pretrain_model_url['x4'])

    TextModel = TextRestoration(ModelName='RRDB', TextModelPath=weight_path, device=device)

    print('{:>25s} : {:s}'.format('Model Name', 'BSRGAN'))
    if use_cuda:
        print('{:>25s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
    else:
        print('{:>25s} : {:s}'.format('GPU ID', 'No GPU is available. Use CPU instead.'))
    torch.cuda.empty_cache()

    L_path = input_path
    E_path = output_path # save path
    print('{:>25s} : {:s}'.format('Input Path', L_path))
    print('{:>25s} : {:s}'.format('Output Path', E_path))
    print('{:>25s} : {:s}'.format('Background SR Path', bg_path if bg_path else 'None'))
    if aligned:
        print('{:>25s} : {:s}'.format('Image Details', 'Aligned Text Layout. No text detection is used.'))
    else:
        print('{:>25s} : {:s}'.format('Image Details', 'UnAligned Text Image. It will crop text region using CnSTD, restore, and paste results back.'))
    print('{:>25s} : {:s}'.format('Save LR & SR text layout', 'True' if save_text else 'False'))

    idx = 0    
    for img in lq_imgs:
        ####################################
        #####(1) Read Image
        ####################################
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        print('{:>20s} {:04d} : x{:<d} --> {:<s}'.format('Restoring ', idx, scale_factor, img_name+ext))

        img_L = imread_uint(img, n_channels=3) #RGB 0~255
        height_L, width_L = img_L.shape[:2]

        # img_L = cv2.resize(img_L, (width_L//2, height_L//2))
        # height_L, width_L = img_L.shape[:2]

        width_S, height_S = 0, 0
        

        if len(sq_imgs) > 0:
            sq_img = sq_imgs[idx-1]
            img_E = imread_uint(sq_img, n_channels=3)
            width_S = img_E.shape[1]
            height_S = img_E.shape[0]

        else:
            img_E = img_L
        img_E = cv2.resize(img_E, (width_L*scale_factor, height_L*scale_factor), interpolation = cv2.INTER_AREA)
        
        
        #########################################################
        #####(2) Restore Each Region and Paste to the whole image
        #########################################################
        SQ, ori_texts, en_texts  = TextModel.handle_texts(img=img_L, bg=img_E, sf=scale_factor, is_aligned=aligned)
        if not aligned:
            if width_S == 0 or height_S == 0:
                width_S = (width_L * scale_factor)
                height_S = (height_L * scale_factor)
            SQ = cv2.resize(SQ.astype(np.float32), (width_S, height_S), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(E_path, img_name +'_BSRGANText.png'), SQ[:,:,::-1])
        else:
            cv2.imwrite(os.path.join(E_path, img_name +'_BSRGANText.png'), en_texts[0][:,:,::-1])

        ####################################
        #####(3) Save Cropped Results
        ####################################
        if save_text and not aligned:
            for m, (et, ot) in enumerate(zip(en_texts, ori_texts)): ##save each face region
                w, h, c = et.shape
                ot = cv2.resize(ot, (h, w))
                cv2.imwrite(os.path.join(E_path, img_name +'_patch_{}.png'.format(m)), np.hstack((ot[:,:,::-1], et[:,:,::-1])))


def textbsr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default='./testsets/LQ', help='The lr text image path')
    parser.add_argument('-b', '--bg_path', type=str, default=None, help='The background sr path, default:None')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='The save path for text sr result')
    parser.add_argument('-a', '--aligned', action='store_true', help='The input text image contains only text region or not, default:False')
    parser.add_argument('-s', '--save_text', action='store_true', help='Save the LR and SR text layout or not, default:False')
    parser.add_argument('-d', '--device', type=str, default=None, help='using cpu or gpu')
    # try:
    args = parser.parse_args()
    bsr(args.input_path, args.bg_path, args.output_path, args.aligned, args.save_text, args.device)
    # except:
    #     parser.print_help()


if __name__ == '__main__':
    textbsr()