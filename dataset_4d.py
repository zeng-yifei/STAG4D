import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import rembg
import glob
class SparseDataset:
    def __init__(self, opt, size,device='cuda', type='train', H=256, W=256):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.size = size
        self.H = H
        self.W = W
        self.path = opt.path
        
        self.cx = self.H / 2
        self.cy = self.W / 2
        self.bg_remover=None

    def collate_ref(self,index):
        #print(index,str(index))
        file = os.path.join(self.path,'ref/{}_rgba.png'.format(str(index)))
        #print(f'[INFO] load image from {file}...')

        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        return self.input_img ,self.input_mask

                
    def collate_zero123(self,index):

        self.pattern=os.path.join(self.path,'zero123/{}_rgba/*.png'.format(str(index)))
        self.input_imgs=[]
        self.input_masks=[]
        file_list = glob.glob(self.pattern)
        #print(self.pattern,file_list)
        for files in sorted(file_list):
                    
                   
                    #print(f'[INFO] load image from {files}...')
                    img = cv2.imread(files, cv2.IMREAD_UNCHANGED)
                    if img.shape[-1] == 3:
                        if self.bg_remover is None:
                            self.bg_remover = rembg.new_session()
                        img = rembg.remove(img, session=self.bg_remover)

                    img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    img = img.astype(np.float32) / 255.0

                    self.input_mask = img[..., 3:]
                    # white bg
                    self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
                    # bgr to rgb
                    self.input_img = self.input_img[..., ::-1].copy()
                    
                    self.input_imgs.append(self.input_img)
                    self.input_masks.append(self.input_mask)
        return self.input_imgs, self.input_masks
    
    def collate(self, index):
        ref_view_batch,input_mask_batch,zero123_view_batch,zero123_masks_batch = [],[],[],[]
        for index in np.arange(self.size):
            ref_view,input_mask = self.collate_ref(index)
            zero123_view,zero123_masks = self.collate_zero123(index)
            ref_view_batch.append(ref_view)
            input_mask_batch.append(input_mask)
            zero123_view_batch.append(zero123_view)
            zero123_masks_batch.append(zero123_masks)
        return ref_view_batch, input_mask_batch,zero123_view_batch,zero123_masks_batch
    

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate,shuffle=False, num_workers=0)
        return loader

    def dataloader_d(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate_d,shuffle=False, num_workers=0)
        return loader