#!#!/usr/bin/python
#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, copy, pickle, random
from datetime import datetime
import numpy as np
import cv2

### torch lib
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

### custom lib
from model_shu import U_net1
from model_shu  import U_net2
from torch.utils.checkpoint import checkpoint


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Video Super-Resolution")

    ### model options
    parser.add_argument('-nf',              type=int,     default=64,               help='#Channels in conv layer')

    ### dataset options
    parser.add_argument('-test_lr_dir',          type=str,     default='youku_00200_00249_l_frame',         help='path to low-resolution videos folder')
    #parser.add_argument('-test_hr_dir',          type=str,     default='/media/song/hdd/song/tianchi/youku_00150_00199_h_GT_frame/Youku_00150_h_GT',         help='path to high-resolution videos folder')
    parser.add_argument('-load_model_dir',       type=str,     default='checkpoints_5_16/model_660.ckpt',    help='path to model folder')
    parser.add_argument('-output_dir',       type=str,     default='output',    help='path to model folder')
    parser.add_argument('-test_mode',       type=str,     default='0',    help='path to model folder')


    opts = parser.parse_args()

    print("========================================================")
    print("=====Loading the model %s======"%opts.load_model_dir)
    print("========================================================")



    print('===Initializing model===')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = torch.device('cpu')
    torch.backends.cudnn.enabled = False

    def backwarp(img, flow):
        W = img.size(3)
        H = img.size(2) 
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        gridX = torch.tensor(gridX, requires_grad=False, device=device)
        gridY = torch.tensor(gridY, requires_grad=False, device=device)
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = gridX.unsqueeze(0).expand_as(u).float() + u
        y = gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/W - 0.5)
        y = 2*(y/H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut


    MSE_LossFn = nn.MSELoss()

    flowCoar = U_net1(opts)
    flowFine = U_net2(opts)

    with torch.no_grad():

        flowCoar.to(device)
        flowFine.to(device)
        dict1 = torch.load(opts.load_model_dir)
        flowCoar.load_state_dict(dict1['state_dict1'])
        flowFine.load_state_dict(dict1['state_dict2'])

        
        if torch.cuda.device_count() > 1:
            flowCoar = nn.DataParallel(flowCoar)
            flowFine = nn.DataParallel(flowFine)
        


        flowCoar.eval()
        flowFine.eval()
        test_videos = os.listdir(opts.test_lr_dir)
        for test_video in test_videos:
            if not os.path.exists(os.path.join(opts.output_dir, test_video)):
                os.mkdir(os.path.join(opts.output_dir, test_video))
                  


            frame_list = glob.glob(os.path.join(opts.test_lr_dir, test_video, '*.bmp'))
            if len(frame_list) == 0:
                raise Exception("No frames in %s" %opts.test_lr_dir)                    

            num_frames = len(frame_list)
            
            for t in range(num_frames):
                #t=0
                lr_frame2 = cv2.imread(os.path.join(opts.test_lr_dir, test_video,  '%04d.bmp'%(t+1)))
                print (os.path.join(opts.test_lr_dir, '%04d.bmp'%(t+1)))
                lr_frame2 = lr_frame2[0:120, 0:120, ::-1]
                lr_frame2 = np.float32(lr_frame2)/255
                lr_frame2 = torch.from_numpy(lr_frame2.transpose(2, 0, 1).astype(np.float32)).contiguous().unsqueeze(0).to(device)
                

                if t == 0:
                    lr_frame1 = torch.zeros(lr_frame2.size()).to(device)
                    frame_o1 = torch.zeros(lr_frame2.size(0), lr_frame2.size(1),  lr_frame2.size(2)*4, lr_frame2.size(3)*4).to(device)
                else:
                    lr_frame1 = cv2.imread(os.path.join(opts.test_lr_dir, test_video, '%04d.bmp'%(t)))
                    lr_frame1 = lr_frame1[:, :, ::-1]
                    lr_frame1 = lr_frame1[0:120, 0:120, ::-1]
                    lr_frame1 = np.float32(lr_frame1)/255
                    lr_frame1 = torch.from_numpy(lr_frame1.transpose(2, 0, 1).astype(np.float32)).contiguous().unsqueeze(0).to(device)

                    frame_o1 = cv2.imread(os.path.join(opts.output_dir, test_video, '%04d_1.bmp'%(t)))
                    frame_o1 =frame_o1[:, :, ::-1]
                    frame_o1 = np.float32(frame_o1)/255
                    frame_o1 = torch.from_numpy(frame_o1.transpose(2, 0, 1).astype(np.float32)).contiguous().unsqueeze(0).to(device)
                    

                flow1 = checkpoint(flowCoar, lr_frame2, lr_frame1)
                flow2 = torch.nn.functional.upsample(input=flow1, scale_factor=4, mode='bilinear')
                frame_coar = backwarp(frame_o1, flow2)
                flow1.to(device1)
                frame_coar.to(device1)

                torch.cuda.empty_cache()
                flowFine.to(device)
                flow1.to(device)
                frame_coar.to(device)
                lr_frame2 = cv2.imread(os.path.join(opts.test_lr_dir, test_video, '%04d.bmp'%(t+1)))
                print (os.path.join(opts.test_lr_dir, test_video, '%04d.bmp'%(t+1)))
                lr_frame2 = lr_frame2[0:120, 0:120, ::-1]
                lr_frame2 = np.float32(lr_frame2)/255
                lr_frame2 = torch.from_numpy(lr_frame2.transpose(2, 0, 1).astype(np.float32)).contiguous().unsqueeze(0).to(device)

         
                fineout, frame_o2_1 = checkpoint(flowFine, frame_coar, lr_frame2)
                flow3 =  fineout[:, :2, :, :] + flow2

                V_0 = torch.nn.functional.sigmoid(fineout[:, 2:3, : , :])
                V_1= 1 - V_0
                frame_fine = backwarp(frame_o1, flow3)
                frame_o2 = frame_fine * V_1 + frame_o2_1 * V_0


                device1 = torch.device('cpu')
                frame_output = frame_o2
                frame_output = frame_output.to(device1).clamp_(0, 1)
                frame_output = frame_output.detach().numpy()
                frame_output = frame_output.squeeze()
                frame_output = frame_output.transpose(1, 2, 0)
                frame_output = frame_output[:, :, ::-1]
                frame_output = (frame_output * 255.0).round()
                cv2.imwrite(os.path.join(opts.output_dir, test_video) + '/%04d_1.bmp'%(t+1), frame_output)
