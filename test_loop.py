import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import time
from tqdm import tqdm
import torchvision.transforms as transforms
from models.mdcnet import MDCNet
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import imageio
import numpy as np


parser = argparse.ArgumentParser(description='MDCNet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--datapath', default='/home/bsplab/Documents/dataset_kitti/train/2011_09_26_drive_0011_sync',
                    help='datapath')
parser.add_argument('--output_dir', type=str, default='output', help='output dir')
parser.add_argument('--loadmodel', type=str, default='results/kitti15_mdcnet/checkpoint.tar', help='checkpoint')
args = parser.parse_args()

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

filenames = sorted(os.listdir(os.path.join(args.datapath, 'RGB_left')))

model = MDCNet(args)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'disp'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'depth'), exist_ok=True)
model = nn.DataParallel(model).cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    imgL = imgL.cuda()
    imgR = imgR.cuda()

    with torch.no_grad():
        _, outputs = model(imgL,imgR)
        output = outputs[-1]  # high scale fine disparity 
    output = torch.squeeze(output).data.cpu().numpy()
    return output

def main():
    height, width = cv2.imread(os.path.join(args.datapath, 'RGB_left', filenames[0])).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(os.path.join(args.output_dir, 'demo.mp4'), fourcc, 20.0, (width*2//3, height*2//3*2))
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(**normal_mean_var)]
    )    
    index_pbar = tqdm(filenames)
    demo_list = []
    for inx in index_pbar:
        test_left_img = os.path.join(args.datapath, 'RGB_left', inx)
        test_right_img = os.path.join(args.datapath, 'RGB_right', inx)
        imgL_o = Image.open(test_left_img).convert('RGB')
        imgR_o = Image.open(test_right_img).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)

        # pad to width and hight to 48 times
        if imgL.shape[1]%48 != 0:
            times = imgL.shape[1] // 48
            top_pad = (times+1)*48 - imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 48 != 0:
            times = imgL.shape[2] // 48
            right_pad = (times+1)*48 - imgL.shape[2]
        else:
            right_pad = 0

        imgL = F.pad(imgL,(0, right_pad, top_pad, 0)).unsqueeze(0)
        imgR = F.pad(imgR,(0, right_pad, top_pad, 0)).unsqueeze(0)

        start_time = time.time()
        pred_disp = test(imgL, imgR)
        index_pbar.set_description('time = %.2f' %(time.time() - start_time))

        if top_pad != 0 or right_pad != 0:
            img = pred_disp[top_pad:, :-right_pad]
        else:
            img = pred_disp
        depth = width_to_focal[width] * 0.54 / img
        # save depth and disparity image
        plt.imsave(os.path.join(args.output_dir, 'depth', inx), depth, cmap='plasma')
        plt.imsave(os.path.join(args.output_dir, 'disp', inx[:-4]+'_disp.png'), img, cmap='plasma')
        disp_img = cv2.imread(os.path.join(args.output_dir, 'disp', inx[:-4]+'_disp.png'))
        ori_img = cv2.imread(os.path.join(args.datapath, 'RGB_left', inx))
        # resize image to 2/3
        disp_img_resized = cv2.resize(disp_img, (width*2//3, height*2//3), interpolation=cv2.INTER_AREA)
        ori_img_resized = cv2.resize(ori_img, (width*2//3, height*2//3), interpolation=cv2.INTER_AREA)
        demo_list.append(np.vstack((ori_img_resized[:, :, ::-1], disp_img_resized[:, :, ::-1])))
        out.write(np.vstack((ori_img_resized, disp_img_resized)))
    imageio.mimsave(os.path.join(args.output_dir, 'demo.gif'), demo_list, 'GIF', duration=0.1)
    out.release()


if __name__ == '__main__':
    main()
