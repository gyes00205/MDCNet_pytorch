import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import KITTILoader as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn
from models.mdcnet import MDCNet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser(description='MDCNet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.5, 0.7, 1.])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/home/bsplab/Documents/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=4,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/finetune_kitti2015_mdcnet_multi_scale_warp_HR',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--pretrained', type=str, default='results/pretrained_mdcnet/checkpoint22.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default='dataset/KITTI2015_val.txt')
parser.add_argument('--evaluate', action='store_true')

args = parser.parse_args()

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls
elif args.datatype == 'other':
    from dataloader import diy_dataset as ls


def main():
    global args
    best_D1 = 1.0
    log = logger.setup_logger(os.path.join(args.save_path, 'training.log'))

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
        args.datapath,log, args.split_file)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = MDCNet(args)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')
    cudnn.benchmark = True
    start_full_time = time.time()
    if args.evaluate:
        test(TestImgLoader, model, log)
        return

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        log.info('This is {}-th epoch'.format(epoch))
        adjust_learning_rate(optimizer, epoch)

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = os.path.join(args.save_path, 'checkpoint.tar')
        

        D1_err = test(TestImgLoader, model, log, epoch)
        if D1_err < best_D1:
            best_D1 = D1_err
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, savefilename)


    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(tqdm(dataloader)):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        _, outputs = model(imgL, imgR)

        
        num_out = len(outputs)

        outputs = [torch.squeeze(output, 1) for output in outputs]
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], reduction='mean')
                for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item()/args.loss_weights[idx])
            writer.add_scalar(f'Train Step/fine loss/stage {idx}', losses[idx].val, epoch*length_loader+batch_idx)

        # if batch_idx % args.print_freq:
        #     info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
        #     info_str = '\t'.join(info_str)

        #     log.info('Epoch{} [{}/{}] {}'.format(
        #         epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)
    for idx in range(stages):
        writer.add_scalar(f'KITTI{args.datatype} Train Epoch/fine loss/stage {idx}', losses[idx].avg, epoch)


def test(dataloader, model, log, epoch=0):

    stages = 3
    fine_D1s = [AverageMeter() for _ in range(stages)]
    rough_D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(tqdm(dataloader)):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
        if imgL.shape[2] % 48 != 0:
            times = imgL.shape[2]//48
            top_pad = (times+1)*48 - imgL.shape[2]
        else:
            top_pad = 0

        if imgL.shape[3] % 48 != 0:
            times = imgL.shape[3]//48
            right_pad = (times+1)*48-imgL.shape[3]
        else:
            right_pad = 0
        imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

        with torch.no_grad():
            rough_outputs, fine_outputs = model(imgL, imgR)
            for x in range(stages):
                fine_output = torch.squeeze(fine_outputs[x], 1)
                rough_output = torch.squeeze(rough_outputs[x], 1)
                if top_pad != 0:
                    fine_output = fine_output[:, top_pad:, :]
                    rough_output = rough_output[:, top_pad:, :]
                if right_pad != 0:
                    fine_output = fine_output[:, :, :-right_pad]
                    rough_output = rough_output[:, :, :-right_pad]
                fine_D1s[x].update(error_estimating(fine_output, disp_L).item())
                rough_D1s[x].update(error_estimating(rough_output, disp_L).item())

    for idx in range(stages):
        writer.add_scalar(f'KITTI{args.datatype} Val Epoch/fine map D1/stage {idx}', fine_D1s[idx].avg, epoch)
        writer.add_scalar(f'KITTI{args.datatype} Val Epoch/rough map D1/stage {idx}', rough_D1s[idx].avg, epoch)
    
    info_str = ', '.join(['Stage {}={:.4f}'.format(x, fine_D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error of fine map = ' + info_str)
    info_str = ', '.join(['Stage {}={:.4f}'.format(x, rough_D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error of rough map = ' + info_str)

    return fine_D1s[-1].avg


def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    writer = SummaryWriter(args.save_path)
    main()
