import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
import utils.logger as logger
from models.mdcnet import MDCNet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


parser = argparse.ArgumentParser(description='MDCNet with Flyingthings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.5, 0.7, 1.])
parser.add_argument('--map_loss_weights', type=float, nargs='+', default=[0.5, 2.])
parser.add_argument('--datapath', default='/media/bsplab/bbf8a27e-33b8-4816-9500-17ff7e724297/SceneFlowData/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=24,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=4,
                    help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/pretrained_mdcnet_warp_HR',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')


args = parser.parse_args()


def main():
    global args

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
        args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = logger.setup_logger(os.path.join(args.save_path, 'training.log'))
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = MDCNet(args)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')

    start_full_time = time.time()
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        adjust_learning_rate(optimizer, epoch)
        log.info('This is {}-th epoch'.format(epoch))

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = os.path.join(args.save_path, f'checkpoint{epoch:02d}.tar')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)
        test(TestImgLoader, model, log, epoch)

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3
    rough_losses = [AverageMeter() for _ in range(stages)]
    fine_losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()
    # io_start_time = time.time()
    for batch_idx, (imgL, imgR, disp_L) in enumerate(tqdm(dataloader)):
        # io_end_time = time.time()
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        optimizer.zero_grad()
        mask = disp_L < args.maxdisp
        if mask.float().sum() == 0:
            continue
        mask.detach_()
        rough_outputs, fine_outputs = model(imgL, imgR)
        rough_outputs = [torch.squeeze(output, 1) for output in rough_outputs]
        fine_outputs = [torch.squeeze(output, 1) for output in fine_outputs]
        rough_loss = [args.loss_weights[x] * F.smooth_l1_loss(rough_outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(stages)]
        fine_loss = [args.loss_weights[x] * F.smooth_l1_loss(fine_outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(stages)]
        rough_total_loss = sum(rough_loss) * args.map_loss_weights[0]
        fine_total_loss = sum(fine_loss) * args.map_loss_weights[1]
        total_loss = rough_total_loss + fine_total_loss
        total_loss.backward()
        optimizer.step()

        for idx in range(stages):
            rough_losses[idx].update(rough_loss[idx].item()/args.loss_weights[idx])
            writer.add_scalar(f'Train Step/rough loss/stage {idx}', rough_losses[idx].val, epoch*length_loader+batch_idx)
        for idx in range(stages):
            fine_losses[idx].update(fine_loss[idx].item()/args.loss_weights[idx])
            writer.add_scalar(f'Train Step/fine loss/stage {idx}', fine_losses[idx].val, epoch*length_loader+batch_idx)
        # if batch_idx % args.print_freq:
        #     info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]
        #     info_str = ' '.join(info_str)

            # log.info('Epoch{} [{}/{}] io time: {:.2f} {}'.format(
            #     epoch, batch_idx, length_loader, io_end_time - io_start_time, info_str))
        # io_start_time = time.time()
    for idx in range(stages):
        writer.add_scalar(f'Train Epoch/rough loss/stage {idx}', rough_losses[idx].avg, epoch)
    for idx in range(stages):
        writer.add_scalar(f'Train Epoch/fine loss/stage {idx}', fine_losses[idx].avg, epoch)
    # info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    # log.info('Average train loss = ' + info_str)


def test(dataloader, model, log, epoch=0):

    stages = 3
    EPEs = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(tqdm(dataloader)):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        if imgL.shape[2]%48 != 0:
            times = imgL.shape[2] // 48
            top_pad = (times+1)*48 - imgL.shape[2]
        else:
            top_pad = 0

        if imgL.shape[3]%48 != 0:
            times = imgL.shape[3] // 48
            right_pad = (times+1)*48 - imgL.shape[3]
        else:
            right_pad = 0
        imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

        mask = disp_L < args.maxdisp
        with torch.no_grad():
            _, outputs = model(imgL, imgR)
            for x in range(stages):
                if len(disp_L[mask]) == 0:
                    EPEs[x].update(0)
                    continue
                output = torch.squeeze(outputs[x], 1)
                if top_pad != 0:
                    output = output[:, top_pad:, :]
                if right_pad != 0:
                    output = output[:, :, :-right_pad]
                EPEs[x].update((output[mask] - disp_L[mask]).abs().mean())

        info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(2, EPEs[2].val, EPEs[2].avg)])

        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

    for idx in range(stages):
        writer.add_scalar(f'Val Epoch/fine map EPE/stage {idx}', EPEs[idx].avg, epoch)
    info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPEs[x].avg) for x in range(stages)])
    log.info('Average test EPE = ' + info_str)

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


def adjust_learning_rate(optimizer, epoch):
    if epoch < 16:
        lr = args.lr
    elif epoch < 18:
        lr = args.lr / 2.0
    else:
        lr = args.lr / 4.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    writer = SummaryWriter(args.save_path)
    main()
