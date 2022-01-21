# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from Code.utils.dataloader import get_loader
from Code.utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F


def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wdice = 1 - 2*(inter + 1)/(union + inter+1)
    
    delta_r = (pred[:,:,1:,:] - pred[:,:,:-1,:])**2 
    delta_c = (pred[:,:,:,1:] - pred[:,:,:,:-1])**2
    delta_pred = torch.abs(delta_r + delta_c) 
    epsilon = 1e-8 
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon))

    region_in  = torch.mean( pred * ( c_in - mask)**2 )    
    region_out = torch.mean( (1-pred) * (mask - c_out)**2 )     
    region = region_in + region_out

    dist = torch.pairwise_distance(region, edge)
    alpha = dist.min()/dist.sum()
    d=(region-inter.mean())
    l=d*(alpha*region_in + 2*alpha*region_out)*d/(512*512)
    
    loss = lenth + l.mean()

    return (wbce + wdice +loss).mean()

def train(train_loader, model, optimizer, epoch, train_save):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]   
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            edges = Variable(edges).cuda()
            # ---- rescaling the inputs (img/gt/edge) ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edges = F.upsample(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(images)
            # ---- loss function ----
            loss5 = joint_loss(lateral_map_5, gts)
            loss4 = joint_loss(lateral_map_4, gts)
            loss3 = joint_loss(lateral_map_3, gts)
            loss2 = joint_loss(lateral_map_2, gts)
            loss1 = BCE(lateral_edge, edges)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train logging ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-edge: {:.4f}, '
                  'lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(),
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    # ---- save model_lung_infection ----
    save_path = './Snapshots/save_weights1/{}/'.format(train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'PA-Net-%d.pth' % (epoch+1))
        print('[Saving Snapshot:]', save_path + 'PA-Net-%d.pth' % (epoch+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('--epoch', type=int, default=100,
                        help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--batchsize', type=int, default=4,
                        help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='set the size of training sample')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='every n epochs decay learning rate')
    parser.add_argument('--is_thop', type=bool, default=False,
                        help='whether calculate FLOPs/Params (Thop)')
    parser.add_argument('--gpu_device', type=int, default=3,
                        help='choose which GPU device you want to use')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers in dataloader. In windows, set num_workers=0')
    parser.add_argument('--net_channel', type=int, default=32)
    parser.add_argument('--n_classes', type=int, default=1,
                        help='binary segmentation when n_classes=1')
    parser.add_argument('--backbone', type=str, default='ResNest50')
    # training dataset
    parser.add_argument('--train_path', type=str,
                        default='./Dataset/TrainingSet/PA-Train/Doctor-label')
    parser.add_argument('--train_save', type=str, default=None)

    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(opt.gpu_device)
    # - please asign your prefer backbone in opt.
    if opt.backbone == 'ResNest50':
        print('Backbone loading: ResNest50')
        from Code.model_pa.PANet_ResNest50 import PA_Net
    elif opt.backbone == 'Res2Net50':
        print('Backbone loading: Res2Net50')
        from Code.model_pa.PANet_ResNest50 import PA_Net
    else:
        raise ValueError('Invalid backbone parameters: {}'.format(opt.backbone))
    model = PA_Net(channel=opt.net_channel, n_class=opt.n_classes).cuda()


    # weights file save path
        train_save = 'PA-Net'

    # ---- calculate FLOPs and Params ----
    if opt.is_thop:
        from Code.utils.utils import CalParams
        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()
        CalParams(model, x)

    # ---- load training sub-modules ----
    BCE = torch.nn.BCEWithLogitsLoss()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root,
                              batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=opt.num_workers)
    total_step = len(train_loader)

    # ---- start !! -----
    print("#"*20, "(via E-mail)\n----\n".format(opt.backbone, opt), "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, train_save)
