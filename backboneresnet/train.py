# import argparse
#
# import time
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim
# import loaddata
# import util
# import numpy as np
# import sobel
# from models import modules, net, resnet, densenet, senet
# import pytorch_ssim
# from test import test
# import time
# parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
# parser.add_argument('--epochs', default=20, type=int,
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int,
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
#                     help='initial learning rate')
# parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                     help='weight decay (default: 1e-4)')
#
#
# def define_model(is_resnet, is_densenet, is_senet):
#     if is_resnet:
#         # original_model = resnet.resnet50(pretrained = True)
#         # original_model = resnet.res2net50_26w_8s(pretrained = True)
#         original_model = resnet.res2net101_v1b_26w_4s(pretrained = True)
#         Encoder = modules.E_resnet(original_model)
#         model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
#     if is_densenet:
#         original_model = densenet.densenet161(pretrained=True)
#         Encoder = modules.E_densenet(original_model)
#         model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
#     if is_senet:
#         original_model = senet.senet154(pretrained='imagenet')
#         Encoder = modules.E_senet(original_model)
#         model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
#
#     return model
#
#
# def main():
#     global args
#     global besterror
#     besterror = 100
#     args = parser.parse_args()
#     model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
#
#     if torch.cuda.device_count() == 8:
#         model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
#         batch_size = 64
#     elif torch.cuda.device_count() == 4:
#         model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
#         batch_size = 32
#     else:
#         model = model.cuda()
#         batch_size = 8
#
#     cudnn.benchmark = True
#     optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
#
#     train_loader = loaddata.getTrainingData(batch_size)
#
#     for epoch in range(args.start_epoch, args.epochs):
#         adjust_learning_rate(optimizer, epoch)
#         train(train_loader, model, optimizer, epoch)
#         save_checkpoint({'state_dict': model.state_dict()})
#         # modelCheckpoint = torch.load('./checkpoint.pth.tar')
#         #
#         # model.load_state_dict(modelCheckpoint['state_dict'], False)
#         # test_loader = loaddata.getTestingData(1)
#         # error = test(test_loader, model, 0.25)
#         # if besterror< error:
#         #     save_checkpoint({'state_dict': model.state_dict()},)
#         # save_checkpoint({'state_dict': model.state_dict()})
#         #modelCheckpoint = torch.load('./checkpoint.pth.tar')
#         #model.load_state_dict(modelCheckpoint['state_dict'], False)
#         test_loader = loaddata.getTestingData(1)
#         error = test(test_loader, model, 0.25)
#         if besterror > error['RMSE']:
#             besterror = error['RMSE']
#             print(besterror)
#             # print(error['RMSE'])
#             save_checkpoint({'state_dict': model.state_dict()}, "res2net50_26w_8s_bestcheckpoint.pth.tar")
#
#     # save_checkpoint({'state_dict': model.state_dict()})
#
#
# def train(train_loader, model, optimizer, epoch):
#     criterion = nn.L1Loss()
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#
#     model.train()
#
#     cos = nn.CosineSimilarity(dim=1, eps=0)
#     get_gradient = sobel.Sobel().cuda()
#
#     end = time.time()
#     for i, sample_batched in enumerate(train_loader):
#         image, depth = sample_batched['image'], sample_batched['depth']
#
#         depth = depth.cuda()
#         image = image.cuda()
#         image = torch.autograd.Variable(image)
#         depth = torch.autograd.Variable(depth)
#
#         ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda()
#         ones = torch.autograd.Variable(ones)
#         optimizer.zero_grad()
#
#         output = model(image)
#
#         depth_grad = get_gradient(depth)
#         output_grad = get_gradient(output)
#         depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
#         depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
#         output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
#         output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
#
#         depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
#         output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
#
# #        depth_normal = F.normalize(depth_normal, p=2, dim=1)
# #        output_normal = F.normalize(output_normal, p=2, dim=1)
#
#         loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
#         loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
#         loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
#         loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
#         """
#         edge = np.zeros((8, 1, 114, 152))
#         depthedge = (depth * 20).cpu().numpy().astype(np.uint8)
#         for k in range(8):
#             edge[k] = cv2.Canny(depthedge[k][0], np.mean(depthedge[k][0]) // 9, np.mean(depthedge[k][0]) // 3)
#             edge[k] = cv2.dilate(edge[k], np.ones((5, 5), np.uint8))
#             edge[k] = edge[k] > 0 + 0
#             edge[k] = edge[k] * 4 + 1
#
#         edge = torch.from_numpy(edge)
#         depth = depth.cuda()
#         edge = edge.cuda()
#         output = output.double()
#         depth = depth.double()
#         output_grad_dx = output_grad_dx.double()
#         depth_grad_dx = depth_grad_dx.double()
#         output_grad_dy = output_grad_dy.double()
#         depth_grad_dy = depth_grad_dy.double()
#         output_normal = output_normal.double()
#         depth_normal = depth_normal.double()
#
#
#         loss_depth = (torch.log(torch.abs(output - depth) * edge + 0.5)).mean()
#         loss_dx = (torch.log(torch.abs(output_grad_dx - depth_grad_dx) * edge + 0.5)).mean()
#         loss_dy = (torch.log(torch.abs(output_grad_dy - depth_grad_dy) * edge + 0.5)).mean()
#         loss_normal = (torch.abs(1 - cos(output_normal, depth_normal)) * edge).mean()
#         ssim_loss = pytorch_ssim.SSIM()
#         ssim_out = -ssim_loss(depth, output)
#
#         loss = loss_depth + loss_normal + (loss_dx + loss_dy) +ssim_out
#
#         loss_depth2 = (torch.log(torch.abs(output - depth) + 0.5)).mean()
#         loss_dx2 = (torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5)).mean()
#         loss_dy2 = (torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5)).mean()
#         loss_normal2 = (torch.abs(1 - cos(output_normal, depth_normal))).mean()
#         """
#         loss = loss_depth + loss_normal + (loss_dx + loss_dy)
#         #loss = loss_depth+ loss_normal+(loss_dx+loss_dy)
#         losses.update(loss.item(), image.size(0))
#         loss.backward()
#         optimizer.step()
#
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         batchSize = depth.size(0)
#         global besterror
#         if i % 1000 == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})'
#                   .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))
#            # print(ssim_out)
#
#
#
# def adjust_learning_rate(optimizer, epoch):
#     lr = args.lr * (0.1 ** (epoch // 5))
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#
# class AverageMeter(object):
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# def save_checkpoint(state, filename='res2net50_26w_8s_checkpoint.pth.tar'):
#     torch.save(state, filename)
#
#
# if __name__ == '__main__':
#     main()
import argparse

import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
import sobel
from models import modules, net, resnet, densenet, senet
from wavelet_ssim_loss import Wnormal, Wgrad, WSloss, Wdepth
import pytorch_ssim

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
from test import test


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        # original_model = resnet.resnet50(pretrained=True)
        # original_model = resnet.res2net50_26w_8s(pretrained=True)
        # original_model = resnet.res2net50_26w_6s(pretrained=True)
        original_model = resnet.res2net101_v1b_26w_4s(pretrained=True)

        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])

    return model


def main():
    global args
    args = parser.parse_args()
    global besterror
    besterror = 100
    #定义模型
    model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
    #显卡对应的batch—size
    if torch.cuda.device_count() == 8:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        batch_size = 64
    elif torch.cuda.device_count() == 4:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        batch_size = 32
    else:
        #咱们跑得
        model = model.cuda()
        batch_size = 8

    cudnn.benchmark = True
    #优化器、学习率、衰减率
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    train_loader = loaddata.getTrainingData(batch_size)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, epoch)
        test_loader = loaddata.getTestingData(1)
        error = test(test_loader, model, 0.25)
        # print(error)

        if besterror > error['RMSE']:
            besterror = error['RMSE']
            save_checkpoint({'state_dict': model.state_dict()}, "res2net_ps_edge_walve_8_7_best.pth.tar")

    save_checkpoint({'state_dict': model.state_dict()})


def train(train_loader, model, optimizer, epoch):
    criterion = nn.L1Loss()
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda()
        image = image.cuda()
        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)

        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)
        optimizer.zero_grad()

        # output = model(image)
        output= model(image)

        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        # depth_normal = F.normalize(depth_normal, p=2, dim=1)
        # output_normal = F.normalize(output_normal, p=2, dim=1)
        # 损失函数无改变
        # loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
        # loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        # loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        # loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
        # 原损失函数+walve
        wsgrad = Wgrad()
        wsnormal = Wnormal()
        wdepth = Wdepth()
        loss_depth = wdepth(output, depth)
        loss_dx = wsgrad(output_grad_dx, depth_grad_dx)
        loss_dy = wsgrad(output_grad_dy, depth_grad_dy)
        loss_normal = wsnormal(output_normal, depth_normal)
        # ssim损失函数
        # ssim_loss = pytorch_ssim.SSIM()
        # loss_ssim = -ssim_loss(depth, output)
        # loss = loss_depth + loss_normal + (loss_dx + loss_dy)+loss_ssim
        #print(loss_wsloss)
        # loss1 = loss_depth + loss_normal + (loss_dx + loss_dy)
        loss = loss_depth + loss_normal + (loss_dx + loss_dy)

        losses.update(loss.item(), image.size(0))
        # losses1.update(loss1.item(), image.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        batchSize = depth.size(0)
        global besterror
        if i % 1000 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t '
                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, loss1=losses1))


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
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


def save_checkpoint(state, filename='res2net_ps_edge_walve_8_7.pth.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
    main()