# import argparse
# import torch
# import torch.nn.parallel
#
# from models import modules, net, resnet, densenet, senet
# import numpy as np
# import loaddata_demo as loaddata
# import pdb
#
# import matplotlib.image
# import matplotlib.pyplot as plt
# plt.set_cmap("jet")
#
# from thop import profile
# # from torchstat import stat
# def define_model(is_resnet, is_densenet, is_senet):
#     if is_resnet:
#         # original_model = resnet.resnet50(pretrained = True)
#         original_model = resnet.res2net50_26w_8s(pretrained=True)
#
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
#     model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
#     model = model.cuda()
#     # model.load_state_dict(torch.load('./c'))
#     modelCheckpoint = torch.load('./0310bestresnet50.pth.tar')
#     model.load_state_dict(modelCheckpoint['state_dict'], False)
#
#     nyu2_loader = loaddata.readNyu2('data/demo/img_nyu2.png')
#     total = sum([param.nelement() for param in model.parameters()])
#     print("Number of parameter: %.2fM" % (total / 1e6))
#     test(nyu2_loader, model)
#
#
#
#     model.eval()
#     input = torch.randn(1, 3, 320, 224).cuda()
#     flops, params = profile(model, inputs=(input,))
#     print('flops:', flops)
#     print('params:', params)
#     print("Number of parameter: %.2fM" % (params / 1e6))
#     # stat(model, (3, 320, 224))
#
# def test(nyu2_loader, model):
#     for i, image in enumerate(nyu2_loader):
#         image = torch.autograd.Variable(image, volatile=True).cuda()
#         out = model(image)
#
#         matplotlib.image.imsave('data/demo/out.png', out.view(out.size(2),out.size(3)).data.cpu().numpy())
#
# if __name__ == '__main__':
#     main()
import argparse
import torch
import torch.nn.parallel
# from thop import profile

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb

import matplotlib.image
import matplotlib.pyplot as plt

plt.set_cmap("jet")


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        # original_model = resnet.res2net101_v1b_26w_4s(pretrained=True)

        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])

    return model


def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)

    model = model.cuda()  # 0313bestresnet50 0313-1checkpoint.pth.tar 312checkpoint.pth.tar
    modelCheckpoint = torch.load('./0403bestresnet101.pth.tar')
    model.load_state_dict(modelCheckpoint['state_dict'], False)
    # model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
    # model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load('./pretrained_model/model_senet'))
    # modelCheckpoint = torch.load('./checkpoint.pth.tar')
    # model.load_state_dict(modelCheckpoint['state_dict'],False)
    # model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
    # model = model.cuda()
    # # model.load_state_dict(torch.load('./c'))
    # model.load_state_dict(torch.load('./0313-1checkpoint.pth.tar')['state_dict'], False)
    # # input = torch.randn(1, 3, 320, 224)
    # flops, params = profile(model, inputs=(input,))
    # print('flops:', flops)
    # print('params:', params)
    # print("Number of parameter: %.2fM" % (params / 1e6))
    # # stat(model, (3, 320, 224))
    # model.eval()
    # input = torch.randn(1, 3, 320, 224).cuda()
    # flops, params = profile(model, inputs=(input,))
    # print('flops:', flops)
    # print('params:', params)
    # print("Number of parameter: %.2fM" % (params / 1e6))
    # stat(model, (3, 320, 224))
    # params = list(model.named_parameters())
    # print(params.__len__())
    # # print(params[-1])
    # print(params)
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of  parameter:%.2fM" % (total / 1e6))
    # model.eval()
    # nyu2_loader1 = loaddata.readNyu2('data/demo/00055_colors.png')
    # nyu2_loader2 = loaddata.readNyu2('data/demo/00056_colors.png') #resnet50
    # nyu2_loader3 = loaddata.readNyu2('data/demo/00061_colors.png')
    # nyu2_loader4 = loaddata.readNyu2('data/demo/00062_colors.png')
    # nyu2_loader5 = loaddata.readNyu2('data/demo/00078_colors.png')
    # nyu2_loader6 = loaddata.readNyu2('data/demo/00083_colors.png')# resnet50
    # nyu2_loader7 = loaddata.readNyu2('data/demo/00087_colors.png')
    # nyu2_loader8 = loaddata.readNyu2('data/demo/00088_colors.png') #resnet50
    # nyu2_loader9 = loaddata.readNyu2('data/demo/00117_colors.png') #resnet50
    # nyu2_loader10 = loaddata.readNyu2('data/demo/ClassRoom1641.jpg') #resnet50
    # nyu2_loader10 = loaddata.readNyu2('data/demo/01400_colors.png')
    nyu2_loader10 = loaddata.readNyu2('data/demo/01289_colors.png')
    # x1 = plt.imread('data/demo/00055_depth.png')
    # x2 = plt.imread('data/demo/00056_depth.png')
    # x3 = plt.imread('data/demo/00061_depth.png')
    # x4 = plt.imread('data/demo/00062_depth.png')
    # x5 = plt.imread('data/demo/00078_depth.png')
    # x6 = plt.imread('data/demo/00083_depth.png')
    # x7 = plt.imread('data/demo/00087_depth.png')
    # x8 = plt.imread('data/demo/00088_depth.png')
    # x9 = plt.imread('data/demo/00117_depth.png')
    # x9 = plt.imread('data/demo/00117_depth.png')
    # x9 = plt.imread('data/demo/01289_depth.png')
    # x9 = plt.imread('data/demo/00410_depth.png')
    # x1 = torch.from_numpy(x1)
    # x1 = torch.from_numpy(x1)
    # x2 = torch.from_numpy(x2)
    # x3 = torch.from_numpy(x3)
    # x4 = torch.from_numpy(x4)
    # x5 = torch.from_numpy(x5)
    # x6 = torch.from_numpy(x6)
    # x7 = torch.from_numpy(x7)
    # x8 = torch.from_numpy(x8)
    # x9 = torch.from_numpy(x9)
    # matplotlib.image.imsave('data/demo/00055_colors_realdepth.png', x1.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/00056_colors_realdepth.png', x2.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/00061_colors_realdepth.png', x3.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/00062_colors_realdepth.png', x4.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/00078_colors_realdepth.png', x5.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/00083_colors_realdepth.png', x6.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/00087_colors_realdepth.png', x7.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/00088_colors_realdepth.png', x8.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/00117_colors_realdepth.png', x9.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/001641_colors_realdepth.png', x9.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/01400_colors_realdepth.png', x9.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/01289_colors_realdepth.png', x9.view(480, 640).data.cpu().numpy())
    # matplotlib.image.imsave('data/demo/00410_colors_realdepth.png', x9.view(480, 640).data.cpu().numpy())
    # test(nyu2_loader1, model)
    # test(nyu2_loader2, model)
    # test(nyu2_loader3, model)
    # test(nyu2_loader4, model)
    # test(nyu2_loader5, model)
    # test(nyu2_loader6, model)
    # test(nyu2_loader7, model)
    # test(nyu2_loader8, model)
    # test(nyu2_loader9, model)
    test(nyu2_loader10, model)


def test(nyu2_loader, model):
    for i, image in enumerate(nyu2_loader):
        image = torch.autograd.Variable(image, volatile=True).cuda()
        out = model(image)

        matplotlib.image.imsave('data/demo/01289_senet154_ps_wave.png', out.view(out.size(2), out.size(3)).data.cpu().numpy())


if __name__ == '__main__':
    main()
