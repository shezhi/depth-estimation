import torch.nn as nn
from dwt import dwt
import pytorch_ssim

ssim_loss = pytorch_ssim.SSIM()

#ssim
class WSloss(nn.Module):
    def __init__(self):
        super(WSloss, self).__init__()

    def forward(self, x, y, r=0.7):
        loss = 0
        loss -= ssim_loss(x, y)
        l, m, h = 1, 1, 1
        for i in range(4):
            # l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
            x0, x1, x2 = dwt(x)
            y0, y1, y2 = dwt(y)
            loss = loss - ssim_loss(x1, y1) * 2 * m - ssim_loss(x2, y2) * h
            x, y = x0, y0
        loss -= ssim_loss(x0, y0) * l
        return loss
import torch





class Wdepth(nn.Module):
    def __init__(self):
        super(Wdepth, self).__init__()

    def forward(self, x, y, r=0.7):
        loss = 0
        loss = torch.log(torch.abs(x - y) + 0.5).mean()
        l, m, h = 1, 1, 1
        for i in range(4):
            # l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
            x0, x1, x2 = dwt(x)
            y0, y1, y2 = dwt(y)
            # print(y1)
            loss = loss +torch.log(torch.abs(x1 - y1) + 0.5).mean() * 2 * m + torch.log(torch.abs(x2 - y2) + 0.5).mean() * h
            x, y = x0, y0
        loss = loss + torch.log(torch.abs(x0 - y0) + 0.5).mean() * l
        return loss

class Wgrad(nn.Module):
    def __init__(self):
        super(Wgrad, self).__init__()

    def forward(self, x, y, r=0.7):
        loss = 0
        loss = torch.log(torch.abs(x - y) + 0.5).mean()
        l, m, h = 1, 1, 1
        for i in range(4):
            l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
            x0, x1, x2 = dwt(x)
            y0, y1, y2 = dwt(y)
            # print(y1)
            loss = loss +torch.log(torch.abs(x1 - y1) + 0.5).mean() * 2 * m + torch.log(torch.abs(x2 - y2) + 0.5).mean() * h
            x, y = x0, y0
        loss = loss + torch.log(torch.abs(x0 - y0) + 0.5).mean() * l
        return loss

class Wnormal(nn.Module):
    def __init__(self):
        super(Wnormal, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, x, y, r=0.7):
        loss = 0
        loss = torch.abs(1 - self.cos(x, y)).mean()
        l, m, h = 1, 1, 1
        for i in range(4):
            l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
            x0, x1, x2 = dwt(x)
            y0, y1, y2 = dwt(y)
            # print(y1)
            loss = loss +torch.abs(1 - self.cos(x1, y1)).mean() * 2 * m + torch.abs(1 - self.cos(x2, y2)).mean() * h
            x, y = x0, y0
        loss = loss + torch.abs(1 - self.cos(x0, y0)).mean() * l
        return loss