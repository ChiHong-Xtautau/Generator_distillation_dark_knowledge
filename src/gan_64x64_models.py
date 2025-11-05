import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super().__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(nz, ngf * 8,
                                         kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4,
                                         4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2,
                                         4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(ngf * 2, ngf,
                                         4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(ngf, nc,
                                         4, 2, 1, bias=False)
        # Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        print("x shape", x.shape)
        x = F.relu(self.bn1(self.tconv1(x)))
        print("x shape", x.shape)
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv5(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.cv1 = nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False)  # (3, 64, 64) -> (64, 32, 32)
        self.cv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)  # (64, 32, 32) -> (128, 16, 16)
        self.bn2 = nn.BatchNorm2d(ndf * 2)  # spatial batch norm is applied on num of channels
        self.cv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)  # (128, 16, 16) -> (256, 8, 8)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.cv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)  # (256, 8, 8) -> (512, 4, 4)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.cv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)  # (512, 4, 4) -> (1, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.cv1(x))
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.cv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.cv4(x)), 0.2, True)
        x = torch.sigmoid(self.cv5(x))
        return x.view(-1, 1).squeeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


parser = argparse.ArgumentParser(description='PyTorch CGAN')
parser.add_argument('--class-num', type=int, default=4, metavar='N',
                    help='number of classes')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--z-dim', type=int, default=100, metavar='N',
                    help='latent variable size')
parser.add_argument('--off-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--distill-method', type=str, default='img',
                    help='img or dark')
parser.add_argument('--distill-batches', type=int, default=2000,
                    help='number of batches per epoch for distillation')

args = parser.parse_args()
args.cuda = not args.off_cuda and torch.cuda.is_available()

