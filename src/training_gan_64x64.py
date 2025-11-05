from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F

from gan_64x64_models import *

G_model = Generator(nc=3, nz=args.z_dim, ngf=64,)
D_model = Discriminator(nc=3, ndf=64)
G_model.apply(weights_init)
D_model.apply(weights_init)

if args.cuda:
    G_model.cuda()
    D_model.cuda()

optimizer_g = optim.Adam(G_model.parameters(), lr=0.0002)
optimizer_d = optim.Adam(D_model.parameters(), lr=0.0002)


def load_dataset(batch_size=32):
    def set_label(attr):
        attr1 = 20
        return attr[attr1]

    transform = transforms.Compose([
        transforms.CenterCrop((160, 160)),
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CelebA(root='../data', split='train', target_type="attr", download=True,
                        target_transform=set_label, transform=transform), batch_size=batch_size, shuffle=True)
    return train_loader


def training():
    train_loader = load_dataset(args.batch_size)

    start_epoch = 0
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        print("starting epoch %d:" % epoch)
        train(epoch, train_loader)
        torch.save(G_model.state_dict(), '../models/G_Epoch_{}.pth'.format(epoch))
        torch.save(D_model.state_dict(), '../models/D_Epoch_{}.pth'.format(epoch))


def resume_training(last_epoch):
    train_loader = load_dataset(args.batch_size)

    last_cp = '../models/G_Epoch_{}.pth'.format(last_epoch)
    G_model.load_state_dict(torch.load(last_cp))
    last_cp = '../models/D_Epoch_{}.pth'.format(last_epoch)
    D_model.load_state_dict(torch.load(last_cp))

    for epoch in range(last_epoch + 1, last_epoch + args.epochs + 1):
        print("starting epoch %d:" % epoch)
        train(epoch, train_loader)
        torch.save(G_model.state_dict(), '../models/G_Epoch_{}.pth'.format(epoch))
        torch.save(D_model.state_dict(), '../models/D_Epoch_{}.pth'.format(epoch))


def train(epoch, data_loader):
    G_model.train()
    D_model.train()

    loss_func = torch.nn.BCELoss()

    batch_idx = 0
    for x, y in data_loader:
        batch_idx += 1

        # training discriminator D
        optimizer_d.zero_grad()

        batch_size = x.size()[0]

        # real data
        y_real = torch.ones(batch_size, requires_grad=False)
        y_fake = torch.zeros(batch_size, requires_grad=False)

        if args.cuda:
            x, y_real, y_fake = x.cuda(), y_real.cuda(), y_fake.cuda()

        D_output = D_model(x)

        # print(D_output.shape, y_real.shape)

        D_real_loss = loss_func(D_output, y_real)

        # fake data
        z = torch.randn((batch_size, args.z_dim, 1, 1))
        if args.cuda:
            z = z.cuda()

        G_output = G_model(z)
        D_output = D_model(G_output)
        D_fake_loss = loss_func(D_output, y_fake)

        D_loss = D_real_loss + D_fake_loss

        D_loss.backward()
        optimizer_d.step()

        # training generator G
        optimizer_g.zero_grad()

        G_output = G_model(z)
        D_output = D_model(G_output)
        G_loss = loss_func(D_output, y_real)
        G_loss.backward()
        optimizer_g.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tD_Loss: {:.6f}\tG_Loss: {:.6f}'.format(
                epoch, batch_idx, (len(data_loader)),
                100. * batch_idx / len(data_loader),
                D_loss.data, G_loss.data))



def load_g_model():
    last_cp = "../models/G_Epoch_3.pth"
    G_model.load_state_dict(torch.load(last_cp))


def rand_faces(num=5, num_imgs=1):
    load_g_model()
    m = G_model

    m.eval()
    for c in range(num_imgs):
        z = torch.randn((num*num, args.z_dim, 1, 1))

        if args.cuda:
            z = z.cuda()

        g_synthetic_img = G_model(z)

        torchvision.utils.save_image(g_synthetic_img.data, '../imgs/cgan_rand_50_%d.jpg'%c, nrow=num, padding=2)


if __name__ == '__main__':
    training()


