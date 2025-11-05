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

from src.cgan_64x64_models import *

G_model = Generator(nc=3, nz=args.z_dim, n_class=2, ngf=64)
D_model = Discriminator(nc=3, n_class=2, ndf=64)
G_model.apply(weights_init)
D_model.apply(weights_init)

SG_model = GeneratorSmall(nc=3, nz=args.z_dim, n_class=2, ngf=64)
SD_model = Discriminator(nc=3, n_class=2, ndf=64)
SG_model.apply(weights_init)
SD_model.apply(weights_init)


if args.cuda:
    G_model.cuda()
    D_model.cuda()
    SG_model.cuda()
    SD_model.cuda()

optimizer_g = optim.Adam(G_model.parameters(), lr=0.0002)
optimizer_d = optim.Adam(D_model.parameters(), lr=0.0002)

optimizer_sg = optim.Adam(SG_model.parameters(), lr=0.0002)
optimizer_sd = optim.Adam(SD_model.parameters(), lr=0.0002)


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
        datasets.CelebA(root='./data', split='train', target_type="attr", download=True,
                        target_transform=set_label, transform=transform), batch_size=batch_size, shuffle=True)
    return train_loader


def training():
    train_loader = load_dataset(args.batch_size)

    start_epoch = 0
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        print("starting epoch %d:" % epoch)
        train(epoch, train_loader)
        torch.save(G_model.state_dict(), './models/G_Epoch_{}.pth'.format(epoch))
        torch.save(D_model.state_dict(), './models/D_Epoch_{}.pth'.format(epoch))


def resume_training(last_epoch):
    train_loader = load_dataset(args.batch_size)

    last_cp = './models/G_Epoch_{}.pth'.format(last_epoch)
    G_model.load_state_dict(torch.load(last_cp))
    last_cp = './models/D_Epoch_{}.pth'.format(last_epoch)
    D_model.load_state_dict(torch.load(last_cp))

    for epoch in range(last_epoch + 1, last_epoch + args.epochs + 1):
        print("starting epoch %d:" % epoch)
        train(epoch, train_loader)
        torch.save(G_model.state_dict(), './models/G_Epoch_{}.pth'.format(epoch))
        torch.save(D_model.state_dict(), './models/D_Epoch_{}.pth'.format(epoch))


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
        y_one_hot = torch.zeros(batch_size, args.class_num, 1, 1)
        y_one_hot.scatter_(1, y.view(batch_size, 1, 1, 1), 1)

        y_real = torch.ones(batch_size, requires_grad=False)
        y_fake = torch.zeros(batch_size, requires_grad=False)

        if args.cuda:
            x, y_real, y_fake, y_one_hot = x.cuda(), y_real.cuda(), y_fake.cuda(), y_one_hot.cuda()

        D_output = D_model(x, y_one_hot)


        D_real_loss = loss_func(D_output, y_real)

        # fake data: rand 0-1, randn can be negative
        y_fake_label = (torch.rand(batch_size) * args.class_num).type(torch.LongTensor)

        y_one_hot_fake = torch.zeros(batch_size, args.class_num, 1, 1)
        y_one_hot_fake.scatter_(1, y_fake_label.view(batch_size, 1, 1, 1), 1)

        z = torch.randn((batch_size, args.z_dim, 1, 1))
        if args.cuda:
            z = z.cuda()
            y_one_hot_fake = y_one_hot_fake.cuda()

        G_output = G_model(z, y_one_hot_fake)
        D_output = D_model(G_output, y_one_hot_fake)
        D_fake_loss = loss_func(D_output, y_fake)

        D_loss = D_real_loss + D_fake_loss

        D_loss.backward()
        optimizer_d.step()

        # training generator G
        optimizer_g.zero_grad()

        G_output = G_model(z, y_one_hot_fake)
        D_output = D_model(G_output, y_one_hot_fake)
        G_loss = loss_func(D_output, y_real)
        G_loss.backward()
        optimizer_g.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tD_Loss: {:.6f}\tG_Loss: {:.6f}'.format(
                epoch, batch_idx, (len(data_loader)),
                100. * batch_idx / len(data_loader),
                D_loss.data, G_loss.data))


def distilling(last_epoch=0):
    D_model.eval()
    G_model.eval()
    SD_model.train()
    SG_model.train()

    load_teacher_dg()  # load the teacher generator and discriminator

    if args.distill_method == 'dark':
        SD_model.load_state_dict(torch.load("./share/CGAN_D_Pretrained.pth"))

    if last_epoch > 0:
        d_dir = './models/Distilled_D_Epoch_{}.pth'.format(last_epoch)
        g_dir = './models/Distilled_G_Epoch_{}.pth'.format(last_epoch)
        SD_model.load_state_dict(torch.load(d_dir))
        SG_model.load_state_dict(torch.load(g_dir))

    for epoch in range(last_epoch + 1, last_epoch + args.epochs + 1):
        distill(epoch)
        torch.save(SD_model.state_dict(), './models/Distilled_D_Epoch_{}.pth'.format(epoch))
        torch.save(SG_model.state_dict(), './models/Distilled_G_Epoch_{}.pth'.format(epoch))


def distill(epoch):
    if args.distill_method == 'dark':
        distill_dark(epoch)
    else:
        print("wrong distill_method parameters")


def distill_dark(epoch):
    for batch_idx in range(args.distill_batches):
        optimizer_sg.zero_grad()
        with torch.no_grad():
            z = torch.randn((args.batch_size, args.z_dim, 1, 1))

            y_fake_label = (torch.rand(args.batch_size) * args.class_num).type(torch.LongTensor)
            y_one_hot_fake = torch.zeros(args.batch_size, args.class_num, 1, 1)
            y_one_hot_fake.scatter_(1, y_fake_label.view(args.batch_size, 1, 1, 1), 1)

            if args.cuda:
                z = z.cuda()
                y_one_hot_fake = y_one_hot_fake.cuda()

            g_synthetic_img, g_dark = G_model.forward_with_dark(z, y_one_hot_fake)

        sg_synthetic_img, sg_dark = SG_model.forward_with_dark(z, y_one_hot_fake)
        loss = loss_distill_dark(sg_synthetic_img, sg_dark, g_synthetic_img, g_dark, y_one_hot_fake)

        loss.backward()
        optimizer_sg.step()
        if batch_idx % args.log_interval == 0:
            print('Distill Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, args.distill_batches,
                100. * batch_idx / args.distill_batches,
                loss.data))


def loss_distill_dark(student_x, dark_k_s, teacher_x, dark_k, y_one_hot_fake):
    weight = 1
    loss = F.l1_loss(student_x, teacher_x)
    loss_dark = weight * F.l1_loss(dark_k_s, dark_k)

    y_real = torch.ones(args.batch_size, requires_grad=False).cuda()
    loss_func = torch.nn.BCELoss()

    D_output = SD_model(student_x, y_one_hot_fake)
    loss_fooling = 0.0001 * loss_func(D_output, y_real)

    return loss + loss_dark + loss_fooling


def load_teacher_dg():
    d_dir = "./share/CGAN_D_Pretrained.pth"
    g_dir = "./share/CGAN_G_Pretrained.pth"
    D_model.load_state_dict(torch.load(d_dir))
    G_model.load_state_dict(torch.load(g_dir))


def load_g_model():
    # loading teacher generator
    last_cp = "./share/CGAN_G_Pretrained.pth"

    G_model.load_state_dict(torch.load(last_cp))


def load_sg_model(f_dir):
    # loading student generator from the given dir
    SG_model.load_state_dict(torch.load(f_dir))


def rand_faces(num=4, num_imgs=1):
    load_g_model()
    # m = G_model

    m = SG_model # student generator

    m.eval()
    for c in range(num_imgs):
        z = torch.randn((num*num, args.z_dim, 1, 1))

        y_fake_label = torch.ones(num*num).type(torch.LongTensor)
        # y_fake_label = torch.zeros(num * num).type(torch.LongTensor)

        y_one_hot_fake = torch.zeros(num*num, args.class_num, 1, 1)
        y_one_hot_fake.scatter_(1, y_fake_label.view(num*num, 1, 1, 1), 1)

        if args.cuda:
            z = z.cuda()
            y_one_hot_fake = y_one_hot_fake.cuda()

        g_synthetic_img = G_model(z, y_one_hot_fake)

        torchvision.utils.save_image(g_synthetic_img.data, './imgs/img_%d.jpg'%c, nrow=num, padding=2)


# def show_synthetic_comp(nrows=4, ncolumns=4, num_student=0):
#
#     load_g_model()
#     G_model.eval()
#
#     bs = nrows * ncolumns
#
#     z = torch.randn((bs, args.z_dim, 1, 1))
#     # z = torch.load('../models/z_.pth')
#
#     y_fake_label = torch.ones(int(bs)).type(torch.LongTensor)  # half for each class
#     # y_fake_label_2 = torch.zeros(int(bs/2)).type(torch.LongTensor)
#
#     # y_fake_label = torch.cat([y_fake_label, y_fake_label_2], dim=0)
#
#     y_one_hot_fake = torch.zeros(bs, args.class_num, 1, 1)
#     y_one_hot_fake.scatter_(1, y_fake_label.view(bs, 1, 1, 1), 1)
#
#     if args.cuda:
#         z = z.cuda()
#         y_one_hot_fake = y_one_hot_fake.cuda()
#
#     # torch.save(z, '../models/z1.pth') # idx 12
#     # torch.save(z, '../models/z2.pth')  # idx 1     idx 11 12 13
#     # torch.save(z, '../models/z3.pth') # idx 6 7 8
#     # torch.save(z, '../models/z4.pth') # idx 11 12 16
#     # torch.save(z, '../models/z5.pth') # idx 1 6
#     # torch.save(z, '../models/z6.pth') # idx 8
#     # torch.save(z, '../models/z7.pth') # idx 0 idx 16
#     # torch.save(z, '../models/z8.pth') # idx 0, 2, idx 12
#     # torch.save(z, '../models/z9.pth') # idx 18
#
#     g_synthetic_img = G_model(z, y_one_hot_fake)
#     torchvision.utils.save_image(g_synthetic_img, '../imgs/cgan_teacher.jpg', nrow=nrows, padding=2)
#
#     if num_student > 0:
#
#         load_sg_model('../models/cgan_distill/dark_final/Distilled_G_Epoch_10.pth')
#         # load_sg_model('../models/Distilled_G_Epoch_5.pth')
#         SG_model.eval()
#
#         sg_synthetic_img = SG_model(z, y_one_hot_fake)
#         torchvision.utils.save_image(sg_synthetic_img, '../imgs/cgan_distill_dark50.jpg', nrow=nrows, padding=2)
#
#     if num_student > 1:
#         load_sg_model('../models/cgan_distill/img_final/Distilled_G_Epoch_15.pth')
#         SG_model.eval()
#
#         sg_synthetic_img = SG_model(z, y_one_hot_fake)
#         # torchvision.utils.save_image(sg_synthetic_img, '../imgs/cgan_distill_img.jpg', nrow=nrows, padding=2)
#         torchvision.utils.save_image(sg_synthetic_img, '../imgs/cgan_distill_img4.jpg', nrow=nrows, padding=2)


