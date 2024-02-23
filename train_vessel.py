import sys

import random

sys.path.append('../')
sys.path.append('../data_process/')
sys.path.append('../networks/common/')
sys.path.append('../networks/MESnet/')

import Constants
import numpy as np
import torch
import cv2
import math
from losses import  DiceBCELoss

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
import torch.optim as optims
from torch import optim
import torch.utils.data as data
import torch.nn.functional as F
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from skimage.segmentation import find_boundaries

from time import time
from data_process.data_load import ImageFolder, get_drive_data
#from networks.second.baseline import UNet
from networks.othernets.Change_Unet import K_U_Net
from networks.othernets.KN_Net import *
from networks.othernets.Change_Unet import *
from networks.othernets.image_patch import *
from networks.othernets.Change_Unet import Joint_Net
#from networks.othernets.new1 import Joint_Net
from networks.othernets.mlp import *
from networks.othernets.R2AttU_Net import U_Net
from networks.othernets.New_Net import UU_Net,UUU_Net
from networks.othernets.R2AttU_Net import AttU_Net
from networks.othernets.R2AttU_Net import R2U_Net
from networks.othernets.R2AttU_Net import R2AttU_Net

from networks.contrast.Unet_plus import Nested_Net
from networks.contrast.iternet_model import Iternet
from networks.contrast.Multi_level_Attention_Network import FCDenseNet
from networks.contrast.DE_DCGCN_EE import DEDCGCNEE
from networks.contrast.CE_Net import CE_Net_
from networks.contrast.CGA_Net import CGAM_UNet2
from networks.contrast.CTFNET65 import LadderNetv6
from networks.contrast.FR_Unet import FR_UNet
from networks.contrast.model_skelcon import LUNet
from networks.contrast.jtfn import JTFN
from networks.contrast.AV_casNet import AV_casNet,U_Net4
from networks.contrast.LITS import Lits
from networks.contrast.axialnet import ResAxialAttentionUNet


from train_test.losses import loss_ce,FocalLoss,weighted_mse_loss
from train_test.losses import loss_ce_gai
from train_test.eval_test import val_vessel,val_vessel_3,val_vessel_2
from torch.utils.tensorboard import SummaryWriter
from train_test.help_functions import platform_info, check_size
from train_test.evaluations import threshold_by_otsu

learning_rates = Constants.learning_rates

gcn_model = False

def DiceLoss(predict, target):
    epsilon = 1e-5
    assert predict.size() == target.size(), "the size of predict and target must be equal."
    num = predict.size(0)

    pre = predict.view(num, -1)
    tar = target.view(num, -1)

    intersection = (pre * tar).sum(-1).sum()      # multiply flags and labels
    union = (pre + tar).sum(-1).sum()

    score = 1 - 2 * (intersection + epsilon) / (union + epsilon)

    return score

def load_model(path):
    net = torch.load(path)
    return net

def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_lr1(optimizer, old_lr, ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = old_lr / ratio
    print('update learning rate: %f -> %f' % (old_lr, old_lr / ratio))
    return old_lr / ratio

def update_lr2(epoch, optimizer, total_epoch=Constants.TOTAL_EPOCH):
    new_lr = learning_rates * (1 - epoch / total_epoch)
    for p in optimizer.param_groups:
        p['lr'] = new_lr

def optimizer_net(net, optimizers,  criterion, images, masks,masks1, ch):
    optimizers.zero_grad()
    # pred,pred1,pred2 = net(images)
    pred = net(images,masks1)
    loss = loss_ce(pred, masks, criterion, ch) +  DiceLoss(pred, masks)
    # loss = loss_ce(pred, masks, criterion, ch)+loss_ce(pred, masks1, criterion, ch)
    # loss = loss_ce(pred, masks, criterion, ch) + FocalLoss()(pred, torch.tensor(masks, dtype=torch.float))
    # loss =  loss_ce(pred, masks, criterion, ch) + DiceLoss(pred, masks)  # 原来的
    # print(loss)
    loss.backward()
    # loss.backward(retain_graph=True)

    optimizers.step()
    return pred, loss

def optimizer_net_test(net, optimizers,  criterion, images, masks,masks1, ch):
    optimizers.zero_grad()
    # pred,pred1,pred2 = net(images)
    pred = net(images)
    loss = loss_ce(pred, masks, criterion, ch) +  DiceLoss(pred, masks)
    # loss = loss_ce(pred, masks, criterion, ch)+loss_ce(pred, masks1, criterion, ch)
    # loss = loss_ce(pred, masks, criterion, ch) + FocalLoss()(pred, torch.tensor(masks, dtype=torch.float))
    # loss =  loss_ce(pred, masks, criterion, ch) + DiceLoss(pred, masks)  # 原来的
    # print(loss)
    loss.backward()
    # loss.backward(retain_graph=True)

    optimizers.step()
    return pred, loss


def optimizer_net_graph(net, optimizers,  criterion, images, masks,masks1, ch):
    optimizers.zero_grad()
    # pred,pred1,pred2 = net(images)
    pred = net(images,masks1)
    loss = loss_ce(pred, masks, criterion, ch) +  DiceLoss(pred, masks)
    # loss = loss_ce(pred, masks, criterion, ch)+loss_ce(pred, masks1, criterion, ch)
    # loss = loss_ce(pred, masks, criterion, ch) + FocalLoss()(pred, torch.tensor(masks, dtype=torch.float))
    # loss =  loss_ce(pred, masks, criterion, ch) + DiceLoss(pred, masks)  # 原来的
    # print(loss)
    loss.backward()
    # loss.backward(retain_graph=True)

    optimizers.step()
    return pred, loss



def optimizer_net_2(net, optimizers,  criterion, images, masks,masks1,ch):
    optimizers.zero_grad()
    pred,pred1= net(images)
    loss = loss_ce(pred, masks, criterion, ch)+loss_ce(pred1, masks1, criterion, ch)
    loss.backward()
    optimizers.step()
    return pred, loss

def optimizer_net_3(net, optimizers,  criterion, images, masks,masks1, ch):
    optimizers.zero_grad()
    pred,pred1,pred2= net(images)
    # loss = loss_ce(pred, masks, criterion, ch)
    loss = loss_ce(pred, masks, criterion, ch)+loss_ce(pred1, masks1, criterion, ch) + loss_ce(pred2, masks, criterion, ch)+ DiceLoss(pred, masks)
    # loss = loss_ce(pred, masks, criterion, ch) + DiceLoss(pred, masks)

    # + DiceLoss(pred1, masks1) + DiceLoss(pred2, masks)
    # FocalLoss()(pred, torch.tensor(masks, dtype=torch.float))
    # loss = loss_ce(pred, masks, criterion, ch) +  loss_ce(pred2, masks, criterion,ch)
    # print(loss)
    # loss =  loss_ce(pred, masks, criterion, ch) + DiceLoss(pred, masks)  # 原来的
    # pred ,out2 , out3 , out4= net(images)
    # pred_img = [pred ,out2 , out3 , out4]
    # mask1 = resize(masks, [512 // 2, 512 // 2])
    # mask2 = resize(masks, [512 // 4, 512 // 4])
    # mask3 = resize(masks, [512 // 8, 512 // 8])
    # ture_mask = []
    # ture_mask.append(masks)
    # ture_mask.append(mask1)
    # ture_mask.append(mask2)
    # ture_mask.append(mask3)
    #
    # loss = 0
    # for i in range(0,4):
    #     loss = loss + loss_ce(pred_img[i], ture_mask[i], criterion, ch)  #原来的
    loss.backward()
    optimizers.step()
    return pred, loss

def visual_preds(preds, is_preds=True):  # This for multi-classification
    rand_arr = torch.rand(size=(preds.size()[1], preds.size()[2], 3))
    color_preds = torch.zeros_like(rand_arr)
    outs = preds.permute((1, 2, 0))  # N H W C
    if is_preds is True:
        outs_one_hot = torch.argmax(outs, dim=2)
    else:
        outs_one_hot = outs.reshape((preds.size()[1], preds.size()[2]))
    for H in range(0, preds.size()[1]):
        for W in range(0, preds.size()[2]):
            if outs_one_hot[H, W] == 1:
                color_preds[H, W, 0] = 255
            if outs_one_hot[H, W] == 2:
                color_preds[H, W, 1] = 255
            if outs_one_hot[H, W] == 3:
                color_preds[H, W, 2] = 255
            if outs_one_hot[H, W] == 4:
                color_preds[H, W, 0] = 255
                color_preds[H, W, 1] = 255
                color_preds[H, W, 2] = 255
    return color_preds.permute((2, 0, 1))

def train_model(learning_rates):

    writer = SummaryWriter(comment=f"MyDRIVETrain01", flush_secs=1)
    tic = time()
    loss_lists = []
    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    ch = Constants.BINARY_CLASS
#     criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    #criterion = DiceBCELoss()
#     criterion = nn.MSELoss()
    #net = SA_U_Net().to(device)
    #net = EZ_Net().to(device)
    #net = Joint_Net().to(device)
    #net = MNext().to(device)
    #net = K_U_Net().to(device)
    # net = Ue_Net().to(device)
    net = UUU_Net().to(device)
    # net = GNN1().to(device)
    # net = Who_Net().to(device)
    # net = img_patch().to(device)
    # net.load_state_dict(torch.load('/root/daima/YuzuSoft/log/weights_save/DRIVE/Unet/17.iter5'),strict=False)
    # net = U_Net().to(device)
    # net = Nested_Net().to(device)
    # net = U1_Net().to(device)
    # net = Iternet().to(device)
    # net = FCDenseNet().to(device) #AACA
    # net = DEDCGCNEE().to(device)
    # net = CE_Net_().to(device)
    # net = CGAM_UNet2().to(device)
    # net = LadderNetv6().to(device)  #CTF
    # net = FR_UNet().to(device)
    # net = LUNet().to(device)    #skelcon
    #net = JTFN().to(device)
    # net = AV_casNet().to(device)
    # net = U_Net4().to(device)
    # net = Lits().to(device)
    # net = ResAxialAttentionUNet().to(device)

    # net = AttU_Net().to(device)
    # net = R2U_Net().to(device)
    #net = R2AttU_Net().to(device)
    optimizers = optims.Adam(net.parameters(), lr=learning_rates, betas=(0.9, 0.999))
    trains, val = get_drive_data()
    dataset = ImageFolder(trains[0], trains[1],trains[2])
    data_loader = data.DataLoader(dataset, batch_size= Constants.BATCH_SIZE, shuffle=True, num_workers=2)

    rand_img, rand_label, rand_pred = None, None, None
    for epoch in range(1, total_epoch + 1):
        net.train(mode=True)
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0
        for img, mask ,mask1 in data_loader_iter:
            img = img.to(device)
            mask = mask.to(device)
            mask1 = mask1.to(device)
            # pred, train_loss = optimizer_net_3(net,  optimizers, criterion, img, mask,mask1, ch)          #修改loss
            pred, train_loss = optimizer_net(net, optimizers, criterion, img, mask,mask1, ch)
            # pred, train_loss = optimizer_net_test(net, optimizers, criterion, img, mask, mask1, ch)
            # pred, train_loss = optimizer_net_2(net, optimizers, criterion, img, mask, mask1, ch)

            train_epoch_loss += train_loss.item()   #防止tensor无线叠加导致的显存爆炸
            index = index + 1
            if np.random.rand(1) > 0.4 and np.random.rand(1) < 0.8:
                rand_img, rand_label, rand_pred = img, mask, pred


        train_epoch_loss = train_epoch_loss / len(data_loader_iter)  #？
        writer.add_scalar('Train/loss', train_epoch_loss, epoch)
        if ch ==1:      # for [N,1,H,W]
            rand_pred_cpu = rand_pred[0, :, :, :].detach().cpu().reshape((-1,)).numpy()  #
            rand_pred_cpu = threshold_by_otsu(rand_pred_cpu)
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,)).numpy()  #
            writer.add_scalar('Train/acc', rand_pred_cpu[np.where(new_mask == rand_pred_cpu)].shape[0] / new_mask.shape[0], epoch)  # for [N,H,W,1]
        if ch ==2:      # for [N,2,H,W]
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,))
            new_pred = torch.argmax(rand_pred[0, :, :, :].permute((1, 2, 0)), dim=2).detach().cpu().reshape((-1,))
            t = new_pred[torch.where(new_mask == new_pred)].size()[0]
            writer.add_scalar('Train/acc', t / new_pred.size()[0], epoch)

        platform_info(epoch, tic, train_epoch_loss, Constants.IMG_SIZE, optimizers)
        # platform_info(epoch, tic, train_epoch_loss, Constants.IMG_SIZE, optimizers2)
        if epoch % 10 == 1:
            writer.add_image('Train/image_origins', rand_img[0, :, :, :], epoch)
            writer.add_image('Train/image_labels', rand_label[0, :, :, :], epoch)
            if ch == 2:  # for [N,1,H,W]      1->2
                writer.add_image('Train/image_predictions', rand_pred[0, :, :, :], epoch)
            if ch == 4:  # for [N,2,H,W]      2->4
                  writer.add_image('Train/image_predictions', torch.unsqueeze(torch.argmax(rand_pred[0, :, :, :], dim=0), 0),
                             epoch)
        update_lr2(epoch, optimizers)  # modify  lr
        # update_lr2(epoch, optimizers2)


        print('************ start to validate current model {}.iter performance ! ************'.format(epoch))
        # acc, sen, f1score, val_loss, iou, precision, auc= val_vessel(net, val[0], val[1], val[0].shape[0], epoch)  #测试集
        acc, sen, f1score, val_loss, iou, precision, auc = val_vessel_2(net, val[0], val[1],val[2], val[0].shape[0],epoch)  # 测试集
        # acc, sen, f1score, val_loss, iou, precision, auc = val_vessel_3(net, val[0], val[1], val[0].shape[0],epoch)  # 测试集
        writer.add_scalar('Val/accuracy', acc, epoch)
        writer.add_scalar('Val/f1score', f1score, epoch)
        writer.add_scalar('Val/auc', auc, epoch)
        writer.add_scalar('Val/iou', iou, epoch)
        writer.add_scalar('Val/sensitivity', sen, epoch)
        writer.add_scalar('Val/precision', precision, epoch)
        writer.add_scalar('Val/val_loss', val_loss, epoch)


        model_name = Constants.saved_path + "{}.iter5".format(epoch)
        # torch.save(net.state_dict(), model_name)
        torch.save(net, model_name)


    print('***************** Finish training process ***************** ')

if __name__ == '__main__':
    RANDOM_SEED = 42  # any random number
    #RANDOM_SEED = 3407
    # RANDOM_SEED = 74751
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # GPU
        torch.cuda.manual_seed_all(seed)  # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
    set_seed(RANDOM_SEED)
    train_model(learning_rates)
    pass





