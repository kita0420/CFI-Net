import sys
import random
sys.path.append('/root/root/Pycharm_Project/')
sys.path.append('../data_process/')
sys.path.append('root/YuzuSoft/networks/')
sys.path.append('../networks/common/')
sys.path.append('../networks/MESnet/')
sys.path.append('../networks/threestage/')
sys.path.append('..')

from semi.skel import img_skel
from networks.othernets.KN_Net import *
from networks.contrast.Multi_level_Attention_Network import FCDenseNet
from semi.vessel_delete import *
from networks.othernets.Change_Unet import *
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optims
import torch.utils.data as data
from torch.autograd import Variable as V
import sklearn.metrics as metrics
import cv2
import os
import matplotlib.pyplot as plt
from networks.othernets.image_patch import *
from networks.common.differentce_retinal import retina_color_different
from data_process.retinal_process import clahe_equalized,dataset_normalized,adjust_gamma,adjust_contrast,gaussianNoise,blurr,drop,sharpen
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from networks.othernets.R2AttU_Net import U_Net
from data_process.data_ultils import read_all_images
from data_process.data_load import ImageFolder,get_drive_data,get_drive_data_skel
# from data_process.data_load_skel import get_drive_data_skel
# from avseg import AVSegNet
from networks.contrast.jtfn import JTFN
# from networks.othernets.New_Net import UU_Net
from train_test.losses import loss_ce, loss_ce_ds
import Constants
from train_test.evaluations import misc_measures,roc_pr_curve,threshold_by_otsu,SkeletalSimilarity
import warnings
from networks.contrast.CE_Net import CE_Net_
from networks.contrast.LITS import Lits
warnings.filterwarnings('ignore')
BATCHSIZE_PER_CARD = 1


def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def load_model(path):
    net = torch.load(path)
    return net

def visualize(data, filename):
    '''
    :param data:     input is 3d tensor of a image,whose size is (H*W*C)
    :param filename:
    :return:         saved into filename positions
    '''
    assert (len(data.shape) == 3)  # height*width*channels
    # print data
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))          # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img

def val_vessel(net1, imgs, masks, length, epoch =0, ch = Constants.BINARY_CLASS):
    acc, sensitivity, specificity, precision, G, F1_score, mse, iou, hausdorff_dis,val_loss, auc= [],[],[],[],[],[],[],[],[],[],[]
    net1.eval()
    with torch.no_grad():
        for iteration in range(0, length):
            x_img = imgs[iteration]
            x_img = np.expand_dims(x_img, axis=0)                     # (H, W, C) to (1, H, W, C)
            x_img = torch.tensor(x_img, dtype=torch.float32).to(device)
            print(x_img.size(),'---------------')
            generated_vessel= crop_eval(net1, x_img)
            # generated_vessel = crop_eval(net1, x_img)
            vl = nn.BCELoss()(generated_vessel.detach().cpu().reshape((-1,)), torch.tensor(masks[iteration].reshape((-1,)), dtype=torch.float))
            val_loss.append(vl.numpy())
            generated_vessel = generated_vessel.permute((0, 2, 3, 1)).detach().cpu().numpy()
            pr_g, pr_l = [], []
            if ch ==1:   # for [N,1,H,W]
                pr_g.append(generated_vessel.reshape((-1,)).tolist())#ROC?
                pr_l.append(masks[iteration].reshape((-1,)).tolist())#
                visualize(np.asarray(generated_vessel[0, :, :, :, ]), Constants.visual_results + 'pred_val_prob_pic' + str(iteration))
                #generated_vessel = threshold_by_otsu(generated_vessel)
                threshold = 0.5
                generated_vessel[generated_vessel >= threshold] = 1
                generated_vessel[generated_vessel < threshold] = 0
                generated_vessel = generated_vessel

            if ch ==2:   # for [N,H,W,2]
                pr_g.append(generated_vessel.reshape((-1,)).tolist())
                pr_l.append(masks[iteration].reshape((-1,)).tolist())
                generated_vessel = np.expand_dims(np.argmax(generated_vessel, axis =3), axis=3)
            generated_vessel = np.squeeze(generated_vessel, axis=0)   # (1, H, W, 1) to (H, W, 1)

            visualize(np.asarray(generated_vessel), Constants.visual_results + 'pred_val_pic' + str(iteration))
            # retina_color_different(np.asarray(generated_vessel), masks[iteration].transpose((1, 2, 0)), Constants.visual_results +'pred_'+ str(iteration) + 'different')
            # print('value check :', np.max(masks[iteration]), str(iteration), np.min(masks[iteration]))
            metrics_current = misc_measures(masks[iteration].reshape((-1,)), generated_vessel.reshape((-1,)), False)
            acc.append(metrics_current[0])
            sensitivity.append(metrics_current[1])
            specificity.append(metrics_current[2])
            precision.append(metrics_current[3])
            G.append(metrics_current[4])
            F1_score.append(metrics_current[5])
            mse.append(metrics_current[6])
            iou.append(metrics_current[7])
            hausdorff_dis.append(metrics_current[8])
            auc.append(metrics_current[9])
            AUC_prec_rec, AUC_ROC = roc_pr_curve(np.array(pr_l).reshape((-1,)), np.array(pr_g).reshape((-1,)),
                                                 Constants.visual_results)
        print('********************** below is validation evaluation of epoch {} results **********************'.format(epoch))
        print('Accuracy average is:{}, std is:{}'.format(np.mean(acc), np.std(acc)))
        print('Sensitivity average is:{}, std is:{}'.format(np.mean(sensitivity), np.std(sensitivity)))
        print('Specificity average is:{}, std is:{}'.format(np.mean(specificity), np.std(specificity)))
        print('Precision average is:{}, std is:{}'.format(np.mean(precision), np.std(precision)))
        print('G average is:{}, std is:{}'.format(np.mean(G), np.std(G)))
        print('F1_score average is:{}, std is:{}'.format(np.mean(F1_score), np.std(F1_score)))
        print('Mse average is:{}, std is:{}'.format(np.mean(mse), np.std(mse)))
        print('Iou average is:{}, std is:{}'.format(np.mean(iou), np.std(iou)))
        print('Hausdorff_distance average is:{}, std is:{}'.format(np.mean(hausdorff_dis), np.std(hausdorff_dis)))
        print('Auc is:{}, std is:{}:'.format(np.mean(AUC_ROC), np.std(AUC_ROC)))
        print('val_loss is :', val_loss)

        s = 'epoch:{}, Accuracy average is:{}, Sensitivity average is:{}, F1_score average is:{}, Iou average is:{}'.format(epoch,
                                                                                np.mean(acc), np.mean(sensitivity), np.mean(F1_score), np.mean(iou),np.mean(precision))
        with open(os.path.join('/root/daima/YuzuSoft/log', 'save_result.txt'), 'a', encoding='utf-8') as f:
            f.write(s)
            f.write('\n')

    return np.mean(acc), np.mean(sensitivity), np.mean(F1_score), np.mean(val_loss), np.mean(iou) , np.mean(precision), np.mean(AUC_ROC)

def val_vessel_2(net1, imgs, masks,skel, length, epoch =0, ch = Constants.BINARY_CLASS):
    acc, sensitivity, specificity, precision, G, F1_score, mse, iou, hausdorff_dis,val_loss, auc= [],[],[],[],[],[],[],[],[],[],[]
    net1.eval()
    with torch.no_grad():
        for iteration in range(0, length):
            x_img = imgs[iteration]
            x_skel = skel[iteration]
            x_img = np.expand_dims(x_img, axis=0)                     # (H, W, C) to (1, H, W, C)
            x_skel = np.expand_dims(x_skel, axis=0)
            x_img = torch.tensor(x_img, dtype=torch.float32).to(device)
            x_skel = torch.tensor(x_skel, dtype=torch.float32).to(device)
            print(x_img.size(),x_skel.size(),'---------------')
            generated_vessel= crop_eval_skel(net1, x_img,x_skel)
            # generated_vessel = crop_eval(net1, x_img)

            vl = nn.BCELoss()(generated_vessel.detach().cpu().reshape((-1,)), torch.tensor(masks[iteration].reshape((-1,)), dtype=torch.float))
            val_loss.append(vl.numpy())
            generated_vessel = generated_vessel.permute((0, 2, 3, 1)).detach().cpu().numpy()
            pr_g, pr_l = [], []
            if ch ==1:   # for [N,1,H,W]
                pr_g.append(generated_vessel.reshape((-1,)).tolist())#ROC?
                pr_l.append(masks[iteration].reshape((-1,)).tolist())#
                visualize(np.asarray(generated_vessel[0, :, :, :, ]), Constants.visual_results + 'pred_val_prob_pic' + str(iteration))
                #generated_vessel = threshold_by_otsu(generated_vessel)
                threshold = 0.5
                generated_vessel[generated_vessel >= threshold] = 1
                generated_vessel[generated_vessel < threshold] = 0
                generated_vessel = generated_vessel

            if ch ==2:   # for [N,H,W,2]
                pr_g.append(generated_vessel.reshape((-1,)).tolist())
                pr_l.append(masks[iteration].reshape((-1,)).tolist())
                generated_vessel = np.expand_dims(np.argmax(generated_vessel, axis =3), axis=3)
            generated_vessel = np.squeeze(generated_vessel, axis=0)   # (1, H, W, 1) to (H, W, 1)

            visualize(np.asarray(generated_vessel), Constants.visual_results + 'pred_val_pic' + str(iteration))
            # retina_color_different(np.asarray(generated_vessel), masks[iteration].transpose((1, 2, 0)), Constants.visual_results +'pred_'+ str(iteration) + 'different')
            # print('value check :', np.max(masks[iteration]), str(iteration), np.min(masks[iteration]))
            metrics_current = misc_measures(masks[iteration].reshape((-1,)), generated_vessel.reshape((-1,)), False)
            acc.append(metrics_current[0])
            sensitivity.append(metrics_current[1])
            specificity.append(metrics_current[2])
            precision.append(metrics_current[3])
            G.append(metrics_current[4])
            F1_score.append(metrics_current[5])
            mse.append(metrics_current[6])
            iou.append(metrics_current[7])
            hausdorff_dis.append(metrics_current[8])
            auc.append(metrics_current[9])
            AUC_prec_rec, AUC_ROC = roc_pr_curve(np.array(pr_l).reshape((-1,)), np.array(pr_g).reshape((-1,)),
                                                 Constants.visual_results)
        print('********************** below is validation evaluation of epoch {} results **********************'.format(epoch))
        print('Accuracy average is:{}, std is:{}'.format(np.mean(acc), np.std(acc)))
        print('Sensitivity average is:{}, std is:{}'.format(np.mean(sensitivity), np.std(sensitivity)))
        print('Specificity average is:{}, std is:{}'.format(np.mean(specificity), np.std(specificity)))
        print('Precision average is:{}, std is:{}'.format(np.mean(precision), np.std(precision)))
        print('G average is:{}, std is:{}'.format(np.mean(G), np.std(G)))
        print('F1_score average is:{}, std is:{}'.format(np.mean(F1_score), np.std(F1_score)))
        print('Mse average is:{}, std is:{}'.format(np.mean(mse), np.std(mse)))
        print('Iou average is:{}, std is:{}'.format(np.mean(iou), np.std(iou)))
        print('Hausdorff_distance average is:{}, std is:{}'.format(np.mean(hausdorff_dis), np.std(hausdorff_dis)))
        print('Auc is:{}, std is:{}:'.format(np.mean(AUC_ROC), np.std(AUC_ROC)))
        print('val_loss is :', val_loss)

        s = 'epoch:{}, Accuracy average is:{}, Sensitivity average is:{}, F1_score average is:{}, Iou average is:{}'.format(epoch,
                                                                                np.mean(acc), np.mean(sensitivity), np.mean(F1_score), np.mean(iou),np.mean(precision))
        with open(os.path.join('/root/daima/YuzuSoft/log', 'save_result.txt'), 'a', encoding='utf-8') as f:
            f.write(s)
            f.write('\n')

    return np.mean(acc), np.mean(sensitivity), np.mean(F1_score), np.mean(val_loss), np.mean(iou) , np.mean(precision), np.mean(AUC_ROC)


def val_vessel_3(net1, imgs, masks, length, epoch =0, ch = Constants.BINARY_CLASS):
    acc, sensitivity, specificity, precision, G, F1_score, mse, iou, hausdorff_dis,val_loss, auc= [],[],[],[],[],[],[],[],[],[],[]
    net1.eval()
    with torch.no_grad():
        for iteration in range(0, length):
            x_img = imgs[iteration]
            x_img = np.expand_dims(x_img, axis=0)                     # (H, W, C) to (1, H, W, C)
            x_img = torch.tensor(x_img, dtype=torch.float32).to(device)
            print(x_img.size(),'---------------')
            generated_vessel,patch_out,patch_128 = crop_eval_3(net1, x_img)


            # generated_vessel = crop_eval(net1, x_img)
            vl = nn.BCELoss()(generated_vessel.detach().cpu().reshape((-1,)), torch.tensor(masks[iteration].reshape((-1,)), dtype=torch.float))
            val_loss.append(vl.numpy())
            generated_vessel = generated_vessel.permute((0, 2, 3, 1)).detach().cpu().numpy()
            patch_out = patch_out.permute((0, 2, 3, 1)).detach().cpu().numpy()
            patch_128 = patch_128.permute((0, 2, 3, 1)).detach().cpu().numpy()
            pr_g, pr_l = [], []
            if ch ==1:   # for [N,1,H,W]
                pr_g.append(generated_vessel.reshape((-1,)).tolist())#ROC?
                pr_l.append(masks[iteration].reshape((-1,)).tolist())#
                visualize(np.asarray(generated_vessel[0, :, :, :, ]), Constants.visual_results + 'pred_val_prob_pic' + str(iteration))
                visualize(np.asarray(patch_out[0, :, :, :, ]),Constants.visual_results + 'img_val_prob_pic' + str(iteration))
                visualize(np.asarray(patch_128[0, :, :, :, ]),Constants.visual_results + 'patch_val_prob_pic' + str(iteration))
                #generated_vessel = threshold_by_otsu(generated_vessel)
                threshold = 0.5
                generated_vessel[generated_vessel >= threshold] = 1
                generated_vessel[generated_vessel < threshold] = 0
                generated_vessel = generated_vessel

                patch_out[patch_out >= threshold] = 1
                patch_out[patch_out < threshold] = 0

                patch_128[patch_128 >= threshold] = 1
                patch_128[patch_128 < threshold] = 0

            if ch ==2:   # for [N,H,W,2]
                pr_g.append(generated_vessel.reshape((-1,)).tolist())
                pr_l.append(masks[iteration].reshape((-1,)).tolist())
                generated_vessel = np.expand_dims(np.argmax(generated_vessel, axis =3), axis=3)
            # print('128:',np.max(patch_128),np.min(patch_128))
            generated_vessel = np.squeeze(generated_vessel, axis=0)   # (1, H, W, 1) to (H, W, 1)
            patch_out = np.squeeze(patch_out, axis=0)
            patch_128 = np.squeeze(patch_128, axis=0)
            visualize(np.asarray(generated_vessel), Constants.visual_results + 'pred_val_pic' + str(iteration))
            visualize(np.asarray(patch_out), Constants.visual_results + 'img_val_pic' + str(iteration))#记得删
            visualize(np.asarray(patch_128), Constants.visual_results + 'edge_val_pic' + str(iteration))#记得删
            retina_color_different(np.asarray(generated_vessel), masks[iteration].transpose((1, 2, 0)), Constants.visual_results +'pred_'+ str(iteration) + 'different')
            # retina_color_different(np.asarray(patch_out), masks[iteration].transpose((1, 2, 0)),
            #                        Constants.visual_results +'image_' +str(iteration) + 'different')
            # retina_color_different(np.asarray(patch_128), masks[iteration].transpose((1, 2, 0)),
            #                        Constants.visual_results +'patch_'+ str(iteration) + 'different')


            # print('value check :', np.max(masks[iteration]), str(iteration), np.min(masks[iteration]))
            metrics_current = misc_measures(masks[iteration].reshape((-1,)), generated_vessel.reshape((-1,)), False)
            acc.append(metrics_current[0])
            sensitivity.append(metrics_current[1])
            specificity.append(metrics_current[2])
            precision.append(metrics_current[3])
            G.append(metrics_current[4])
            F1_score.append(metrics_current[5])
            mse.append(metrics_current[6])
            iou.append(metrics_current[7])
            hausdorff_dis.append(metrics_current[8])
            auc.append(metrics_current[9])
            AUC_prec_rec, AUC_ROC = roc_pr_curve(np.array(pr_l).reshape((-1,)), np.array(pr_g).reshape((-1,)),
                                                 Constants.visual_results)
        print('********************** below is validation evaluation of epoch {} results **********************'.format(epoch))
        print('Accuracy average is:{}, std is:{}'.format(np.mean(acc), np.std(acc)))
        print('Sensitivity average is:{}, std is:{}'.format(np.mean(sensitivity), np.std(sensitivity)))
        print('Specificity average is:{}, std is:{}'.format(np.mean(specificity), np.std(specificity)))
        print('Precision average is:{}, std is:{}'.format(np.mean(precision), np.std(precision)))
        print('G average is:{}, std is:{}'.format(np.mean(G), np.std(G)))
        print('F1_score average is:{}, std is:{}'.format(np.mean(F1_score), np.std(F1_score)))
        print('Mse average is:{}, std is:{}'.format(np.mean(mse), np.std(mse)))
        print('Iou average is:{}, std is:{}'.format(np.mean(iou), np.std(iou)))
        print('Hausdorff_distance average is:{}, std is:{}'.format(np.mean(hausdorff_dis), np.std(hausdorff_dis)))
        print('Auc is:{}, std is:{}:'.format(np.mean(AUC_ROC), np.std(AUC_ROC)))
        print('val_loss is :', val_loss)

        s = 'epoch:{}, Accuracy average is:{}, Sensitivity average is:{}, F1_score average is:{}, Iou average is:{}'.format(epoch,
                                                                                np.mean(acc), np.mean(sensitivity), np.mean(F1_score), np.mean(iou),np.mean(precision))
        with open(os.path.join('/root/daima/YuzuSoft/log', 'save_result.txt'), 'a', encoding='utf-8') as f:
            f.write(s)
            f.write('\n')

    return np.mean(acc), np.mean(sensitivity), np.mean(F1_score), np.mean(val_loss), np.mean(iou) , np.mean(precision), np.mean(AUC_ROC)

from evaluations import hausdorff_dis,confusion_matrix,AUC_ROC
def extra_test(true_vessel_arr, pred_vessel_arr):
    R = 0.8  # noise ratio
    TAU = 3  # thickness threshold
    true_vessel_arr = true_vessel_arr.reshape(true_vessel_arr.shape[1],true_vessel_arr.shape[2])
    pred_vessel_arr =pred_vessel_arr.reshape(pred_vessel_arr.shape[0],pred_vessel_arr.shape[1])
    im = thinning(true_vessel_arr)  # 确实很瘦
    rects = []
    polys = traceSkeleton(im, 0, 0, im.shape[1], im.shape[0], 10, 999, rects)
    #        polys = np.load(os.path.join(poly_file, f.split('/')[-1]), allow_pickle=True)
    #         np.save(os.path.join(poly_file, f.split('/')[-1].replace('gif', 'npy')), polys)
    # Select the ratio
    cut_map, poly_s = cut_vessel(true_vessel_arr*255, polys, R, TAU)
    # np.save(os.path.join(polys_file, f.split('/')[-1].replace('gif', 'npy')), poly_s)
    cut_map = cut_map / 255
    back_true_vessel = true_vessel_arr - cut_map

    fore_pred_vessel = pred_vessel_arr * cut_map
    back_pred_vessel = pred_vessel_arr - fore_pred_vessel
    # imageio.imwrite('a.png',fore_pred_vessel*255)
    # imageio.imwrite('b.png', back_pred_vessel*255)
    dis_xi = hausdorff_dis(cut_map.reshape((-1,)), fore_pred_vessel.reshape((-1,)))
    cm = confusion_matrix(cut_map.reshape((-1,)), fore_pred_vessel.reshape((-1,)))  # 混淆矩阵
    sensitivity_xi = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    auc_xi = AUC_ROC(cut_map.reshape((-1,)), fore_pred_vessel.reshape((-1,)))

    dis_cu = hausdorff_dis(back_true_vessel.reshape((-1,)), back_pred_vessel.reshape((-1,)))
    cm1 = confusion_matrix(back_true_vessel.reshape((-1,)), back_pred_vessel.reshape((-1,)))  # 混淆矩阵
    sensitivity_cu = 1. * cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    auc_cu = AUC_ROC(back_true_vessel.reshape((-1,)), back_pred_vessel.reshape((-1,)))
    return dis_xi,dis_cu,sensitivity_xi,sensitivity_cu,auc_xi,auc_cu


def test_vessel(path_1, ch = Constants.BINARY_CLASS):
    images, masks = get_drive_data(is_train=False)
    # images, masks = get_drive_data_skel(is_train=False)
    test_list = ['our_DPF']
    # pre = PreWho1()
    for i in test_list:
        path = path_1 + '/DRIVE/' +str(i) + '.iter5'
        print('test:', i)
        acc, sensitivity, specificity, precision, G, F1_score, mse, iou, hausdorff_dis,hasu_xi,hasu_cu,acc_xi,acc_cu,auc,auc_xi,auc_cu= [], [], [], [], [], [], [], [], [],[],[],[],[],[],[],[]
        pr_g, pr_l = [], []
        mcc = []
        F, C, A, L, rSe, rSp, rAcc = [], [], [], [], [], [], []
        # net = UU_Net().to(device)
        # net = Who_Net().to(device)
        # net = img_patch().to(device)
        # net = U1_Net().to(device)
        # net = FCDenseNet().to(device)
        # net = Ue_Net().to(device)
        # net = CE_Net_().to(device)
        # net = JTFN().to(device)
        # net = Lits().to(device)
        # net.load_state_dict(torch.load(path))
        net = load_model(path)
        net.eval()
        with torch.no_grad():
            for iter_ in range(int(Constants.total_drive)):
            # for iter_ in range(0,8):
            # for iter_ in range(10,11):
            # for iter_ in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
            # for iter_ in [0,1,2,4,5,6,7,9,10,11,12,14,16,17,18,20,21,22,24,25,27,28,29,30,31,33]:
                x_img = images[iter_]
                x_img = np.expand_dims(x_img, axis=0)
                x_img = torch.tensor(x_img, dtype=torch.float32).to(device)
                # generated_vessel,_,_ = crop_eval_3(net, x_img)
                # generated_vessel, patch_out, patch_128 = crop_eval_3(net, x_img)
                generated_vessel = crop_eval(net, x_img)
                # generated_vessel1 = generated_vessel1.permute((0, 2, 3, 1)).detach().cpu().numpy()
                generated_vessel = generated_vessel.permute((0, 2, 3, 1)).detach().cpu().numpy()
                # patch_out = patch_out.permute((0, 2, 3, 1)).detach().cpu().numpy()
                # patch_128 = patch_128.permute((0, 2, 3, 1)).detach().cpu().numpy()
                if ch == 1:  # for [N,1,H,W]
                    pr_g.append(generated_vessel.reshape((-1,)).tolist())
                    pr_l.append(masks[iter_].reshape((-1,)).tolist())
                    visualize(np.asarray(generated_vessel[0, :, :, :, ]), Constants.visual_results + 'prob_'+ str(iter_) )
                    threshold = 0.5                                          # for [N,H,W,1]
                    generated_vessel[generated_vessel >= threshold] = 1
                    generated_vessel[generated_vessel < threshold] = 0
                    # patch_out[patch_out >= 0.35] = 1
                    # patch_out[patch_out < 0.35] = 0

                    # patch_128[patch_128 >= threshold] = 1
                    # patch_128[patch_128 < threshold] = 0
                    # generated_vessel1[generated_vessel1 >= threshold] = 1
                    # generated_vessel1[generated_vessel1 < threshold] = 0
                    # generated_vessel = generated_vessel + generated_vessel1
                    # generated_vessel[generated_vessel == 2]=1
                    # generated_vessel = threshold_by_otsu(generated_vessel)
                if ch == 2:  # for [N,H,W,2]
                    generated_vessel = np.expand_dims(np.argmax(generated_vessel, axis=3), axis=3)  # for [N,H,W,2]
                    pr_g.append(generated_vessel.reshape((-1,)).tolist())
                    pr_l.append(masks[iter_].reshape((-1,)).tolist())
                generated_vessel = np.squeeze(generated_vessel, axis=0)  # (1, H, W, 1) to (H, W)
                # visualize(np.asarray(generated_vessel),Constants.visual_results + str(iter_)+ 'seg')
                # patch_out = np.squeeze(patch_out, axis=0)
                # patch_128 = np.squeeze(patch_128, axis=0)
                # visualize(np.asarray(patch_out), Constants.visual_results + 'img_val_pic' + str(iter_))  # 记得删
                # visualize(np.asarray(patch_128), Constants.visual_results + 'edge_val_pic' + str(iter_))  # 记得删
                visualize(np.asarray(generated_vessel), Constants.visual_results + 'pre_' + str(iter_) )
                retina_color_different(np.asarray(generated_vessel), masks[iter_].transpose((1, 2, 0)), Constants.visual_results + 'different_'+ str(iter_) )
                print('value check :', np.max(masks[iter_]), str(iter_), np.min(masks[iter_]))
                metrics_current = misc_measures(masks[iter_].reshape((-1,)), generated_vessel.reshape((-1,)))

                # ff,c,a,l = CAL(masks[iter_], generated_vessel)
                # F.append(ff)
                # C.append(c)
                # A.append(c)
                # L.append(c)

                acc.append(metrics_current[0])
                sensitivity.append(metrics_current[1])
                specificity.append(metrics_current[2])
                precision.append(metrics_current[3])
                G.append(metrics_current[4])
                F1_score.append(metrics_current[5])
                mse.append(metrics_current[6])
                iou.append(metrics_current[7])
                hausdorff_dis.append(metrics_current[8])
                auc.append(metrics_current[9])
                mcc.append(metrics_current[10])
                print('image: {} test evaluations **** acc is: {}, sensitivity is: {},specificity is: {},precision is: {},G is: {},F1_score is: {},'
                      'mse is: {},iou is: {},hausdorff is: {} ****'.format(iter_, metrics_current[0],metrics_current[1],metrics_current[2],metrics_current[3],
                                                                 metrics_current[4],metrics_current[5],metrics_current[6],metrics_current[7],
                                                                 metrics_current[8]))
                # dis_xi,dis_cu,acc1_xi,acc1_cu,auc1_xi,auc1_cu = extra_test(masks[iter_],generated_vessel)
                # hasu_xi.append(dis_xi)
                # hasu_cu.append(dis_cu)
                # acc_xi.append(acc1_xi)
                # acc_cu.append(acc1_cu)
                # auc_xi.append(auc1_xi)
                # auc_cu.append(auc1_cu)
            AUC_prec_rec, AUC_ROC = roc_pr_curve(np.array(pr_l).reshape((-1,)), np.array(pr_g).reshape((-1,)),
                                                 Constants.visual_results)
            path_files_saved = '/root/daima/YuzuSoft/log/weights_save/' + 'evaluation.txt'
            print('********************** final test results has been saved in to {} **********************'.format(path_files_saved))
            str_a = 'Area of PR curve is: {}, Area of ROC curve is: {}'.format(AUC_prec_rec, AUC_ROC)
            str_b = 'Accuracy average is: {}, std is: {}'.format(np.mean(acc), np.std(acc))
            str_c = 'Sensitivity average is: {}, std is: {}'.format(np.mean(sensitivity), np.std(sensitivity))
            str_d = 'Specificity average is: {}, std is: {}'.format(np.mean(specificity), np.std(specificity))
            str_e = 'Precision average is: {}, std is: {}'.format(np.mean(precision), np.std(precision))
            str_f = 'G average is: {}, std is: {}'.format(np.mean(G), np.std(G))
            str_g = 'F1_score average is:{}, std is: {}'.format(np.mean(F1_score), np.std(F1_score))
            str_h = 'Mse average is: {}, std is: {}'.format(np.mean(mse), np.std(mse))
            str_i = 'Iou average is: {}, std is: {}'.format(np.mean(iou), np.std(iou))
            str_j = 'Hausdorff_distance average is: {}, std is: {}'.format(np.mean(hausdorff_dis), np.std(hausdorff_dis))
            str_y = 'MCC average is: {}, std is: {}'.format(np.mean(mcc),np.std(mcc))
            str_z = 'AUC average is: {}, std is: {}'.format(np.mean(auc),np.std(auc))
            # str_k = 'hasu_xi average is: {}, std is: {}'.format(np.mean(hasu_xi), np.std(hasu_xi))
            # str_l = 'hasu_cu average is: {}, std is: {}'.format(np.mean(hasu_cu), np.std(hasu_cu))
            # str_o = 'SEN1_xi average is: {}, std is: {}'.format(np.mean(acc_xi), np.std(acc_xi))
            # str_p = 'SEN_cu average is: {}, std is: {}'.format(np.mean(acc_cu), np.std(acc_cu))
            # str_q = 'AUC_xi average is: {}, std is: {}'.format(np.mean(auc_xi), np.std(acc_xi))
            # str_r = 'AUC_cu average is: {}, std is: {}'.format(np.mean(auc_cu), np.std(acc_cu))

            # str_F = 'F average is: {}, std is: {}'.format(np.mean(F), np.std(F))
            # str_C = 'C average is: {}, std is: {}'.format(np.mean(C), np.std(C))
            # str_A = 'A average is: {}, std is: {}'.format(np.mean(A), np.std(A))
            # str_L = 'L average is: {}, std is: {}'.format(np.mean(L), np.std(L))
            f = open(path_files_saved, 'a', encoding='utf-8')
            f.write(str(i)+'.iter5'+'\n')
            f.write(str_a+'\n')
            f.write(str_b+'\n')
            f.write(str_c+'\n')
            f.write(str_d+'\n')
            f.write(str_e+'\n')
            f.write(str_i + '\n')
            f.write(str_g + '\n')
            f.write(str_f+'\n')
            f.write(str_h+'\n')
            f.write(str_j+'\n')
            f.write(str_z + '\n')
            f.write(str_y + '\n')
            # f.write(str_k + '\n')
            # f.write(str_l + '\n')
            # f.write(str_o + '\n')
            # f.write(str_p + '\n')
            # f.write(str_q + '\n')
            # f.write(str_r + '\n')

            # f.write(str_F + '\n')
            # f.write(str_C + '\n')
            # f.write(str_A + '\n')
            # f.write(str_L + '\n')
            f.write('\n')
            f.close()

def test_vessel_2(path_1, ch = Constants.BINARY_CLASS):
    images, masks,skels = get_drive_data(is_train=False)
    # images, masks = get_drive_data_skel(is_train=False)
    test_list = ['CE']
    # pre = PreWho1()
    for i in test_list:
        path = path_1  +'/DRIVE/'+str(i) + '.iter5'
        print('test:', i)
        acc, sensitivity, specificity, precision, G, F1_score, mse, iou, hausdorff_dis,hasu_xi,hasu_cu,acc_xi,acc_cu,auc,auc_xi,auc_cu= [], [], [], [], [], [], [], [], [],[],[],[],[],[],[],[]
        pr_g, pr_l = [], []
        mcc = []
        F, C, A, L, rSe, rSp, rAcc = [], [], [], [], [], [], []
        # net = UU_Net().to(device)
        # net = Who_Net().to(device)
        # net = img_patch().to(device)
        # net = U1_Net().to(device)
        # net = FCDenseNet().to(device)
        # net = Ue_Net().to(device)
        # net = CE_Net_().to(device)
        # net = JTFN().to(device)
        # net = Lits().to(device)
        # net.load_state_dict(torch.load(path))
        net = load_model(path)
        net.eval()
        with torch.no_grad():
            for iter_ in range(int(Constants.total_drive)):
            # for iter_ in range(0,8):
            # for iter_ in range(10,11):
            # for iter_ in [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
            # for iter_ in [0,1,2,4,5,6,7,9,10,11,12,14,16,17,18,20,21,22,24,25,27,28,29,30,31,33]:
                skel_img = skels[iter_]
                skel_img = np.expand_dims(skel_img, axis=0)
                skel_img = torch.tensor(skel_img, dtype=torch.float32).to(device)
                x_img = images[iter_]
                x_img = np.expand_dims(x_img, axis=0)
                x_img = torch.tensor(x_img, dtype=torch.float32).to(device)
                generated_vessel = crop_eval(net, x_img)
                # generated_vessel, patch_out, patch_128 = crop_eval_3(net, x_img)
                # generated_vessel = crop_eval_skel(net, x_img,skel_img)
                # generated_vessel1 = generated_vessel1.permute((0, 2, 3, 1)).detach().cpu().numpy()
                generated_vessel = generated_vessel.permute((0, 2, 3, 1)).detach().cpu().numpy()
                # patch_out = patch_out.permute((0, 2, 3, 1)).detach().cpu().numpy()
                # patch_128 = patch_128.permute((0, 2, 3, 1)).detach().cpu().numpy()
                if ch == 1:  # for [N,1,H,W]
                    pr_g.append(generated_vessel.reshape((-1,)).tolist())
                    pr_l.append(masks[iter_].reshape((-1,)).tolist())
                    visualize(np.asarray(generated_vessel[0, :, :, :, ]), Constants.visual_results + 'prob_'+ str(iter_+1) )
                    threshold = 0.5                                          # for [N,H,W,1]
                    generated_vessel[generated_vessel >= threshold] = 1
                    generated_vessel[generated_vessel < threshold] = 0
                    # patch_out[patch_out >= 0.35] = 1
                    # patch_out[patch_out < 0.35] = 0

                    # patch_128[patch_128 >= threshold] = 1
                    # patch_128[patch_128 < threshold] = 0
                    # generated_vessel1[generated_vessel1 >= threshold] = 1
                    # generated_vessel1[generated_vessel1 < threshold] = 0
                    # generated_vessel = generated_vessel + generated_vessel1
                    # generated_vessel[generated_vessel == 2]=1
                    # generated_vessel = threshold_by_otsu(generated_vessel)
                if ch == 2:  # for [N,H,W,2]
                    generated_vessel = np.expand_dims(np.argmax(generated_vessel, axis=3), axis=3)  # for [N,H,W,2]
                    pr_g.append(generated_vessel.reshape((-1,)).tolist())
                    pr_l.append(masks[iter_].reshape((-1,)).tolist())
                generated_vessel = np.squeeze(generated_vessel, axis=0)  # (1, H, W, 1) to (H, W)
                # visualize(np.asarray(generated_vessel),Constants.visual_results + str(iter_)+ 'seg')
                # patch_out = np.squeeze(patch_out, axis=0)
                # patch_128 = np.squeeze(patch_128, axis=0)
                # visualize(np.asarray(patch_out), Constants.visual_results + 'img_val_pic' + str(iter_))  # 记得删
                # visualize(np.asarray(patch_128), Constants.visual_results + 'edge_val_pic' + str(iter_))  # 记得删
                visualize(np.asarray(generated_vessel), Constants.visual_results + 'pre_' + str(iter_+1) )
                retina_color_different(np.asarray(generated_vessel), masks[iter_].transpose((1, 2, 0)), Constants.visual_results + 'different_'+ str(iter_+1) )
                print('value check :', np.max(masks[iter_]), str(iter_), np.min(masks[iter_]))
                metrics_current = misc_measures(masks[iter_].reshape((-1,)), generated_vessel.reshape((-1,)))

                # ff,c,a,l = CAL(masks[iter_], generated_vessel)
                # F.append(ff)
                # C.append(c)
                # A.append(c)
                # L.append(c)

                acc.append(metrics_current[0])
                sensitivity.append(metrics_current[1])
                specificity.append(metrics_current[2])
                precision.append(metrics_current[3])
                G.append(metrics_current[4])
                F1_score.append(metrics_current[5])
                mse.append(metrics_current[6])
                iou.append(metrics_current[7])
                hausdorff_dis.append(metrics_current[8])
                auc.append(metrics_current[9])
                mcc.append(metrics_current[10])
                print('image: {} test evaluations **** acc is: {}, sensitivity is: {},specificity is: {},precision is: {},G is: {},F1_score is: {},'
                      'mse is: {},iou is: {},hausdorff is: {} ****'.format(iter_, metrics_current[0],metrics_current[1],metrics_current[2],metrics_current[3],
                                                                 metrics_current[4],metrics_current[5],metrics_current[6],metrics_current[7],
                                                                 metrics_current[8]))
                # dis_xi,dis_cu,acc1_xi,acc1_cu,auc1_xi,auc1_cu = extra_test(masks[iter_],generated_vessel)
                # hasu_xi.append(dis_xi)
                # hasu_cu.append(dis_cu)
                # acc_xi.append(acc1_xi)
                # acc_cu.append(acc1_cu)
                # auc_xi.append(auc1_xi)
                # auc_cu.append(auc1_cu)
            AUC_prec_rec, AUC_ROC = roc_pr_curve(np.array(pr_l).reshape((-1,)), np.array(pr_g).reshape((-1,)),
                                                 Constants.visual_results)
            path_files_saved = '/root/daima/YuzuSoft/log/weights_save/' + 'evaluation.txt'
            print('********************** final test results has been saved in to {} **********************'.format(path_files_saved))
            str_a = 'Area of PR curve is: {}, Area of ROC curve is: {}'.format(AUC_prec_rec, AUC_ROC)
            str_b = 'Accuracy average is: {}, std is: {}'.format(np.mean(acc), np.std(acc))
            str_c = 'Sensitivity average is: {}, std is: {}'.format(np.mean(sensitivity), np.std(sensitivity))
            str_d = 'Specificity average is: {}, std is: {}'.format(np.mean(specificity), np.std(specificity))
            str_e = 'Precision average is: {}, std is: {}'.format(np.mean(precision), np.std(precision))
            str_f = 'G average is: {}, std is: {}'.format(np.mean(G), np.std(G))
            str_g = 'F1_score average is:{}, std is: {}'.format(np.mean(F1_score), np.std(F1_score))
            str_h = 'Mse average is: {}, std is: {}'.format(np.mean(mse), np.std(mse))
            str_i = 'Iou average is: {}, std is: {}'.format(np.mean(iou), np.std(iou))
            str_j = 'Hausdorff_distance average is: {}, std is: {}'.format(np.mean(hausdorff_dis), np.std(hausdorff_dis))
            str_y = 'MCC average is: {}, std is: {}'.format(np.mean(mcc),np.std(mcc))
            str_z = 'AUC average is: {}, std is: {}'.format(np.mean(auc),np.std(auc))
            # str_k = 'hasu_xi average is: {}, std is: {}'.format(np.mean(hasu_xi), np.std(hasu_xi))
            # str_l = 'hasu_cu average is: {}, std is: {}'.format(np.mean(hasu_cu), np.std(hasu_cu))
            # str_o = 'SEN1_xi average is: {}, std is: {}'.format(np.mean(acc_xi), np.std(acc_xi))
            # str_p = 'SEN_cu average is: {}, std is: {}'.format(np.mean(acc_cu), np.std(acc_cu))
            # str_q = 'AUC_xi average is: {}, std is: {}'.format(np.mean(auc_xi), np.std(acc_xi))
            # str_r = 'AUC_cu average is: {}, std is: {}'.format(np.mean(auc_cu), np.std(acc_cu))

            # str_F = 'F average is: {}, std is: {}'.format(np.mean(F), np.std(F))
            # str_C = 'C average is: {}, std is: {}'.format(np.mean(C), np.std(C))
            # str_A = 'A average is: {}, std is: {}'.format(np.mean(A), np.std(A))
            # str_L = 'L average is: {}, std is: {}'.format(np.mean(L), np.std(L))
            f = open(path_files_saved, 'a', encoding='utf-8')
            f.write(str(i)+'.iter5'+'\n')
            f.write(str_a+'\n')
            f.write(str_b+'\n')
            f.write(str_c+'\n')
            f.write(str_d+'\n')
            f.write(str_e+'\n')
            f.write(str_i + '\n')
            f.write(str_g + '\n')
            f.write(str_f+'\n')
            f.write(str_h+'\n')
            f.write(str_j+'\n')
            f.write(str_z + '\n')
            f.write(str_y + '\n')
            # f.write(str_k + '\n')
            # f.write(str_l + '\n')
            # f.write(str_o + '\n')
            # f.write(str_p + '\n')
            # f.write(str_q + '\n')
            # f.write(str_r + '\n')

            # f.write(str_F + '\n')
            # f.write(str_C + '\n')
            # f.write(str_A + '\n')
            # f.write(str_L + '\n')
            f.write('\n')
            f.close()

def crop_eval(net, image, crop_size = Constants.resize_drive):
    '''

    :param net:
    :param image:     image is tensor form of [N, C, H, W], size is (584, 565)
    :param crop_size: 512 default
    :return:          584 , 565
    '''
    # print('val image size is:'.format(image.size()))

    d_h, d_w, h, w = image.size()[2] - crop_size, image.size()[3] - crop_size, image.size()[2],image.size()[3]
    crop_lu_im = image[:,:,0:h - d_h, 0:w-d_w]
    crop_ld_im = image[:,:,0:h - d_h, d_w:w]
    crop_ru_im = image[:,:,d_h:h, 0:w-d_w]
    crop_rd_im = image[:,:,d_h:h, d_w:w]

    crop_lu_im = crop_lu_im.cpu().numpy()
    crop_ld_im = crop_ld_im.cpu().numpy()
    crop_ru_im = crop_ru_im.cpu().numpy()
    crop_rd_im = crop_rd_im.cpu().numpy()


    crop_lu_im = dataset_normalized(crop_lu_im)
    crop_lu_im = clahe_equalized(crop_lu_im)
    crop_lu_im = adjust_gamma(crop_lu_im, 1.5)
    # crop_lu_im = adjust_contrast(crop_lu_im)
    # crop_lu_im = gaussianNoise(crop_lu_im)
    # crop_lu_im = adjust_gamma(crop_lu_im, 1.5)
    # crop_lu_im = dataset_normalized(crop_lu_im)
    # crop_lu_im = clahe_equalized(crop_lu_im)
    # crop_lu_im = blurr(crop_lu_im)
    # crop_lu_im = sharpen(crop_lu_im)
    # crop_lu_im = adjust_contrast(crop_lu_im)

    crop_ld_im = dataset_normalized(crop_ld_im)
    crop_ld_im = clahe_equalized(crop_ld_im)
    crop_ld_im = adjust_gamma(crop_ld_im, 1.5)
    crop_ld_im = adjust_contrast(crop_ld_im)
    # crop_ld_im = gaussianNoise(crop_ld_im)
    # crop_ld_im = adjust_gamma(crop_ld_im, 1.5)
    # crop_ld_im = dataset_normalized(crop_ld_im)
    # crop_ld_im = clahe_equalized(crop_ld_im)
    # crop_ld_im = blurr(crop_ld_im)
    # crop_ld_im = sharpen(crop_ld_im)
    # crop_ld_im = adjust_contrast(crop_ld_im)

    crop_ru_im = dataset_normalized(crop_ru_im)
    crop_ru_im = clahe_equalized(crop_ru_im)
    crop_ru_im = adjust_gamma(crop_ru_im, 1.5)
    crop_ru_im = adjust_contrast(crop_ru_im)
    # crop_ru_im = gaussianNoise(crop_ru_im)
    # crop_ru_im = adjust_gamma(crop_ru_im, 1.5)
    # crop_ru_im = dataset_normalized(crop_ru_im)
    # crop_ru_im = clahe_equalized(crop_ru_im)
    # crop_ru_im = blurr(crop_ru_im)
    # crop_ru_im = sharpen(crop_ru_im)
    # crop_ru_im = adjust_contrast(crop_ru_im)
    #
    crop_rd_im = dataset_normalized(crop_rd_im)
    crop_rd_im = clahe_equalized(crop_rd_im)
    crop_rd_im = adjust_gamma(crop_rd_im, 1.5)
    crop_rd_im = adjust_contrast(crop_rd_im)
    # crop_rd_im = gaussianNoise(crop_rd_im)
    # crop_rd_im = adjust_gamma(crop_rd_im, 1.5)
    # crop_rd_im = dataset_normalized(crop_rd_im)
    # crop_rd_im = clahe_equalized(crop_rd_im)
    # crop_rd_im = blurr(crop_rd_im)
    # crop_rd_im = sharpen(crop_rd_im)
    # crop_rd_im = adjust_contrast(crop_rd_im)
    # img = Image.fromarray( crop_rd_im[0, 0].astype(np.uint8))
    # img.save('/root/daima/YuzuSoft/' + str(0) + '.png')
    crop_lu_im = torch.from_numpy(crop_lu_im).cuda(device).float()
    crop_ld_im = torch.from_numpy(crop_ld_im).cuda(device).float()
    crop_ru_im = torch.from_numpy(crop_ru_im).cuda(device).float()
    crop_rd_im = torch.from_numpy(crop_rd_im).cuda(device).float()
    crop_lu_im = crop_lu_im / 255.
    crop_ld_im = crop_ld_im / 255.
    crop_ru_im = crop_ru_im / 255.
    crop_rd_im = crop_rd_im / 255.

    # img = Image.fromarray(crop_lu_im[0,0].astype(np.uint8))
    # img.save( '/root/daima/YuzuSoft/data_process/image_show/test/'+str(0)+'.png')
    # lu,_,_,_ = net(crop_lu_im)
    # ru,_,_,_ = net(crop_ld_im)
    # ld,_,_,_ = net(crop_ru_im)
    # rd,_,_,_ = net(crop_rd_im)

    # lu,lu1,lu2= net(crop_lu_im)
    # ru,ru1,ru2= net(crop_ld_im)
    # ld,ld1,ld2= net(crop_ru_im)
    # rd,rd1,rd2= net(crop_rd_im)

    lu= net(crop_lu_im)
    ru= net(crop_ld_im)
    ld= net(crop_ru_im)
    rd= net(crop_rd_im)
######################################################threshold-learning#######################################################

    # who = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))
    # for i in range(0, h):
    #     for j in range(0, w):
    #         if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
    #             who[:,:,i,j] = (lu1[:,:,i,j] + ru1[:,:,i,j-d_w] + ld1[:,:,i-d_h,j] + rd1[:,:,i-d_h,j-d_w]) /4
    #         if i>=0 and j >=0 and i<d_h and j<d_w:
    #             who[:, :, i, j] = lu1[:,:,i,j]
    #         if i>=0 and j >=d_w and i<d_h and j<crop_size:
    #             who[:, :, i, j] = (lu1[:,:,i,j] + ru1[:,:,i,j-d_w])/2
    #         if i>=0 and j >=crop_size and i<d_h:
    #             who[:, :, i, j] = ru1[:,:,i,j-d_w]
    #         if i>=d_h and j >=0 and i<crop_size and j<d_w:
    #             who[:, :, i, j] = (lu1[:,:,i,j] + ld1[:,:,i-d_h,j])/2
    #         if i>=d_h and j >=crop_size and i<crop_size:
    #             who[:, :, i, j] = (ru1[:,:,i,j-d_w] + rd1[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >=0 and j<d_w:
    #             who[:, :, i, j] = ld1[:,:,i-d_h,j]
    #         if i>=crop_size and j>=d_w and j <crop_size :
    #             who[:, :, i, j] = (ld1[:,:,i-d_h,j] + rd1[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >crop_size:
    #             who[:, :, i, j] = rd1[:,:,i-d_h,j-d_w]
    #
    #
    # d1 = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))
    # for i in range(0, h):
    #     for j in range(0, w):
    #         if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
    #             d1[:,:,i,j] = (lu2[:,:,i,j] + ru2[:,:,i,j-d_w] + ld2[:,:,i-d_h,j] + rd2[:,:,i-d_h,j-d_w]) /4
    #         if i>=0 and j >=0 and i<d_h and j<d_w:
    #             d1[:, :, i, j] = lu2[:,:,i,j]
    #         if i>=0 and j >=d_w and i<d_h and j<crop_size:
    #             d1[:, :, i, j] = (lu2[:,:,i,j] + ru2[:,:,i,j-d_w])/2
    #         if i>=0 and j >=crop_size and i<d_h:
    #             d1[:, :, i, j] = ru2[:,:,i,j-d_w]
    #         if i>=d_h and j >=0 and i<crop_size and j<d_w:
    #             d1[:, :, i, j] = (lu2[:,:,i,j] + ld2[:,:,i-d_h,j])/2
    #         if i>=d_h and j >=crop_size and i<crop_size:
    #             d1[:, :, i, j] = (ru2[:,:,i,j-d_w] + rd2[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >=0 and j<d_w:
    #             d1[:, :, i, j] = ld2[:,:,i-d_h,j]
    #         if i>=crop_size and j>=d_w and j <crop_size :
    #             d1[:, :, i, j] = (ld2[:,:,i-d_h,j] + rd2[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >crop_size:
    #             d1[:, :, i, j] = rd2[:,:,i-d_h,j-d_w]


    new_image = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))

    for i in range(0, h):
        for j in range(0, w):
            if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
                new_image[:,:,i,j] = (lu[:,:,i,j] + ru[:,:,i,j-d_w] + ld[:,:,i-d_h,j] + rd[:,:,i-d_h,j-d_w]) /4
            if i>=0 and j >=0 and i<d_h and j<d_w:
                new_image[:, :, i, j] = lu[:,:,i,j]
            if i>=0 and j >=d_w and i<d_h and j<crop_size:
                new_image[:, :, i, j] = (lu[:,:,i,j] + ru[:,:,i,j-d_w])/2
            if i>=0 and j >=crop_size and i<d_h:
                new_image[:, :, i, j] = ru[:,:,i,j-d_w]
            if i>=d_h and j >=0 and i<crop_size and j<d_w:
                new_image[:, :, i, j] = (lu[:,:,i,j] + ld[:,:,i-d_h,j])/2
            if i>=d_h and j >=crop_size and i<crop_size:
                new_image[:, :, i, j] = (ru[:,:,i,j-d_w] + rd[:,:,i-d_h,j-d_w])/2
            if i>=crop_size and j >=0 and j<d_w:
                new_image[:, :, i, j] = ld[:,:,i-d_h,j]
            if i>=crop_size and j>=d_w and j <crop_size :
                new_image[:, :, i, j] = (ld[:,:,i-d_h,j] + rd[:,:,i-d_h,j-d_w])/2
            if i>=crop_size and j >crop_size:
                new_image[:, :, i, j] = rd[:,:,i-d_h,j-d_w]
    # return new_image.to(device),who.to(device),d1.to(device)
    return new_image.to(device)

def crop_eval_skel(net, image,skel, crop_size = Constants.resize_drive):
    '''

    :param net:
    :param image:     image is tensor form of [N, C, H, W], size is (584, 565)
    :param crop_size: 512 default
    :return:          584 , 565
    '''
    # print('val image size is:'.format(image.size()))

    d_h, d_w, h, w = image.size()[2] - crop_size, image.size()[3] - crop_size, image.size()[2],image.size()[3]
    crop_lu_im = image[:,:,0:h - d_h, 0:w-d_w]
    crop_ld_im = image[:,:,0:h - d_h, d_w:w]
    crop_ru_im = image[:,:,d_h:h, 0:w-d_w]
    crop_rd_im = image[:,:,d_h:h, d_w:w]

    crop_lu_im = crop_lu_im.cpu().numpy()
    crop_ld_im = crop_ld_im.cpu().numpy()
    crop_ru_im = crop_ru_im.cpu().numpy()
    crop_rd_im = crop_rd_im.cpu().numpy()

    d_h, d_w, h, w = skel.size()[2] - crop_size, skel.size()[3] - crop_size, skel.size()[2],skel.size()[3]
    crop_lu_aa = skel[:,:,0:h - d_h, 0:w-d_w]
    crop_ld_aa = skel[:,:,0:h - d_h, d_w:w]
    crop_ru_aa = skel[:,:,d_h:h, 0:w-d_w]
    crop_rd_aa = skel[:,:,d_h:h, d_w:w]


    crop_lu_im = dataset_normalized(crop_lu_im)
    crop_lu_im = clahe_equalized(crop_lu_im)
    crop_lu_im = adjust_gamma(crop_lu_im, 1.5)
    crop_lu_im = adjust_contrast(crop_lu_im)
    # crop_lu_im = gaussianNoise(crop_lu_im)
    # crop_lu_im = adjust_gamma(crop_lu_im, 1.5)
    # crop_lu_im = dataset_normalized(crop_lu_im)
    # crop_lu_im = clahe_equalized(crop_lu_im)
    # crop_lu_im = blurr(crop_lu_im)
    # crop_lu_im = sharpen(crop_lu_im)
    # crop_lu_im = adjust_contrast(crop_lu_im)

    crop_ld_im = dataset_normalized(crop_ld_im)
    crop_ld_im = clahe_equalized(crop_ld_im)
    crop_ld_im = adjust_gamma(crop_ld_im, 1.5)
    crop_ld_im = adjust_contrast(crop_ld_im)
    # crop_ld_im = gaussianNoise(crop_ld_im)
    # crop_ld_im = adjust_gamma(crop_ld_im, 1.5)
    # crop_ld_im = dataset_normalized(crop_ld_im)
    # crop_ld_im = clahe_equalized(crop_ld_im)
    # crop_ld_im = blurr(crop_ld_im)
    # crop_ld_im = sharpen(crop_ld_im)
    # crop_ld_im = adjust_contrast(crop_ld_im)

    crop_ru_im = dataset_normalized(crop_ru_im)
    crop_ru_im = clahe_equalized(crop_ru_im)
    crop_ru_im = adjust_gamma(crop_ru_im, 1.5)
    crop_ru_im = adjust_contrast(crop_ru_im)
    # crop_ru_im = gaussianNoise(crop_ru_im)
    # crop_ru_im = adjust_gamma(crop_ru_im, 1.5)
    # crop_ru_im = dataset_normalized(crop_ru_im)
    # crop_ru_im = clahe_equalized(crop_ru_im)
    # crop_ru_im = blurr(crop_ru_im)
    # crop_ru_im = sharpen(crop_ru_im)
    # crop_ru_im = adjust_contrast(crop_ru_im)
    #
    crop_rd_im = dataset_normalized(crop_rd_im)
    crop_rd_im = clahe_equalized(crop_rd_im)
    crop_rd_im = adjust_gamma(crop_rd_im, 1.5)
    crop_rd_im = adjust_contrast(crop_rd_im)
    # crop_rd_im = gaussianNoise(crop_rd_im)
    # crop_rd_im = adjust_gamma(crop_rd_im, 1.5)
    # crop_rd_im = dataset_normalized(crop_rd_im)
    # crop_rd_im = clahe_equalized(crop_rd_im)
    # crop_rd_im = blurr(crop_rd_im)
    # crop_rd_im = sharpen(crop_rd_im)
    # crop_rd_im = adjust_contrast(crop_rd_im)
    # img = Image.fromarray( crop_rd_im[0, 0].astype(np.uint8))
    # img.save('/root/daima/YuzuSoft/' + str(0) + '.png')
    crop_lu_im = torch.from_numpy(crop_lu_im).cuda(device).float()
    crop_ld_im = torch.from_numpy(crop_ld_im).cuda(device).float()
    crop_ru_im = torch.from_numpy(crop_ru_im).cuda(device).float()
    crop_rd_im = torch.from_numpy(crop_rd_im).cuda(device).float()
    crop_lu_im = crop_lu_im / 255.
    crop_ld_im = crop_ld_im / 255.
    crop_ru_im = crop_ru_im / 255.
    crop_rd_im = crop_rd_im / 255.

    # img = Image.fromarray(crop_lu_im[0,0].astype(np.uint8))
    # img.save( '/root/daima/YuzuSoft/data_process/image_show/test/'+str(0)+'.png')
    # lu,_,_,_ = net(crop_lu_im)
    # ru,_,_,_ = net(crop_ld_im)
    # ld,_,_,_ = net(crop_ru_im)
    # rd,_,_,_ = net(crop_rd_im)

    # lu,lu1,lu2= net(crop_lu_im)
    # ru,ru1,ru2= net(crop_ld_im)
    # ld,ld1,ld2= net(crop_ru_im)
    # rd,rd1,rd2= net(crop_rd_im)

    lu= net(crop_lu_im,crop_lu_aa)
    ru= net(crop_ld_im,crop_ld_aa)
    ld= net(crop_ru_im,crop_ru_aa)
    rd= net(crop_rd_im,crop_rd_aa)
######################################################threshold-learning#######################################################

    # who = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))
    # for i in range(0, h):
    #     for j in range(0, w):
    #         if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
    #             who[:,:,i,j] = (lu1[:,:,i,j] + ru1[:,:,i,j-d_w] + ld1[:,:,i-d_h,j] + rd1[:,:,i-d_h,j-d_w]) /4
    #         if i>=0 and j >=0 and i<d_h and j<d_w:
    #             who[:, :, i, j] = lu1[:,:,i,j]
    #         if i>=0 and j >=d_w and i<d_h and j<crop_size:
    #             who[:, :, i, j] = (lu1[:,:,i,j] + ru1[:,:,i,j-d_w])/2
    #         if i>=0 and j >=crop_size and i<d_h:
    #             who[:, :, i, j] = ru1[:,:,i,j-d_w]
    #         if i>=d_h and j >=0 and i<crop_size and j<d_w:
    #             who[:, :, i, j] = (lu1[:,:,i,j] + ld1[:,:,i-d_h,j])/2
    #         if i>=d_h and j >=crop_size and i<crop_size:
    #             who[:, :, i, j] = (ru1[:,:,i,j-d_w] + rd1[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >=0 and j<d_w:
    #             who[:, :, i, j] = ld1[:,:,i-d_h,j]
    #         if i>=crop_size and j>=d_w and j <crop_size :
    #             who[:, :, i, j] = (ld1[:,:,i-d_h,j] + rd1[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >crop_size:
    #             who[:, :, i, j] = rd1[:,:,i-d_h,j-d_w]
    #
    #
    # d1 = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))
    # for i in range(0, h):
    #     for j in range(0, w):
    #         if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
    #             d1[:,:,i,j] = (lu2[:,:,i,j] + ru2[:,:,i,j-d_w] + ld2[:,:,i-d_h,j] + rd2[:,:,i-d_h,j-d_w]) /4
    #         if i>=0 and j >=0 and i<d_h and j<d_w:
    #             d1[:, :, i, j] = lu2[:,:,i,j]
    #         if i>=0 and j >=d_w and i<d_h and j<crop_size:
    #             d1[:, :, i, j] = (lu2[:,:,i,j] + ru2[:,:,i,j-d_w])/2
    #         if i>=0 and j >=crop_size and i<d_h:
    #             d1[:, :, i, j] = ru2[:,:,i,j-d_w]
    #         if i>=d_h and j >=0 and i<crop_size and j<d_w:
    #             d1[:, :, i, j] = (lu2[:,:,i,j] + ld2[:,:,i-d_h,j])/2
    #         if i>=d_h and j >=crop_size and i<crop_size:
    #             d1[:, :, i, j] = (ru2[:,:,i,j-d_w] + rd2[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >=0 and j<d_w:
    #             d1[:, :, i, j] = ld2[:,:,i-d_h,j]
    #         if i>=crop_size and j>=d_w and j <crop_size :
    #             d1[:, :, i, j] = (ld2[:,:,i-d_h,j] + rd2[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >crop_size:
    #             d1[:, :, i, j] = rd2[:,:,i-d_h,j-d_w]


    new_image = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))

    for i in range(0, h):
        for j in range(0, w):
            if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
                new_image[:,:,i,j] = (lu[:,:,i,j] + ru[:,:,i,j-d_w] + ld[:,:,i-d_h,j] + rd[:,:,i-d_h,j-d_w]) /4
            if i>=0 and j >=0 and i<d_h and j<d_w:
                new_image[:, :, i, j] = lu[:,:,i,j]
            if i>=0 and j >=d_w and i<d_h and j<crop_size:
                new_image[:, :, i, j] = (lu[:,:,i,j] + ru[:,:,i,j-d_w])/2
            if i>=0 and j >=crop_size and i<d_h:
                new_image[:, :, i, j] = ru[:,:,i,j-d_w]
            if i>=d_h and j >=0 and i<crop_size and j<d_w:
                new_image[:, :, i, j] = (lu[:,:,i,j] + ld[:,:,i-d_h,j])/2
            if i>=d_h and j >=crop_size and i<crop_size:
                new_image[:, :, i, j] = (ru[:,:,i,j-d_w] + rd[:,:,i-d_h,j-d_w])/2
            if i>=crop_size and j >=0 and j<d_w:
                new_image[:, :, i, j] = ld[:,:,i-d_h,j]
            if i>=crop_size and j>=d_w and j <crop_size :
                new_image[:, :, i, j] = (ld[:,:,i-d_h,j] + rd[:,:,i-d_h,j-d_w])/2
            if i>=crop_size and j >crop_size:
                new_image[:, :, i, j] = rd[:,:,i-d_h,j-d_w]
    # return new_image.to(device),who.to(device),d1.to(device)
    return new_image.to(device)


def crop_eval_2(net, image, crop_size = Constants.resize_drive):
    '''

    :param net:
    :param image:     image is tensor form of [N, C, H, W], size is (584, 565)
    :param crop_size: 512 default
    :return:          584 , 565
    '''
    # print('val image size is:'.format(image.size()))

    d_h, d_w, h, w = image.size()[2] - crop_size, image.size()[3] - crop_size, image.size()[2],image.size()[3]
    crop_lu_im = image[:,:,0:h - d_h, 0:w-d_w]
    crop_ld_im = image[:,:,0:h - d_h, d_w:w]
    crop_ru_im = image[:,:,d_h:h, 0:w-d_w]
    crop_rd_im = image[:,:,d_h:h, d_w:w]

    crop_lu_im = crop_lu_im.cpu().numpy()
    crop_ld_im = crop_ld_im.cpu().numpy()
    crop_ru_im = crop_ru_im.cpu().numpy()
    crop_rd_im = crop_rd_im.cpu().numpy()


    crop_lu_im = dataset_normalized(crop_lu_im)
    crop_lu_im = clahe_equalized(crop_lu_im)
    crop_lu_im = adjust_gamma(crop_lu_im, 1.0)
    # crop_lu_im = adjust_contrast(crop_lu_im)
    # crop_lu_im = gaussianNoise(crop_lu_im)
    # crop_lu_im = adjust_gamma(crop_lu_im, 1.5)
    # crop_lu_im = dataset_normalized(crop_lu_im)
    # crop_lu_im = clahe_equalized(crop_lu_im)
    # crop_lu_im = blurr(crop_lu_im)
    # crop_lu_im = sharpen(crop_lu_im)
    # crop_lu_im = adjust_contrast(crop_lu_im)

    crop_ld_im = dataset_normalized(crop_ld_im)
    crop_ld_im = clahe_equalized(crop_ld_im)
    crop_ld_im = adjust_gamma(crop_ld_im, 1.0)
    # crop_ld_im = adjust_contrast(crop_ld_im)
    # crop_ld_im = gaussianNoise(crop_ld_im)
    # crop_ld_im = adjust_gamma(crop_ld_im, 1.5)
    # crop_ld_im = dataset_normalized(crop_ld_im)
    # crop_ld_im = clahe_equalized(crop_ld_im)
    # crop_ld_im = blurr(crop_ld_im)
    # crop_ld_im = sharpen(crop_ld_im)
    # crop_ld_im = adjust_contrast(crop_ld_im)

    crop_ru_im = dataset_normalized(crop_ru_im)
    crop_ru_im = clahe_equalized(crop_ru_im)
    crop_ru_im = adjust_gamma(crop_ru_im, 1.0)
    # crop_ru_im = adjust_contrast(crop_ru_im)
    # crop_ru_im = gaussianNoise(crop_ru_im)
    # crop_ru_im = adjust_gamma(crop_ru_im, 1.5)
    # crop_ru_im = dataset_normalized(crop_ru_im)
    # crop_ru_im = clahe_equalized(crop_ru_im)
    # crop_ru_im = blurr(crop_ru_im)
    # crop_ru_im = sharpen(crop_ru_im)
    # crop_ru_im = adjust_contrast(crop_ru_im)
    #
    crop_rd_im = dataset_normalized(crop_rd_im)
    crop_rd_im = clahe_equalized(crop_rd_im)
    crop_rd_im = adjust_gamma(crop_rd_im, 1.0)
    # crop_rd_im = adjust_contrast(crop_rd_im)
    # crop_rd_im = gaussianNoise(crop_rd_im)
    # crop_rd_im = adjust_gamma(crop_rd_im, 1.5)
    # crop_rd_im = dataset_normalized(crop_rd_im)
    # crop_rd_im = clahe_equalized(crop_rd_im)
    # crop_rd_im = blurr(crop_rd_im)
    # crop_rd_im = sharpen(crop_rd_im)
    # crop_rd_im = adjust_contrast(crop_rd_im)
    # img = Image.fromarray( crop_rd_im[0, 0].astype(np.uint8))
    # img.save('/root/daima/YuzuSoft/' + str(0) + '.png')
    crop_lu_im = torch.from_numpy(crop_lu_im).cuda(device).float()
    crop_ld_im = torch.from_numpy(crop_ld_im).cuda(device).float()
    crop_ru_im = torch.from_numpy(crop_ru_im).cuda(device).float()
    crop_rd_im = torch.from_numpy(crop_rd_im).cuda(device).float()
    crop_lu_im = crop_lu_im / 255.
    crop_ld_im = crop_ld_im / 255.
    crop_ru_im = crop_ru_im / 255.
    crop_rd_im = crop_rd_im / 255.

    # img = Image.fromarray(crop_lu_im[0,0].astype(np.uint8))
    # img.save( '/root/daima/YuzuSoft/data_process/image_show/test/'+str(0)+'.png')
    # lu,_,_,_ = net(crop_lu_im)
    # ru,_,_,_ = net(crop_ld_im)
    # ld,_,_,_ = net(crop_ru_im)
    # rd,_,_,_ = net(crop_rd_im)

    # lu,lu1,lu2= net(crop_lu_im)
    # ru,ru1,ru2= net(crop_ld_im)
    # ld,ld1,ld2= net(crop_ru_im)
    # rd,rd1,rd2= net(crop_rd_im)

    lu,lu1= net(crop_lu_im)
    ru,ru1= net(crop_ld_im)
    ld,ld1= net(crop_ru_im)
    rd,rd1= net(crop_rd_im)
######################################################threshold-learning#######################################################

    who = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))
    for i in range(0, h):
        for j in range(0, w):
            if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
                who[:,:,i,j] = (lu1[:,:,i,j] + ru1[:,:,i,j-d_w] + ld1[:,:,i-d_h,j] + rd1[:,:,i-d_h,j-d_w]) /4
            if i>=0 and j >=0 and i<d_h and j<d_w:
                who[:, :, i, j] = lu1[:,:,i,j]
            if i>=0 and j >=d_w and i<d_h and j<crop_size:
                who[:, :, i, j] = (lu1[:,:,i,j] + ru1[:,:,i,j-d_w])/2
            if i>=0 and j >=crop_size and i<d_h:
                who[:, :, i, j] = ru1[:,:,i,j-d_w]
            if i>=d_h and j >=0 and i<crop_size and j<d_w:
                who[:, :, i, j] = (lu1[:,:,i,j] + ld1[:,:,i-d_h,j])/2
            if i>=d_h and j >=crop_size and i<crop_size:
                who[:, :, i, j] = (ru1[:,:,i,j-d_w] + rd1[:,:,i-d_h,j-d_w])/2
            if i>=crop_size and j >=0 and j<d_w:
                who[:, :, i, j] = ld1[:,:,i-d_h,j]
            if i>=crop_size and j>=d_w and j <crop_size :
                who[:, :, i, j] = (ld1[:,:,i-d_h,j] + rd1[:,:,i-d_h,j-d_w])/2
            if i>=crop_size and j >crop_size:
                who[:, :, i, j] = rd1[:,:,i-d_h,j-d_w]
    #
    #
    # d1 = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))
    # for i in range(0, h):
    #     for j in range(0, w):
    #         if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
    #             d1[:,:,i,j] = (lu2[:,:,i,j] + ru2[:,:,i,j-d_w] + ld2[:,:,i-d_h,j] + rd2[:,:,i-d_h,j-d_w]) /4
    #         if i>=0 and j >=0 and i<d_h and j<d_w:
    #             d1[:, :, i, j] = lu2[:,:,i,j]
    #         if i>=0 and j >=d_w and i<d_h and j<crop_size:
    #             d1[:, :, i, j] = (lu2[:,:,i,j] + ru2[:,:,i,j-d_w])/2
    #         if i>=0 and j >=crop_size and i<d_h:
    #             d1[:, :, i, j] = ru2[:,:,i,j-d_w]
    #         if i>=d_h and j >=0 and i<crop_size and j<d_w:
    #             d1[:, :, i, j] = (lu2[:,:,i,j] + ld2[:,:,i-d_h,j])/2
    #         if i>=d_h and j >=crop_size and i<crop_size:
    #             d1[:, :, i, j] = (ru2[:,:,i,j-d_w] + rd2[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >=0 and j<d_w:
    #             d1[:, :, i, j] = ld2[:,:,i-d_h,j]
    #         if i>=crop_size and j>=d_w and j <crop_size :
    #             d1[:, :, i, j] = (ld2[:,:,i-d_h,j] + rd2[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >crop_size:
    #             d1[:, :, i, j] = rd2[:,:,i-d_h,j-d_w]


    new_image = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))

    for i in range(0, h):
        for j in range(0, w):
            if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
                new_image[:,:,i,j] = (lu[:,:,i,j] + ru[:,:,i,j-d_w] + ld[:,:,i-d_h,j] + rd[:,:,i-d_h,j-d_w]) /4
            if i>=0 and j >=0 and i<d_h and j<d_w:
                new_image[:, :, i, j] = lu[:,:,i,j]
            if i>=0 and j >=d_w and i<d_h and j<crop_size:
                new_image[:, :, i, j] = (lu[:,:,i,j] + ru[:,:,i,j-d_w])/2
            if i>=0 and j >=crop_size and i<d_h:
                new_image[:, :, i, j] = ru[:,:,i,j-d_w]
            if i>=d_h and j >=0 and i<crop_size and j<d_w:
                new_image[:, :, i, j] = (lu[:,:,i,j] + ld[:,:,i-d_h,j])/2
            if i>=d_h and j >=crop_size and i<crop_size:
                new_image[:, :, i, j] = (ru[:,:,i,j-d_w] + rd[:,:,i-d_h,j-d_w])/2
            if i>=crop_size and j >=0 and j<d_w:
                new_image[:, :, i, j] = ld[:,:,i-d_h,j]
            if i>=crop_size and j>=d_w and j <crop_size :
                new_image[:, :, i, j] = (ld[:,:,i-d_h,j] + rd[:,:,i-d_h,j-d_w])/2
            if i>=crop_size and j >crop_size:
                new_image[:, :, i, j] = rd[:,:,i-d_h,j-d_w]

    who = torch.zeros_like(torch.unsqueeze(image[:, 0, :, :, ], dim=1))
    for i in range(0, h):
        for j in range(0, w):
            if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
                who[:,:,i,j] = (lu1[:,:,i,j] + ru1[:,:,i,j-d_w] + ld1[:,:,i-d_h,j] + rd1[:,:,i-d_h,j-d_w]) /4
            if i>=0 and j >=0 and i<d_h and j<d_w:
                who[:, :, i, j] = lu1[:,:,i,j]
            if i>=0 and j >=d_w and i<d_h and j<crop_size:
                who[:, :, i, j] = (lu1[:,:,i,j] + ru1[:,:,i,j-d_w])/2
            if i>=0 and j >=crop_size and i<d_h:
                who[:, :, i, j] = ru1[:,:,i,j-d_w]
            if i>=d_h and j >=0 and i<crop_size and j<d_w:
                who[:, :, i, j] = (lu1[:,:,i,j] + ld1[:,:,i-d_h,j])/2
            if i>=d_h and j >=crop_size and i<crop_size:
                who[:, :, i, j] = (ru1[:,:,i,j-d_w] + rd1[:,:,i-d_h,j-d_w])/2
            if i>=crop_size and j >=0 and j<d_w:
                who[:, :, i, j] = ld1[:,:,i-d_h,j]
            if i>=crop_size and j>=d_w and j <crop_size :
                who[:, :, i, j] = (ld1[:,:,i-d_h,j] + rd1[:,:,i-d_h,j-d_w])/2
            if i>=crop_size and j >crop_size:
                who[:, :, i, j] = rd1[:,:,i-d_h,j-d_w]
    return new_image.to(device),who.to(device)
    # return new_image.to(device)


def crop_eval_3(net, image, crop_size = Constants.resize_drive):
    '''

    :param net:
    :param image:     image is tensor form of [N, C, H, W], size is (584, 565)
    :param crop_size: 512 default
    :return:          584 , 565
    '''
    # print('val image size is:'.format(image.size()))
    d_h, d_w, h, w = image.size()[2] - crop_size, image.size()[3] - crop_size, image.size()[2],image.size()[3]
    crop_lu_im = image[:,:,0:h - d_h, 0:w-d_w]
    crop_ld_im = image[:,:,0:h - d_h, d_w:w]
    crop_ru_im = image[:,:,d_h:h, 0:w-d_w]
    crop_rd_im = image[:,:,d_h:h, d_w:w]

    crop_lu_im = crop_lu_im.cpu().numpy()
    crop_ld_im = crop_ld_im.cpu().numpy()
    crop_ru_im = crop_ru_im.cpu().numpy()
    crop_rd_im = crop_rd_im.cpu().numpy()


    crop_lu_im = dataset_normalized(crop_lu_im)
    crop_lu_im = clahe_equalized(crop_lu_im)
    crop_lu_im = adjust_gamma(crop_lu_im, 1.5)
    crop_lu_im = adjust_contrast(crop_lu_im)

    crop_ld_im = dataset_normalized(crop_ld_im)
    crop_ld_im = clahe_equalized(crop_ld_im)
    crop_ld_im = adjust_gamma(crop_ld_im, 1.5)
    crop_ld_im = adjust_contrast(crop_ld_im)

    crop_ru_im = dataset_normalized(crop_ru_im)
    crop_ru_im = clahe_equalized(crop_ru_im)
    crop_ru_im = adjust_gamma(crop_ru_im, 1.5)
    crop_ru_im = adjust_contrast(crop_ru_im)

    crop_rd_im = dataset_normalized(crop_rd_im)
    crop_rd_im = clahe_equalized(crop_rd_im)
    crop_rd_im = adjust_gamma(crop_rd_im, 1.5)
    crop_rd_im = adjust_contrast(crop_rd_im)
    # img = Image.fromarray( crop_rd_im[0, 0].astype(np.uint8))
    # img.save('/root/daima/YuzuSoft/' + str(0) + '.png')
    crop_lu_im = torch.from_numpy(crop_lu_im).cuda(device).float()
    crop_ld_im = torch.from_numpy(crop_ld_im).cuda(device).float()
    crop_ru_im = torch.from_numpy(crop_ru_im).cuda(device).float()
    crop_rd_im = torch.from_numpy(crop_rd_im).cuda(device).float()
    crop_lu_im = crop_lu_im / 255.
    crop_ld_im = crop_ld_im / 255.
    crop_ru_im = crop_ru_im / 255.
    crop_rd_im = crop_rd_im / 255.

    # img = Image.fromarray(crop_lu_im[0,0].astype(np.uint8))
    # img.save( '/root/daima/YuzuSoft/data_process/image_show/test/'+str(0)+'.png')
    # lu,_,_,_ = net(crop_lu_im)
    # ru,_,_,_ = net(crop_ld_im)
    # ld,_,_,_ = net(crop_ru_im)
    # rd,_,_,_ = net(crop_rd_im)
    lu,lu1,lu2= net(crop_lu_im)

    ru,ru1,ru2= net(crop_ld_im)

    ld,ld1,ld2= net(crop_ru_im)

    rd,rd1,rd2= net(crop_rd_im)
    # affr = Affr()
    # lu = affr(d = lu,x = crop_lu_im)
    # ru = affr(d=ru, x=crop_ld_im)
    # ld = affr(d=ld, x=crop_ru_im)
    # rd = affr(d=rd, x=crop_rd_im)

    # lu2 = F.interpolate(lu2, size=(512, 512), mode='bilinear')
    # ru2 = F.interpolate(ru2, size=(512, 512), mode='bilinear')
    # ld2 = F.interpolate(ld2, size=(512, 512), mode='bilinear')
    # rd2 = F.interpolate(rd2, size=(512, 512), mode='bilinear')

######################################################threshold-learning#######################################################
    new_image = torch.zeros_like(torch.unsqueeze(image[:, 0, :, :, ], dim=1))

    for i in range(0, h):
        for j in range(0, w):
            if i >= d_h and j >= d_w and i < crop_size and j < crop_size:
                new_image[:, :, i, j] = (lu[:, :, i, j] + ru[:, :, i, j - d_w] + ld[:, :, i - d_h, j] + rd[:, :,
                                                                                                        i - d_h,
                                                                                                        j - d_w]) / 4
            if i >= 0 and j >= 0 and i < d_h and j < d_w:
                new_image[:, :, i, j] = lu[:, :, i, j]
            if i >= 0 and j >= d_w and i < d_h and j < crop_size:
                new_image[:, :, i, j] = (lu[:, :, i, j] + ru[:, :, i, j - d_w]) / 2
            if i >= 0 and j >= crop_size and i < d_h:
                new_image[:, :, i, j] = ru[:, :, i, j - d_w]
            if i >= d_h and j >= 0 and i < crop_size and j < d_w:
                new_image[:, :, i, j] = (lu[:, :, i, j] + ld[:, :, i - d_h, j]) / 2
            if i >= d_h and j >= crop_size and i < crop_size:
                new_image[:, :, i, j] = (ru[:, :, i, j - d_w] + rd[:, :, i - d_h, j - d_w]) / 2
            if i >= crop_size and j >= 0 and j < d_w:
                new_image[:, :, i, j] = ld[:, :, i - d_h, j]
            if i >= crop_size and j >= d_w and j < crop_size:
                new_image[:, :, i, j] = (ld[:, :, i - d_h, j] + rd[:, :, i - d_h, j - d_w]) / 2
            if i >= crop_size and j > crop_size:
                new_image[:, :, i, j] = rd[:, :, i - d_h, j - d_w]

    who = torch.zeros_like(torch.unsqueeze(image[:, 0, :, :, ], dim=1))
    # for i in range(0, h):
    #     for j in range(0, w):
    #         if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
    #             who[:,:,i,j] = (lu1[:,:,i,j] + ru1[:,:,i,j-d_w] + ld1[:,:,i-d_h,j] + rd1[:,:,i-d_h,j-d_w]) /4
    #         if i>=0 and j >=0 and i<d_h and j<d_w:
    #             who[:, :, i, j] = lu1[:,:,i,j]
    #         if i>=0 and j >=d_w and i<d_h and j<crop_size:
    #             who[:, :, i, j] = (lu1[:,:,i,j] + ru1[:,:,i,j-d_w])/2
    #         if i>=0 and j >=crop_size and i<d_h:
    #             who[:, :, i, j] = ru1[:,:,i,j-d_w]
    #         if i>=d_h and j >=0 and i<crop_size and j<d_w:
    #             who[:, :, i, j] = (lu1[:,:,i,j] + ld1[:,:,i-d_h,j])/2
    #         if i>=d_h and j >=crop_size and i<crop_size:
    #             who[:, :, i, j] = (ru1[:,:,i,j-d_w] + rd1[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >=0 and j<d_w:
    #             who[:, :, i, j] = ld1[:,:,i-d_h,j]
    #         if i>=crop_size and j>=d_w and j <crop_size :
    #             who[:, :, i, j] = (ld1[:,:,i-d_h,j] + rd1[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >crop_size:
    #             who[:, :, i, j] = rd1[:,:,i-d_h,j-d_w]
    #
    #


    d1 = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))
    # for i in range(0, h):
    #     for j in range(0, w):
    #         if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
    #             d1[:,:,i,j] = (lu2[:,:,i,j] + ru2[:,:,i,j-d_w] + ld2[:,:,i-d_h,j] + rd2[:,:,i-d_h,j-d_w]) /4
    #         if i>=0 and j >=0 and i<d_h and j<d_w:
    #             d1[:, :, i, j] = lu2[:,:,i,j]
    #         if i>=0 and j >=d_w and i<d_h and j<crop_size:
    #             d1[:, :, i, j] = (lu2[:,:,i,j] + ru2[:,:,i,j-d_w])/2
    #         if i>=0 and j >=crop_size and i<d_h:
    #             d1[:, :, i, j] = ru2[:,:,i,j-d_w]
    #         if i>=d_h and j >=0 and i<crop_size and j<d_w:
    #             d1[:, :, i, j] = (lu2[:,:,i,j] + ld2[:,:,i-d_h,j])/2
    #         if i>=d_h and j >=crop_size and i<crop_size:
    #             d1[:, :, i, j] = (ru2[:,:,i,j-d_w] + rd2[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >=0 and j<d_w:
    #             d1[:, :, i, j] = ld2[:,:,i-d_h,j]
    #         if i>=crop_size and j>=d_w and j <crop_size :
    #             d1[:, :, i, j] = (ld2[:,:,i-d_h,j] + rd2[:,:,i-d_h,j-d_w])/2
    #         if i>=crop_size and j >crop_size:
    #             d1[:, :, i, j] = rd2[:,:,i-d_h,j-d_w]



    return new_image.to(device),who.to(device),d1.to(device)
    # return new_image.to(device)

def crop_eval_3to4(net, image, crop_size = Constants.resize_drive):
    '''

    :param net:
    :param image:     image is tensor form of [N, C, H, W], size is (584, 565)
    :param crop_size: 512 default
    :return:          584 , 565
    '''
    # print('val image size is:'.format(image.size()))


    crop_lu_im = image.cpu().numpy()



    crop_lu_im = dataset_normalized(crop_lu_im)
    crop_lu_im = clahe_equalized(crop_lu_im)
    crop_lu_im = adjust_gamma(crop_lu_im, 1.5)
    crop_lu_im = adjust_contrast(crop_lu_im)


    # img = Image.fromarray( crop_rd_im[0, 0].astype(np.uint8))
    # img.save('/root/daima/YuzuSoft/' + str(0) + '.png')
    crop_lu_im = torch.from_numpy(crop_lu_im).cuda(device).float()

    crop_lu_im = crop_lu_im / 255.


    # img = Image.fromarray(crop_lu_im[0,0].astype(np.uint8))
    # img.save( '/root/daima/YuzuSoft/data_process/image_show/test/'+str(0)+'.png')
    # lu,_,_,_ = net(crop_lu_im)
    # ru,_,_,_ = net(crop_ld_im)
    # ld,_,_,_ = net(crop_ru_im)
    # rd,_,_,_ = net(crop_rd_im)
    lu,lu1,lu2= net(crop_lu_im)




    return lu.to(device)

def read_numpy_into_npy(arrays, path):
    np.save(path, arrays)
    print('have saved all arrays in to path ', path)
def test_vessel_train(path_1, ch = Constants.BINARY_CLASS):
    images, masks = get_drive_data_skel(is_train=False)
    # images, masks = get_drive_data_skel(is_train=False)
    test_list = ['our_DPF']
    # pre = PreWho1()
    for i in test_list:
        path = path_1 + '/STARE/' +str(i) + '.iter5'
        print('test:', i)
        acc, sensitivity, specificity, precision, G, F1_score, mse, iou, hausdorff_dis,hasu_xi,hasu_cu,acc_xi,acc_cu,auc,auc_xi,auc_cu= [], [], [], [], [], [], [], [], [],[],[],[],[],[],[],[]
        pr_g, pr_l = [], []
        mcc = []
        F, C, A, L, rSe, rSp, rAcc = [], [], [], [], [], [], []
        # net = UU_Net().to(device)
        # net = Who_Net().to(device)
        # net = img_patch().to(device)
        # net = U1_Net().to(device)
        # net = FCDenseNet().to(device)
        # net = Ue_Net().to(device)
        # net = CE_Net_().to(device)
        # net = JTFN().to(device)
        # net = Lits().to(device)
        # net.load_state_dict(torch.load(path))
        net = load_model(path)
        # skel_images = np.empty(shape=(10, 3, 605, 700))  # 20,2,584,565#720,2,512, 512
        skel_images = np.empty(shape=(720, 3, 512, 512))  # 20,2,584,565#720,2,512, 512
        # skel_images = np.empty(shape=(8, 3, 960,999))  # 20,2,584,565#720,2,512, 512
        # skel_images = np.empty(shape=(720, 3, 960, 960))  # 20,2,584,565#720,2,512, 512
        # skel_images = np.empty(shape=(3600, 3, 288, 288))  # 20,2,584,565#720,2,512, 512
        # skel_images = np.empty(shape=(34, 3, 300, 300))  # 20,2,584,565#720,2,512, 512
        # skel_images = np.empty(shape=(20,3,584,565))#20,2,584,565#720,2,512, 512
        # skel_images = np.empty(shape=(images.shape[0], 3, 512, 512))  # 20,2,584,565#720,2,512, 512
        net.eval()
        with torch.no_grad():
            for iter_ in range(0,images.shape[0]):
                x_img = images[iter_]
                x_img = np.expand_dims(x_img, axis=0)
                x_img = torch.tensor(x_img, dtype=torch.float32).to(device)

                # generated_vessel,_,_ = crop_eval_3(net, x_img)#test
                generated_vessel = crop_eval_3to4(net, x_img)#train
                print('OK')
                # generated_vessel, patch_out, patch_128 = crop_eval_3(net, x_img)
                # generated_vessel = crop_eval(net, x_img)
                # generated_vessel1 = generated_vessel1.permute((0, 2, 3, 1)).detach().cpu().numpy()
                generated_vessel = generated_vessel.permute((0, 2, 3, 1)).detach().cpu().numpy()
                skel_images[iter_, 0, :, :],skel_images[iter_, 1, :, :] ,skel_images[iter_, 2, :, :]= img_skel(generated_vessel[0,:,:,0])
                # skel_images[iter_, 0, :, :], skel_images[iter_, 1, :, :], skel_images[iter_, 2, :, :] = img_skel(masks[iter_, 0, :, :])
                print("skeling"+str(iter_)+"!")
                pic = Image.fromarray((255 * skel_images[iter_, 0]).astype(np.uint8))
                pic.save('/root/daima/YuzuSoft/log/visual_results/skel1/' + str(iter_) + '.png')
                pic = Image.fromarray((255 * skel_images[iter_, 1]).astype(np.uint8))
                pic.save('/root/daima/YuzuSoft/log/visual_results/skel2/' + str(iter_) + '.png')
                pic = Image.fromarray((255 * skel_images[iter_, 2]).astype(np.uint8))
                pic.save('/root/daima/YuzuSoft/log/visual_results/point/' + str(iter_) + '.png')

                # patch_out = patch_out.permute((0, 2, 3, 1)).detach().cpu().numpy()
                # patch_128 = patch_128.permute((0, 2, 3, 1)).detach().cpu().numpy()

                # if ch == 1:  # for [N,1,H,W]
                #     pr_g.append(generated_vessel.reshape((-1,)).tolist())
                #     pr_l.append(masks[iter_].reshape((-1,)).tolist())
                #     # visualize(np.asarray(generated_vessel[0, :, :, :, ]), Constants.visual_results + 'prob_'+ str(iter_) )
                #     threshold = 0.5                                          # for [N,H,W,1]
                #     generated_vessel[generated_vessel >= threshold] = 1
                #     generated_vessel[generated_vessel < threshold] = 0
                #     # patch_out[patch_out >= 0.35] = 1
                #     # patch_out[patch_out < 0.35] = 0
                #
                #     # patch_128[patch_128 >= threshold] = 1
                #     # patch_128[patch_128 < threshold] = 0
                #     # generated_vessel1[generated_vessel1 >= threshold] = 1
                #     # generated_vessel1[generated_vessel1 < threshold] = 0
                #     # generated_vessel = generated_vessel + generated_vessel1
                #     # generated_vessel[generated_vessel == 2]=1
                #     # generated_vessel = threshold_by_otsu(generated_vessel)
                # if ch == 2:  # for [N,H,W,2]
                #     generated_vessel = np.expand_dims(np.argmax(generated_vessel, axis=3), axis=3)  # for [N,H,W,2]
                #     pr_g.append(generated_vessel.reshape((-1,)).tolist())
                #     pr_l.append(masks[iter_].reshape((-1,)).tolist())
                #
                #
                #
                # generated_vessel = np.squeeze(generated_vessel, axis=0)  # (1, H, W, 1) to (H, W)
                # # visualize(np.asarray(generated_vessel),Constants.visual_results + str(iter_)+ 'seg')
                # # patch_out = np.squeeze(patch_out, axis=0)
                # # patch_128 = np.squeeze(patch_128, axis=0)
                # # visualize(np.asarray(patch_out), Constants.visual_results + 'img_val_pic' + str(iter_))  # 记得删
                # # visualize(np.asarray(patch_128), Constants.visual_results + 'edge_val_pic' + str(iter_))  # 记得删
                # visualize(np.asarray(generated_vessel), Constants.visual_results + 'pre_' + str(iter_) )
                # retina_color_different(np.asarray(generated_vessel), masks[iter_].transpose((1, 2, 0)), Constants.visual_results + 'different_'+ str(iter_) )
        # for i in range(0,images.shape[0]):#720
        #     pic = Image.fromarray((255*skel_images[i,0]).astype(np.uint8))
        #     pic.save('/root/daima/YuzuSoft/log/visual_results/skel1/' + str(i) +'.png')
        #     pic = Image.fromarray((255*skel_images[i, 1]).astype(np.uint8))
        #     pic.save('/root/daima/YuzuSoft/log/visual_results/skel2/' + str(i) + '.png')
        read_numpy_into_npy(skel_images,Constants.path_skel)
        # read_numpy_into_npy(skel_images,Constants.path_skel_test)#当前使用的3通道的数，各有不同含义，保存在_3.npy


def test_vessel_train_2(path_1, ch = Constants.BINARY_CLASS):
    images, masks = get_drive_data_skel(is_train=False)
    # images, masks = get_drive_data_skel(is_train=False)
    test_list = ['our_DPF']
    # skel = load_from_npy(Constants.path_skel)  ####

    # pre = PreWho1()
    for i in test_list:
        path = path_1 + '/DCA1/' +str(i) + '.iter5'
        print('test:', i)
        acc, sensitivity, specificity, precision, G, F1_score, mse, iou, hausdorff_dis,hasu_xi,hasu_cu,acc_xi,acc_cu,auc,auc_xi,auc_cu= [], [], [], [], [], [], [], [], [],[],[],[],[],[],[],[]
        pr_g, pr_l = [], []
        mcc = []
        F, C, A, L, rSe, rSp, rAcc = [], [], [], [], [], [], []
        # net = UU_Net().to(device)
        # net = Who_Net().to(device)
        # net = img_patch().to(device)
        # net = U1_Net().to(device)
        # net = FCDenseNet().to(device)
        # net = Ue_Net().to(device)
        # net = CE_Net_().to(device)
        # net = JTFN().to(device)
        # net = Lits().to(device)
        # net.load_state_dict(torch.load(path))
        net = load_model(path)
        # skel_test = load_from_npy(Constants.path_skel_test)
        skel_test = load_from_npy(Constants.path_skel)
        # image_test = np.empty(shape=(20,1,584,565))
        image_test = np.empty(shape=(720, 1, 512, 512))
        # skel_images = np.empty(shape=(20,4,584,565))#20,2,584,565#720,2,512, 512
        # skel_images = np.empty(shape=(images.shape[0], 3, 512, 512))  # 20,2,584,565#720,2,512, 512
        net.eval()
        with torch.no_grad():
            for iter_ in range(0,images.shape[0]):
                x_img = images[iter_]
                x_img = np.expand_dims(x_img, axis=0)
                x_img = torch.tensor(x_img, dtype=torch.float32).to(device)
                # generated_vessel,_,_ = crop_eval_3(net, x_img)#test
                generated_vessel = crop_eval_3to4(net, x_img)#train
                # generated_vessel, patch_out, patch_128 = crop_eval_3(net, x_img)
                # generated_vessel = crop_eval(net, x_img)
                # generated_vessel1 = generated_vessel1.permute((0, 2, 3, 1)).detach().cpu().numpy()
                generated_vessel = generated_vessel.detach().cpu().numpy()
                image_test[iter_,:,:,:]=generated_vessel
                print('finish'+str(iter_))
        # for i in range(0,images.shape[0]):#720
        #     pic = Image.fromarray((255*skel_images[i,0]).astype(np.uint8))
        #     pic.save('/root/daima/YuzuSoft/log/visual_results/skel1/' + str(i) +'.png')
        #     pic = Image.fromarray((255*skel_images[i, 1]).astype(np.uint8))
        #     pic.save('/root/daima/YuzuSoft/log/visual_results/skel2/' + str(i) + '.png')
        # read_numpy_into_npy(skel_images,Constants.path_skel)
        skel_images = np.concatenate((skel_test,image_test),axis = 1)
        read_numpy_into_npy(skel_images,Constants.path_skel)#当前使用的3通道的数，各有不同含义，保存在_3.npy


if __name__ == '__main__':
    # path = '/root/daima/YuzuSoft/log/weights_save/waitfortest/'
    # path = '/root/daima/YuzuSoft/log/weights_save/DRIVE01/'
    path = '/root/daima/YuzuSoft/log/weights_save/DRIVE/'
    test_vessel_2(path)
    # test_vessel_train(path)
    # test_vessel_train_2(path)
    pass