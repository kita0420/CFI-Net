import sys
sys.path.append('../')
sys.path.append('/root/root/Project_Project/YuzuSoft/')
import cv2

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image, ImageEnhance

from data_process.retinal_process import rgb2g ,rgb2gray, clahe_equalized, dataset_normalized, adjust_gamma,adjust_contrast,gaussianNoise,blurr,drop,sharpen
from data_process.data_ultils import group_images, visualize, label2rgb
import Constants
#from test.ACE import ace_process
import warnings
warnings.filterwarnings("ignore")

save_drive = '_drive'
save_drive_color = '_drive_color'
save_mo = '_mo'
save_pylop = '_pylop'
save_tbnc = '_tbnc'


def visual_sample(images, mask, path, per_row =5):
    image = images[:, 0, :, :]
    images = np.reshape(image, (images.shape[0], 1, images.shape[2], images.shape[3]))
    visualize(group_images(images, per_row), Constants.visual_samples + path + '0')
    visualize(group_images(mask, per_row), Constants.visual_samples + path + '1')


def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def get_drive_data(val_ratio =1, is_train = True):  #0.6
    images = load_from_npy(Constants.path_image_drive)
    mask = load_from_npy(Constants.path_label_drive)
    mask1 = load_from_npy(Constants.path_label1_drive)
    # skel = load_from_npy(Constants.path_label_drive)
    skel = load_from_npy(Constants.path_skel)####
    # print(mask.shape)
    # weight = load_from_npy(Constants.weight_drive)
    images_test = load_from_npy(Constants.path_test_image_drive)
    mask_test = load_from_npy(Constants.path_test_label_drive)
    # mask1_test = load_from_npy(Constants.path_test_label1_drive)
    skel_test = load_from_npy(Constants.path_skel_test)###
    # skel_test = load_from_npy(Constants.path_test_label_drive_skel)
    # images_valid = load_from_npy(Constants.path_valid_image_drive)
    # mask_valid = load_from_npy(Constants.path_valid_label_drive)
    # for i in range(20):
    #     image1 = np.transpose(images, (0,2, 3, 1))
    #     img = Image.fromarray(image1[i*36].astype(np.uint8))
    #     img.save( './image_show/train/'+str(i)+'.png')

    images = rgb2gray(images)   #3维边1维
    # images = gaussianNoise(images)

    images = dataset_normalized(images)###
    images = clahe_equalized(images)
    images = adjust_gamma(images, 1.5)
    images = adjust_contrast(images)

    # images = blurr(images)
    # images = sharpen(images)
    # if is_train == True:
    #     images = drop(images)
    # for i in range(20):
    #     image1 = np.transpose(images, (0,2, 3, 1))
    #     img = Image.fromarray(image1[i*36].astype(np.uint8))
    #     img.save( './image_show/train/'+str(i)+'.png')
    #     img = Image.fromarray(255*mask[i * 36,0].astype(np.uint8))
    #     img.save('/root/daima/YuzuSoft/data_process/image_show/train/' + str(i) + 'mask.png')



    # for i in range(20): #输出对比
    #     # img = Image.fromarray((mask1[i * 36, 0] * 255).astype(np.uint8))
    #     img = Image.fromarray((skel[i*36 , 0] * 255).astype(np.uint8))
    #     img = img
    #     img.save('/root/daima/YuzuSoft/data_process/image_show/test/' + str(i) + '.png')
    #     img = Image.fromarray((skel[i * 36, 1] * 255).astype(np.uint8))
    #     img = img
    #     img.save('/root/daima/YuzuSoft/data_process/image_show/test1/' + str(i) + '.png')

    # images = images[36:,:,:,:]
    # mask = mask[36:,:,:,:]
    #images = adjust_contrast(images)
    # images = torch.from_numpy(images)
    # images = images.numpy()



    # images_test = blurr(images_test)
    # images_test = sharpen(images_test)
    images_test = rgb2gray(images_test)

    # images_test = dataset_normalized(images_test)
    # images_test = clahe_equalized(images_test)
    # images_test = adjust_gamma(images_test, 1.5)
    # images_test = adjust_contrast(images_test)
    print(skel_test.shape)
    # for i in range(20):
    #     img = Image.fromarray((255*skel_test[i,0]).astype(np.uint8))
    #     img.save( './image_show/test/'+str(i)+'.png')
    #     img = Image.fromarray((255*mask_test[i, 0]).astype(np.uint8))
    #     img.save('./image_show/test1/' + str(i) + '.png')
    # for i in range(2):
    #     img = Image.fromarray(images_test[i,0].astype(np.uint8))
    #     img.save( '/root/daima/YuzuSoft/data_process/image_show/test/'+str(i)+'.png')
    # images_valid = rgb2gray(images_valid)
    # images_valid = dataset_normalized(images_valid)
    # images_valid = clahe_equalized(images_valid)
    # images_valid = adjust_gamma(images_valid, 1.0)

    images = images / 255.  # reduce to 0-1 range

    # images_test = images_test / 255.
    # images_valid = images_valid / 255.


    # images_val = images_val / 255.
    #print(images.shape, mask.shape, '=================', np.max(images), np.max(mask))
    print(images.shape, mask.shape, '=================')
    print('========  success load all Drive files ==========')
    # visual_sample(images[0:15,:,:,:,], mask[0:15,:,:,:,], save_drive)
    val_num = int(images_test.shape[0] * val_ratio)
    # train_list = [images[val_num:, :, :, :, ], mask[val_num:, :, :, :, ]]
    # train_list = [images[0:, :, :, :, ], mask[0:, :, :, :, ], mask1[0:, :, :, :, ]]
    train_list = [images[0:, :, :, :, ], mask[0:, :, :, :, ],skel[0:, :, :, :,]]
    # train_list = [images, mask]
    # val_list = [images_val, mask_val]
    # val_list = [images_test, mask_test]
    # val_list = [images_test[0:val_num, :, :, :, ], mask_test[0:val_num, :, :, :, ]]
    val_list = [images_test[0:val_num, :, :, :, ], mask_test[0:val_num, :, :, :, ],skel_test[0:val_num, :, :, :, ]]

    if is_train is True:
        return train_list, val_list
    else:
        # return images_test, mask_test
        return images_test, mask_test,skel_test#暂时替换

def get_drive_data_skel(val_ratio =1, is_train = True):  #0.6
    images = load_from_npy(Constants.path_image_drive)
    mask = load_from_npy(Constants.path_label_drive)
    # mask1 = load_from_npy(Constants.path_label1_drive)
    skel = load_from_npy(Constants.path_label_drive)
    # skel = load_from_npy(Constants.path_skel)####
    # print(mask.shape)
    # weight = load_from_npy(Constants.weight_drive)
    images_test = load_from_npy(Constants.path_test_image_drive)
    mask_test = load_from_npy(Constants.path_test_label_drive)
    skel_test = load_from_npy(Constants.path_label_drive)
    # skel_test = load_from_npy(Constants.path_test_label_drive_skel)
    # images_valid = load_from_npy(Constants.path_valid_image_drive)
    # mask_valid = load_from_npy(Constants.path_valid_label_drive)
    # for i in range(20):
    #     image1 = np.transpose(images, (0,2, 3, 1))
    #     img = Image.fromarray(image1[i*36].astype(np.uint8))
    #     img.save( './image_show/train/'+str(i)+'.png')

    images = rgb2gray(images)   #3维边1维
    # images = gaussianNoise(images)

    # images = dataset_normalized(images)###
    # images = clahe_equalized(images)
    # images = adjust_gamma(images, 1.5)
    # images = adjust_contrast(images)

    # images = blurr(images)
    # images = sharpen(images)
    # if is_train == True:
    #     images = drop(images)
    # for i in range(20):
    #     image1 = np.transpose(images, (0,2, 3, 1))
    #     img = Image.fromarray(image1[i*36].astype(np.uint8))
    #     img.save( './image_show/train/'+str(i)+'.png')
    #     img = Image.fromarray(255*mask[i * 36,0].astype(np.uint8))
    #     img.save('/root/daima/YuzuSoft/data_process/image_show/train/' + str(i) + 'mask.png')



    # for i in range(20): #输出对比
    #     # img = Image.fromarray((mask1[i * 36, 0] * 255).astype(np.uint8))
    #     img = Image.fromarray((skel[i*36 , 0] * 255).astype(np.uint8))
    #     img = img
    #     img.save('/root/daima/YuzuSoft/data_process/image_show/test/' + str(i) + '.png')
    #     img = Image.fromarray((skel[i * 36, 1] * 255).astype(np.uint8))
    #     img = img
    #     img.save('/root/daima/YuzuSoft/data_process/image_show/test1/' + str(i) + '.png')

    # images = images[36:,:,:,:]
    # mask = mask[36:,:,:,:]
    #images = adjust_contrast(images)
    # images = torch.from_numpy(images)
    # images = images.numpy()



    # images_test = blurr(images_test)
    # images_test = sharpen(images_test)
    images_test = rgb2gray(images_test)

    # images_test = dataset_normalized(images_test)
    # images_test = clahe_equalized(images_test)
    # images_test = adjust_gamma(images_test, 1.5)
    # images_test = adjust_contrast(images_test)
    print(skel_test.shape)
    # for i in range(20):
    #     img = Image.fromarray((255*skel_test[i,0]).astype(np.uint8))
    #     img.save( './image_show/test/'+str(i)+'.png')
    #     img = Image.fromarray((255*mask_test[i, 0]).astype(np.uint8))
    #     img.save('./image_show/test1/' + str(i) + '.png')
    # for i in range(2):
    #     img = Image.fromarray(images_test[i,0].astype(np.uint8))
    #     img.save( '/root/daima/YuzuSoft/data_process/image_show/test/'+str(i)+'.png')
    # images_valid = rgb2gray(images_valid)
    # images_valid = dataset_normalized(images_valid)
    # images_valid = clahe_equalized(images_valid)
    # images_valid = adjust_gamma(images_valid, 1.0)

    images = images / 255.  # reduce to 0-1 range

    # images_test = images_test / 255.
    # images_valid = images_valid / 255.


    # images_val = images_val / 255.
    #print(images.shape, mask.shape, '=================', np.max(images), np.max(mask))
    print(images.shape, mask.shape, '=================')
    print('========  success load all Drive files ==========')
    # visual_sample(images[0:15,:,:,:,], mask[0:15,:,:,:,], save_drive)
    val_num = int(images_test.shape[0] * val_ratio)
    # train_list = [images[val_num:, :, :, :, ], mask[val_num:, :, :, :, ]]
    train_list = [images[0:, :, :, :, ], mask[0:, :, :, :, ],skel[0:, :, :, :,]]
    # train_list = [images, mask]
    # val_list = [images_val, mask_val]
    # val_list = [images_test, mask_test]
    val_list = [images_test[0:val_num, :, :, :, ], mask_test[0:val_num, :, :, :, ],skel_test[0:val_num, :, :, :, ]]

    if is_train is True:
        return train_list, val_list
    else:
        return images, mask
        # return images_test, mask_test#暂时替换

class ImageFolder(data.Dataset):
    '''
    image is RGB original image, mask is one hot GT and label is grey image to visual
    img and mask is necessary while label is alternative
    '''
    def __init__(self,img, mask, label=None):
        self.img = img
        self.mask = mask
        self.label = label

    def __getitem__(self, index):
        imgs  = torch.from_numpy(self.img[index]).float()
        masks = torch.from_numpy(self.mask[index]).float()
        if self.label is not None:
            label = torch.from_numpy(self.label[index]).float()
            return imgs, masks, label
        else:
            return imgs, masks

    def __len__(self):
        assert self.img.shape[0] == self.mask.shape[0], 'The number of images must be equal to labels'
        return self.img.shape[0]


if __name__ == '__main__':

    get_drive_data()
    # get_monuclei_data()
    # get_MRI_chaos_data()
    # get_test_MRI_chaos_data()
    # get_tnbc_data(0.2, is_train = True)
    # get_pylyp_data()
    # get_drive_color_data()

    pass
