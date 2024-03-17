import torch
import numpy as np
from PIL import Image,ImageEnhance
#from PIL import ImageCms
import cv2
from skimage import color

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
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img

def rgb2gray(rgb: object) -> object:
    assert (len(rgb.shape)==4)   #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))

    return bn_imgs

def rgb2g(rgb):
    assert (len(rgb.shape) == 4)  # 4D arrays
    assert (rgb.shape[1] == 3)
    r_imgs = rgb[:, 1, :, :]
    r_imgs = np.reshape(r_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return r_imgs

def PreProc1(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.   #reduce to 0-1 range
    return train_imgs

#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

# CLAHE (Contrast Limited Adaptive Histogram Equalization)    #直方图修正
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)       #标准差
    imgs_mean = np.mean(imgs)     #平均值
    imgs_normalized = (imgs-imgs_mean)/imgs_std   #标准差
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255   #归一化，会被收敛到【0,1】之间
    return imgs_normalized


def gaussianNoise(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    for i in range(imgs.shape[0]):
        image = imgs[i,0]
        im = np.zeros(image.shape, np.uint8)
        if (len(image.shape) == 2):
            m = 0               #
            s = 10              #
        else:
            m = (0, 0, 0)
            s = (10, 10, 10)
        cv2.randn(im, m, s)
        image_noise = cv2.add(src1 = image, src2=im,dtype=cv2.CV_8UC3)
        imgs[i,0]=image_noise
    return imgs


def blurr(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            img = imgs[i,j]
            blur = cv2.GaussianBlur(img, (3, 3), 0)
            imgs[i, j] = np.asarray(blur)
    return imgs

def sharpen(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            img = imgs[i,j]
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

            sharpen = cv2.filter2D(img, -1, kernel)
            imgs[i, j] = np.asarray(sharpen)
    return imgs

import random
def drop(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    for i in range(imgs.shape[0]):
        img = imgs[i, 0]
        elems = [(x, y) for x in range(0, img.shape[0]) for y in range(0, img.shape[1])]
        random.shuffle(elems)
        dropoutelems = elems[0:int(0.01 * len(elems))]
        imageC = img.copy()
        for dropoutelem in dropoutelems:
            imageC[dropoutelem[0], dropoutelem[1]] = 0
        imgs[i, 0] = np.asarray(imageC)
    return imgs

def adjust_contrast(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    for i in range(imgs.shape[0]):
        img = Image.fromarray(imgs[i,0].astype(np.uint8))
        contrastEnhancer = ImageEnhance.Contrast(img)
        img = contrastEnhancer.enhance(factor=1.1)
        SharpnessEnhancer = ImageEnhance.Sharpness(img)
        #img = SharpnessEnhancer.enhance(4)
        imgs[i,0] = np.asarray(img)
    return imgs

def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

# ==================================================================================

if __name__ == '__main__':
    # cat_photo()
    # print(torch.nn.ConvTranspose2d(12,32, kernel_size=3, stride=2, padding=1, output_padding=1)(torch.rand(size=(2,12,128,128))).size())
    # pro_re()
    # pro_re2()
    # print('<->')



    pass