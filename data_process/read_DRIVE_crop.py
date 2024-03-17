import sys
import torch
sys.path.append('../')
sys.path.append('/root/daima/YuzuSoft/')

import sys
sys.path.append('../')                  # import Constant file


import  numpy as np
import os
import matplotlib.pyplot as plt
from  PIL import  Image
import cv2
import Constants
from data_process.data_ultils import  read_all_images, data_shuffle
# from YuzuSoft.data_process.retinal_process import rgb2gray, clahe_equalized, dataset_normalized, adjust_gamma
from skimage import feature
from skimage.segmentation import  find_boundaries
# path_images_drive = '../dataset1/DRIVE/training/images/'
# path_gt_drive = '../dataset1/DRIVE/training/1st_manual/'
# path_images_test_drive = '../dataset1/DRIVE/test/images/'
# path_gt_test_drive = '../dataset1/DRIVE/test/1st_manual/'

# path_images_drive = '../dataset1/CHASE_DB1/training/images/'
# path_gt_drive = '../dataset1/CHASE_DB1/training/1st_manual/'
# path_images_test_drive = '../dataset1/CHASE_DB1/test/images/'
# path_gt_test_drive = '../dataset1/CHASE_DB1/test/1st_manual/'

path_images_drive = '../dataset1/STARE1/training/images/'
path_gt_drive = '../dataset1/STARE1/training/1st_manual/'
path_images_test_drive = '../dataset1/STARE1/test/images/'
path_gt_test_drive = '../dataset1/STARE1/test/1st_manual/'

# path_images_drive = '../dataset1/DCA1_134/training/images/'
# path_gt_drive = '../dataset1/DCA1_134/training/1st_manual/'
# path_images_test_drive = '../dataset1/DCA1_134/test/images/'
# path_gt_test_drive = '../dataset1/DCA1_134/test/1st_manual/'





# path_images_drive = './dataset1/HRF/training/images/'
# path_gt_drive = './dataset1/HRF/training/1st_manual/'
# path_images_test_drive = './dataset1/HRF/test/images/'
# path_gt_test_drive = './dataset1/HRF/test/1st_manual/'
# path_images_valid_drive = './dataset1/HRF/valid/images/'
# path_gt_valid_drive = './dataset1/HRF/valid/1st_manual/'


# path_images_drive = '/root/daima/YuzuSoft/dataset1/DRIVE/training/images/'
# path_gt_drive = '/root/daima/YuzuSoft/dataset1/DRIVE/training/1st_manual/'
# path_images_test_drive = '/root/daima/YuzuSoft/dataset1/DRIVE/test/images/'
# path_gt_test_drive = '/root/daima/YuzuSoft/dataset1/DRIVE/test/1st_manual/'





def blurr(imgs):
    imgs = np.array(imgs, np.float32).transpose(0, 3, 1, 2)
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 3)  # check the channel is 1
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            img = imgs[i,j]
            blur = cv2.GaussianBlur(img, (3, 3), 0)
            imgs[i, j] = np.asarray(blur)
    imgs = np.array(imgs, np.float32).transpose(0, 2, 3, 1)
    return imgs

def sharpen(imgs):
    imgs = np.array(imgs, np.float32).transpose(0, 3, 1, 2)
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 3)  # check the channel is 1
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            img = imgs[i,j]
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

            sharpen = cv2.filter2D(img, -1, kernel)
            imgs[i, j] = np.asarray(sharpen)
    imgs = np.array(imgs, np.float32).transpose(0, 2, 3, 1)
    return imgs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_numpy_into_npy(arrays, path):
    np.save(path, arrays)
    print('have saved all arrays in to path ', path)

def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def read_drive_images(size_h,size_w, path_images, path_gt,total_imgs, mask_ch =1):
    all_images = np.empty(shape=(total_imgs, size_h, size_w, 3))
    all_masks  = np.empty(shape=(total_imgs, size_h, size_w, 1)) #DCA为3
    all_images = read_all_images(path_images, all_images,size_h, size_w,type ='non_resize')
    all_masks  = read_all_images(path_gt, all_masks, size_h, size_w,type ='non_resize')
    # all_masks = all_masks[:, :, :, 1, ]   # DCA1 dataset
    # all_masks = np.expand_dims(all_masks, axis=3) # DCA1 dataset
    print('============= have read all images ==============')
    return all_images, all_masks

def gaussian_noise(img, mean, sigma):
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out)
    return gaussian_out

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
    return image, mask

def randomHorizontalFlip(image, mask):
    if np.random.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return  image, mask

def randomVerticleFlip(image, mask):
    if np.random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return  image, mask

def crop_images(image, mask, crop_size = Constants.resize_drive):
    select_id = np.random.randint(0, 4)
    d_h, d_w, h, w =  image.shape[0] - crop_size, image.shape[1] - crop_size,image.shape[0],image.shape[1]
    crop_lu_im,  crop_lu_ma = image[d_h:h, d_w:w, :,], mask[d_h:h, d_w:w, :,]
    crop_ld_im,  crop_ld_ma = image[d_h:h, 0:w-d_w, :, ], mask[d_h:h, 0:w-d_w, :, ]
    crop_ru_im,  crop_ru_ma = image[0:h - d_h, d_w:w, :, ], mask[0:h - d_h, d_w:w, :, ]
    crop_rd_im,  crop_rd_ma = image[0:h - d_h, 0:w-d_w, :, ], mask[0:h - d_h, 0:w-d_w, :, ]
    # crop_img = np.concatenate([np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_ld_im, axis=0),
    #                 np.expand_dims(crop_ru_im, axis=0),np.expand_dims(crop_rd_im, axis=0)], axis = 0)
    # crop_mask = np.concatenate([np.expand_dims(crop_lu_ma, axis=0), np.expand_dims(crop_lu_ma, axis=0),
    #                 np.expand_dims(crop_lu_ma, axis=0),np.expand_dims(crop_lu_ma, axis=0)], axis = 0)
    crop_img, crop_mask =None, None
    if select_id ==0:
        crop_img, crop_mask = np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_lu_ma, axis=0)
    if select_id ==1:
        crop_img, crop_mask = np.expand_dims(crop_ld_im, axis=0), np.expand_dims(crop_ld_ma, axis=0)
    if select_id ==2:
        crop_img, crop_mask = np.expand_dims(crop_ru_im, axis=0), np.expand_dims(crop_ru_ma, axis=0)
    if select_id ==3:
        crop_img, crop_mask = np.expand_dims(crop_rd_im, axis=0), np.expand_dims(crop_rd_ma, axis=0)
    return crop_img, crop_mask



def deformation_set(image, mask,
                           shift_limit=(-0.2, 0.2),
                           scale_limit=(-0.2, 0.2),
                           rotate_limit=(-180.0, 180.0),
                           aspect_limit=(-0.1, 0.1),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    print('deformation_set size check: ', image.shape, mask.shape)

    start_angele, per_rotate = -180, 10
    rotate_num = - start_angele // per_rotate * 2
    image_set, mask_set = [], []
    for rotate_id in range(0, rotate_num):
        masks = mask
        img = image
        height, width, channel = img.shape
        sx, sy = 1., 1.
        angle = start_angele + rotate_id * per_rotate
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
        cc = np.cos(angle / 180 * np.pi) * sx
        ss = np.sin(angle / 180 * np.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])
        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height],])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))
        masks = cv2.warpPerspective(masks, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(0, 0, 0,))
        #以上是随机移位缩放旋转l
        img, masks = randomHorizontalFlip(img, masks)
        #以上是随即水平旋转 上下翻转
        img, masks = randomVerticleFlip(img, masks)
        #以上是随机垂直旋转  左右翻转
        masks = np.expand_dims(masks, axis=2)               #给masks升一维


        # image_set.append(np.expand_dims(img, axis=0))
        # mask_set.append(np.expand_dims(masks, axis=0))

        # crop_im, crop_ma = crop_images(img, masks)#DRIVE CHASEDB STARE
        crop_im, crop_ma = crop_images(img, masks)    #图像的裁剪 切成12张
        image_set.append(crop_im)
        mask_set.append(crop_ma)
        #print(img.shape, masks.shape,'====================')
    aug_img  = np.concatenate([image_set[i] for i in range(0, len(image_set))],axis=0)     #image_set[0].shape  [12, 512, 512, 3]
    aug_mask = np.concatenate([mask_set[i] for i in range(0, len(mask_set))], axis=0)     #变成512×512
    return aug_img, aug_mask

def data_auguments(aug_num,size_h, size_w,path_images, path_gt,total_imgs, mask_ch, augu=True):

    all_images, all_masks = read_drive_images(size_h, size_w,path_images, path_gt,total_imgs, mask_ch)         # original data   读入图片放入四维向量中

    if augu is False:
        return all_images, all_masks
    # print('image and gt shape is:', all_images.shape, all_masks.shape)
    img_list = []
    gt_list = []
    for nums in range(0, aug_num):
        for i_d in range(0, all_images.shape[0]):#图片数量
            aug_img, aug_gt = deformation_set(all_images[i_d, :, :, :, ], all_masks[i_d, :, :, :, ])#得到数据增强的一堆图片
            print(aug_img.shape,'---------',aug_gt.shape)
            img_list.append(aug_img)
            gt_list.append(aug_gt)
    img_au = np.concatenate(img_list, axis=0)
    gt_au = np.concatenate(gt_list, axis=0)
    # print(img_au.shape, gt_au.shape)
    # visualize(group_images(all_masks, 5), './image_test')
    return img_au,gt_au

def data_for_train(aug_num,size_h, size_w,path_images, path_gt,total_imgs, mask_ch,augu):
    all_images, all_masks = data_auguments(aug_num, size_h, size_w, path_images, path_gt,total_imgs, mask_ch,augu)
    print('image and gt shape is:', all_images.shape, all_masks.shape)
    # img = np.array(all_images, np.float32).transpose(0,3,1,2) / 255.0
    # mask = np.array(all_masks, np.float32).transpose(0,3,1,2) / 255.0

    img = np.array(all_images, np.float32).transpose(0,3,1,2)
    mask = np.array(all_masks, np.float32).transpose(0,3,1,2)

    if mask_ch ==1:
        mask[mask >= 0.5] = 1  #图像阈值
        mask[mask < 0.5] = 0
    index = np.arange(img.shape[0]) #960的排序
    # np.random.shuffle(index)  #打乱 测试时不需要
    # img  = img[index, :, :, :]
    # mask = mask[index, :,:,:]

    mask1 = mask.copy() * 255
    mask1 = mask1.astype( np.uint8 )
    for i in range(mask.shape[0]):
        mask1[i,0] = cv2.Canny(mask1[i,0],50,200)
    mask1 = mask1.astype('float32')
    mask1 = mask1/255
    return img, mask,mask1
    #
    # return img, mask

def save_drive_data(mum_arg = 1):
    images, mask ,mask1= data_for_train(mum_arg, Constants.size_h,Constants.size_w,
                                  path_images_drive, path_gt_drive, 20, mask_ch=1, augu=True)
    # images_test, mask_test = data_for_train(mum_arg, Constants.size_h,Constants.size_w,
    #                               path_images_test_drive, path_gt_test_drive, 20, mask_ch=1, augu=False)
    images_test, mask_test ,_= data_for_train(mum_arg, Constants.size_h,Constants.size_w,
                                            path_images_test_drive, path_gt_test_drive,10, mask_ch=1, augu=False)
    # images_valid, mask_valid = data_for_train(mum_arg, Constants.size_h, Constants.size_w,
    #                                         path_images_valid_drive, path_gt_valid_drive, 3, mask_ch=1, augu=False)

    # images_test,mask_test = data_shuffle(images_test,mask_test)
    # images_valid, mask_valid = data_shuffle(images_valid, mask_valid)

    try:
        read_numpy_into_npy(images, Constants.path_image_drive)
        read_numpy_into_npy(mask, Constants.path_label_drive)
        read_numpy_into_npy(mask1, Constants.path_label1_drive)
        read_numpy_into_npy(images_test, Constants.path_test_image_drive)
        read_numpy_into_npy(mask_test, Constants.path_test_label_drive)
        # read_numpy_into_npy(images_valid, Constants.path_valid_image_drive)
        # read_numpy_into_npy(mask_valid, Constants.path_valid_label_drive)

        print('========  all drive train and test data has been saved ! ==========')
    except:
        print(' file save exception has happened! ')

    pass


def check_bst_data():
    a=load_from_npy(Constants.path_image_drive)
    b = load_from_npy(Constants.path_label_drive)
    a = a.transpose(0,2,3,1)
    print(a.shape)
    # b = b.transpose(0,2,3,1)
    for i in range(0,a.shape[0]):
        # pil_image = Image.fromarray((255*a[i,:,:,:]).astype(np.uint8)).convert('RGB')
        # pil_image.save('./image_show/tempt/'+str(i)+'.png')
        pic = Image.fromarray((a[i,:,:,:]).astype(np.uint8))
        pic.save( './image_show/tempt/'+str(i)+'.png')
        # pic = Image.fromarray((255*b[i, 0, :, :]).astype(np.uint8))
        # pic.save('./image_show/tempt_label/' + str(i) + '.png')
    # e = load_from_npy(Constants.path_label1_drive)
    # c=load_from_npy(Constants.path_test_image_drive)
    # d = load_from_npy(Constants.path_valid_label_drive)
    # e = load_from_npy(Constants.path_valid_image_drive)
    # f = load_from_npy(Constants.path_test_label_drive)
    # print(a.shape, b.shape,e.shape , c.shape, f.shape)
    # print(np.max(a), np.max(b),np.max(e) , np.max(c), np.max(f))


if __name__ == '__main__':
    save_drive_data()
    check_bst_data()
    pass
