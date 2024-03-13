

#========================== public configure ==========================
# IMG_SIZE = (584, 565)
TOTAL_EPOCH = 50
INITAL_EPOCH_LOSS = 1000000
NUM_EARLY_STOP = 60
NUM_UPDATE_LR = 100
BINARY_CLASS = 1
BATCH_SIZE = 2
# learning_rates= 0.0003
learning_rates= 0.001
IMG_SIZE = (605, 700)

patch_size = 4

# ===================   DRIVE configure =========================
#DATA_SET = 'DRIVE_second'
DATA_SET = 'DRIVE'
visual_samples = '/root/daima/YuzuSoft/log/visual_samples/'
saved_path = '/root/daima/YuzuSoft/log/weights_save/'+ DATA_SET + '/'
visual_results = '/root/daima/YuzuSoft/log/visual_results/'+ DATA_SET + '/'




# size_h, size_w = 605, 700

# drive
# resize_drive = 512
# resize_size_drive = (resize_drive, resize_drive)
# size_h, size_w = 584, 565

# # chasedb
# resize_drive = 960
# resize_size_drive = (resize_drive, resize_drive)
# size_h, size_w = 960, 999

#HRF
# resize_drive = 960
# resize_size_drive = (resize_drive, resize_drive)
# size_h, size_w = 2336, 3504

# total_drive = 20
# path_image_drive = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/train_image_save.npy'
# path_label_drive = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/train_label_save.npy'
# path_label1_drive = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/train_label1_save.npy'
# path_test_image_drive = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/test_image_save.npy'
# path_test_label_drive = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/test_label_save.npy'
# path_test_label1_drive = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/test_label1_save.npy'
# path_test_image_drive_skel = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/skel_image_save.npy'
# path_test_label_drive_skel = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/skel_label_save.npy'
#
# path_skel_test = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/skel_test_3.npy'
# path_skel = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/skel_3.npy'
# # path_graph_edge_train = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/graph_train/graph_edge_save_train'
# # path_graph_edge_test = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/graph_test/graph_edge_save_test'
# # path_graph_point_train = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/point_type/train.pt'
# # path_graph_point_test = '/root/daima/YuzuSoft/dataset1/npy/DRIVE/tempt/point_type/test.pt'
# size_h, size_w = 584,565
# resize_drive = 512   #592


# total_drive = 8
# path_image_drive = '/root/daima/YuzuSoft/dataset1/npy/CHASE_DB/tempt/train_image_save.npy'
# path_label_drive = '/root/daima/YuzuSoft/dataset1/npy/CHASE_DB/tempt/train_label_save.npy'
# path_label1_drive = '/root/daima/YuzuSoft/dataset1/npy/CHASE_DB/tempt/train_label1_save.npy'
# path_test_image_drive = '/root/daima/YuzuSoft/dataset1/npy/CHASE_DB/tempt/test_image_save.npy'
# path_test_label_drive = '/root/daima/YuzuSoft/dataset1/npy/CHASE_DB/tempt/test_label_save.npy'
# path_skel_test = '/root/daima/YuzuSoft/dataset1/npy/CHASE_DB/tempt/skel_test_3.npy'
# path_skel = '/root/daima/YuzuSoft/dataset1/npy/CHASE_DB/tempt/skel_3.npy'
# size_h, size_w = 960, 999
# resize_drive = 960#ours

# stare
total_drive = 10
path_image_drive = '/root/daima/YuzuSoft/dataset1/npy/STARE20/tempt/train_image_save.npy'
path_label_drive = '/root/daima/YuzuSoft/dataset1/npy/STARE20/tempt/train_label_save.npy'
path_label1_drive = '/root/daima/YuzuSoft/dataset1/npy/STARE20/tempt/train_label_save1.npy'
path_test_image_drive = '/root/daima/YuzuSoft/dataset1/npy/STARE20/tempt/test_image_save.npy'
path_test_label_drive = '/root/daima/YuzuSoft/dataset1/npy/STARE20/tempt/test_label_save.npy'
path_skel_test = '/root/daima/YuzuSoft/dataset1/npy/STARE20/tempt/skel_test_3.npy'
path_skel = '/root/daima/YuzuSoft/dataset1/npy/STARE20/tempt/skel_3.npy'
size_h, size_w = 605, 700
resize_drive = 512#ours


# total_drive = 34 #34
# path_image_drive = '/root/daima/YuzuSoft/dataset1/npy/DCA1/tempt/train_image_save.npy'
# path_label_drive = '/root/daima/YuzuSoft/dataset1/npy/DCA1/tempt/train_label_save.npy'
# path_label1_drive = '/root/daima/YuzuSoft/dataset1/npy/DCA1/tempt/train_label_save1.npy'
# path_test_image_drive = '/root/daima/YuzuSoft/dataset1/npy/DCA1/tempt/test_image_save.npy'
# path_test_label_drive = '/root/daima/YuzuSoft/dataset1/npy/DCA1/tempt/test_label_save.npy'
# path_skel_test = '/root/daima/YuzuSoft/dataset1/npy/DCA1/tempt/skel_test_3.npy'
# path_skel = '/root/daima/YuzuSoft/dataset1/npy/DCA1/tempt/skel_3.npy'
# size_h, size_w = 300,300
# resize_drive = 288   #592 288

# path_image_drive = './dataset1/npy/HRF/tempt/train_image_save.npy'
# path_label_drive = './dataset1/npy/HRF/tempt/train_label_save.npy'
# path_test_image_drive = './dataset1/npy/HRF/tempt/test_image_save.npy'
# path_test_label_drive = './dataset1/npy/HRF/tempt/test_label_save.npy'
# path_valid_image_drive = './dataset1/npy/HRF/tempt/valid_image_save.npy'
# path_valid_label_drive = './dataset1/npy/HRF/tempt/valid_label_save.npy'


resize_size_drive = (resize_drive, resize_drive)
Classes_drive_color = 5
###########################################################################################