import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd
import os
import random
from torchvision.datasets import DatasetFolder, ImageFolder

class Affectdataset(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, basic_aug=False, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df_train = pd.read_csv(os.path.join(self.root, 'train_set/train_annotations_7class.txt'), sep=' ',
                               header=None)
        df_valid = pd.read_csv(os.path.join(self.root, 'valid_set/valid_annotations_7class.txt'), sep=' ',
                               header=None)
        if self.train:
            dataset = df_train
        else:
            dataset = df_valid

        if self.dataidxs is not None:
            file_names = np.array(dataset.iloc[:, NAME_COLUMN].values)[self.dataidxs]
            self.target = np.array(dataset.iloc[:,LABEL_COLUMN].values)[self.dataidxs]

        else:
            file_names = dataset.iloc[:, NAME_COLUMN].values
            self.target = dataset.iloc[:,LABEL_COLUMN].values
            # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths = []

        for f in file_names:
            # f = f.split(".")[0]
            # f = f + "_aligned.jpg"
            if  self.train:
                path = os.path.join(self.root,'train_set/images',  f)
            else:
                path = os.path.join(self.root, 'valid_set/images', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]
        #######################################################################################

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.target

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        target = self.target[idx]
        if self.train:
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

# import numpy as np
# import glob
# from os.path import *
#
# ######################Train set  7class: 283901   8class :287651
# path = ("/home/cezheng/HPE/emotion/dataset/AffectNet/train_set/annotations")
# files =  sorted(glob.glob(path + '/*_exp.npy'))
# id_file = []
# label = []
# for i in range(len(files)):
#     if np.load(files[i]).astype(int)<7:
#         id_file.append(files[i][66:-8])
#         label.append(np.array(np.load(files[i])).tolist())
# print(len(files))
# print("af", len(label))
#
# with open('train_annotations.txt', 'w+') as f:
#     for i in range (len(id_file)):
#         # f.write("%s\n" % item)
#         f.write("%s.jpg %s\n" % (id_file[i], label[i]))
#     f.close()
#
#
# ############################ Test set    3500(7class) 3999(8class)
# path = ("/home/cezheng/HPE/emotion/dataset/AffectNet/valid_set/annotations")
# files =  sorted(glob.glob(path + '/*_exp.npy'))
# id_file = []
# label = []
# for i in range(len(files)):
#     if np.load(files[i]).astype(int)<8:
#         id_file.append(files[i][66:-8])
#         label.append(np.array(np.load(files[i])).tolist())
# print(len(files))
# print("af", len(label))
#
#
# with open('valid_annotations.txt', 'w+') as f:
#     for i in range (len(id_file)):
#         # f.write("%s\n" % item)
#         f.write("%s.jpg %s\n" % (id_file[i], label[i]))
#     f.close()