import random
import cv2, time
import torch
import skimage.color as sc
import utils
from data import common
import imageio
import numpy as np
import torch.utils.data as data
import os
from tqdm import tqdm
# from src.bicubic_numpy import imresize as bicubic


class dataloader_rescaling(data.Dataset):
    def __init__(self, args):
        self.args = args
        self._set_filesystem()

        if self.args.store_in_ram:
            self.img_HQ, self.img_LQ = [], []
            with tqdm(total=len(self.filepath_HQ), ncols=224) as pbar:
                for idx in range(len(self.filepath_HQ)):
                    img_HQ = imageio.imread(self.filepath_HQ[idx])
                    img_LQ = imageio.imread(self.filepath_LQ[idx])
                    # img_LQ = bicubic(img_HQ, scalar_scale=1 / self.args.scale, method='bicubic')
                    self.img_HQ.append(img_HQ)
                    self.img_LQ.append(img_LQ)
                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(name=self.filepath_HQ[idx].split('/')[-1])
            self.n_train = len(self.img_HQ)

    def _set_filesystem(self):
        self.filepath_HQ = np.array([])
        self.filepath_LQ = np.array([])
        for idx_dataset in range(len(self.args.data_train)):
            if self.args.n_train[idx_dataset] > 0:
                path = self.args.dir_data + 'Train/' + self.args.data_train[idx_dataset]
                names_HQ = os.listdir(os.path.join(path, 'HR'))
                task_note = 'LR_bicubic/X{}'.format(self.args.scale)
                names_LQ = os.listdir(os.path.join(path, task_note))

                names_HQ.sort()
                names_LQ.sort()
                data_length = len(names_HQ)
                idx = np.arange(0, data_length)
                if self.args.n_train[idx_dataset] < data_length:
                    if self.args.shuffle:
                        idx = np.random.choice(idx, size=self.args.n_train[idx_dataset])
                    else:
                        idx = np.arange(0, self.args.n_train[idx_dataset])

                filepath_HQ = np.array([])
                filepath_LQ = np.array([])

                for idx_ in idx:
                    filepath_HQ = np.append(filepath_HQ, os.path.join(path + '/HR', names_HQ[idx_]))
                    filepath_LQ = np.append(filepath_LQ, os.path.join(path + '/' + task_note, names_LQ[idx_]))

                self.filepath_HQ = np.append(self.filepath_HQ, filepath_HQ)
                self.filepath_LQ = np.append(self.filepath_LQ, filepath_LQ)

    def __getitem__(self, idx):
        if self.args.store_in_ram:
            idx = idx % len(self.img_HQ)
            img_HQ = self.img_HQ[idx]
            img_LQ = self.img_LQ[idx]
        else:
            raise InterruptedError

        img_LQ, img_HQ = common.set_channel([img_LQ, img_HQ], self.args.n_colors)
        img_LQ, img_HQ = common.get_patch([img_LQ, img_HQ], self.args.patch_size, self.args.scale)

        flag_aug = random.randint(0, 7)
        img_LQ = common.augment(img_LQ, flag_aug)
        img_HQ = common.augment(img_HQ, flag_aug)
        img_LQ = common.np2Tensor(img_LQ, self.args.value_range)
        img_HQ = common.np2Tensor(img_HQ, self.args.value_range)

        return img_LQ, img_HQ

    def __len__(self):
        return self.args.iter_epoch * self.args.batch_size


class dataloader_decolor(data.Dataset):
    def __init__(self, args):
        self.args = args
        self._set_filesystem()

        if self.args.store_in_ram:
            self.img_HQ, self.img_LQ = [], []
            with tqdm(total=len(self.filepath_HQ), ncols=224) as pbar:
                for idx in range(len(self.filepath_HQ)):
                    img_HQ = imageio.imread(self.filepath_HQ[idx])
                    img_LQ = sc.rgb2lab(img_HQ) / 100.
                    img_LQ = (255. * img_LQ[:, :, 0]).astype(np.uint8)

                    self.img_HQ.append(img_HQ)
                    self.img_LQ.append(img_LQ)
                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(name=self.filepath_HQ[idx].split('/')[-1])
            self.n_train = len(self.img_HQ)

    def _set_filesystem(self):
        self.filepath_HQ = np.array([])
        for idx_dataset in range(len(self.args.data_train)):
            if self.args.n_train[idx_dataset] > 0:
                path = self.args.dir_data + 'Train/' + self.args.data_train[idx_dataset]
                names_HQ = os.listdir(os.path.join(path, 'HR'))
                names_HQ.sort()
                filepath_HQ = np.array([])

                for idx_image in range(len(names_HQ)):
                    filepath_HQ = np.append(filepath_HQ, os.path.join(path + '/HR', names_HQ[idx_image]))

                data_length = len(filepath_HQ)
                idx = np.arange(0, data_length)
                if self.args.n_train[idx_dataset] < data_length:
                    if self.args.shuffle:
                        idx = np.random.choice(idx, size=self.args.n_train[idx_dataset])
                    else:
                        idx = np.arange(0, self.args.n_train[idx_dataset])

                self.filepath_HQ = np.append(self.filepath_HQ, filepath_HQ[idx])

    def __getitem__(self, idx):

        if self.args.store_in_ram:
            idx = idx % len(self.img_HQ)
            img_HQ = self.img_HQ[idx]
            img_LQ = self.img_LQ[idx]
        else:
            raise InterruptedError

        img_LQ = common.set_channel(img_LQ, self.args.out_channels)
        img_HQ = common.set_channel(img_HQ, self.args.in_channels)
        img_LQ, img_HQ = common.get_patch([img_LQ, img_HQ], self.args.patch_size, self.args.scale)
        flag_aug = random.randint(0, 7)
        img_LQ = common.augment(img_LQ, flag_aug)
        img_HQ = common.augment(img_HQ, flag_aug)
        img_LQ = common.np2Tensor(img_LQ, self.args.value_range)
        img_HQ = common.np2Tensor(img_HQ, self.args.value_range)

        return img_LQ, img_HQ

    def __len__(self):
        return self.args.iter_epoch * self.args.batch_size


class dataloader_hiding(data.Dataset):
    def __init__(self, args):
        self.args = args
        self._set_filesystem()

        if self.args.store_in_ram:
            self.img_cover = []
            with tqdm(total=len(self.filepath_HQ), ncols=224) as pbar:
                for idx in range(len(self.filepath_HQ)):
                    img_cover = imageio.imread(self.filepath_HQ[idx])
                    self.img_cover.append(img_cover)
                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(name=self.filepath_HQ[idx].split('/')[-1])
            self.n_train = len(self.img_cover)

    def _set_filesystem(self):
        self.filepath_HQ = np.array([])
        for idx_dataset in range(len(self.args.data_train)):
            if self.args.n_train[idx_dataset] > 0:
                path = self.args.dir_data + 'Train/' + self.args.data_train[idx_dataset]
                names_HQ = os.listdir(os.path.join(path, 'HR'))
                names_HQ.sort()
                filepath_HQ = np.array([])

                for idx_image in range(len(names_HQ)):
                    filepath_HQ = np.append(filepath_HQ, os.path.join(path + '/HR', names_HQ[idx_image]))

                data_length = len(filepath_HQ)
                idx = np.arange(0, data_length)
                if self.args.n_train[idx_dataset] < data_length:
                    if self.args.shuffle:
                        idx = np.random.choice(idx, size=self.args.n_train[idx_dataset])
                    else:
                        idx = np.arange(0, self.args.n_train[idx_dataset])

                self.filepath_HQ = np.append(self.filepath_HQ, filepath_HQ[idx])

    def __getitem__(self, idx):

        if self.args.store_in_ram:
            idx = idx % len(self.img_cover)
            img_cover = self.img_cover[idx]
            img_secrets = []
            for i in range(self.args.num_secrets):
                img_secrets.append(self.img_cover[(idx + np.random.randint(0, len(self.img_cover))) % len(self.img_cover)])
        else:
            raise InterruptedError

        img_secrets = common.set_channel(img_secrets, self.args.in_channels)
        img_cover = common.set_channel(img_cover, self.args.in_channels)
        img_cover = common.get_patch(img_cover, self.args.patch_size, 1)
        flag_aug = random.randint(0, 7)
        img_cover = common.augment(img_cover, flag_aug)
        img_cover = common.np2Tensor(img_cover, self.args.value_range)

        for i in range(self.args.num_secrets):
            img_secrets[i] = common.get_patch(img_secrets[i], self.args.patch_size, 1)
            flag_aug = random.randint(0, 7)
            img_secrets[i] = common.augment(img_secrets[i], flag_aug)
            img_secrets[i] = common.np2Tensor(img_secrets[i], self.args.value_range)
        img_secrets = torch.cat(img_secrets, dim=0)
        return img_cover, torch.cat((img_cover, img_secrets), dim=0)

    def __len__(self):
        return self.args.iter_epoch * self.args.batch_size


