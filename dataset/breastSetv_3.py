import logging
import math
import os

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader

from dataset.randaugment import RandAugmentMC
import cv2

breast_labeled_path = '/media/cpf/BUSI3/train/'
breast_Unlabeled_path = '/media/cpf/BUSI3/train/'
breast_test = '/media/cpf/BUSI3/val'
logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes #每类有多少张图片
    _,labels = get_images_and_labels()# 把label变成数字
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0] #有[0]是因为np.where得到的是一个tuple,需要把tuple的元素提取出来
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

def get_breast():
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize([32,32]),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split(
    #     args, base_dataset.targets)
    labeled_path = breast_labeled_path
    train_labeled_dataset = breastSet(labeled_path, transform=transform_labeled)

    unlabeled_path = breast_Unlabeled_path
    train_unlabeled_dataset = breastSet(unlabeled_path,
                                        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_path = breast_test
    test_dataset = breastSet(test_path, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_images_and_labels(dir_path='/media/cpf/BUSI2'):
    images_list = []  # 文件名列表
    labels_list = []  # 标签列表

    for i in os.listdir(dir_path):
        images_list.append(i)
    print("images_list_len:", len(images_list))
    for i in images_list:
        if 'malignant' in i:
            labels_list.append(1)
        elif 'benign' in i:
            labels_list.append(0)
        elif 'normal' in i:
            labels_list.append(2)
    print("labels_list_len:", len(labels_list))

    return images_list, labels_list


class breastSet(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path  # 数据集根目录
        print("dir_path:", self.dir_path)
        self.transform = transform
        self.images, self.labels = get_images_and_labels(self.dir_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.dir_path, img_name)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片，np.fromfile解决路径中含有中文的问题

        # img = torch.from_numpy(img)  # Numpy需要转成torch之后才可以使用transform
        # img = img.permute(2, 0, 1)
        img = Image.fromarray(img)  # 实现array到image的转换，Image可以直接用transform
        # img = self.transform(img)  # 重点！！！如果为无标签的一致性正则化，那么此处会返回两个图   img即为一个list
        label = self.labels[index]
        # label=self.target_transform(label)
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        img = self.transform(img)




        return img, label
        # return img, label


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect')])  # 弱增强

        self.strong = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])  # 强增强，比弱增强多了两种图像失真处理

        self.resize=transforms.Compose([
            transforms.Resize([32, 32])])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        # print()
        org = self.resize(x)
        # 将弱增强后的图  强增强的图  分别进行标准化
        return self.normalize(weak), self.normalize(strong),self.normalize(org)# 返回一对弱增强、强增强

#
# if __name__ == '__main__':
#     labeled_dataset, unlabeled_dataset, test_dataset = get_breast()
#     labeled_trainloader = DataLoader(
#         labeled_dataset,
#         batch_size=4,
#         drop_last=True)
#     unlabeled_trainloader = DataLoader(
#         unlabeled_dataset,
#         batch_size=4,
#         drop_last=True)
#     test_trainloader = DataLoader(
#         test_dataset,
#         batch_size=1,
#         drop_last=True)
#     # torch.Size([4, 3, 32, 32]) tensor([0, 1, 0, 0])
#     for i, (img, label) in enumerate(labeled_trainloader):
#         print("load labeled_datatset!")
#         print(img.shape, label)
#     #     torch.Size([4, 3, 512, 512])
#     #     torch.Size([4, 3, 512, 512])
#     #     torch.Size([4, 3, 512, 512])
#     #     tensor([0, 1, 2, 0])
#     for (img1, img2,img3), y in unlabeled_trainloader:
#         print("load unlabeled_datatset!")
#         print(img1.shape, img2.shape, img3.shape, y)
#     for (img, label) in test_trainloader:
#         print("load test_datatset!")
#         print(img.shape, label)

DATASET_GETTERS = {'busi': get_breast()}
