#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2020/10/29 14:52
# @File    : augmentation.py
# @Software: PyCharm


import numpy as np
import os
import glob
from skimage import io
from skimage import transform
from skimage import img_as_ubyte
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class AUGMENTATION:
    """
    数据预处理类，进行数据增强、数据可视化等
    """

    def __init__(self):
        sky = [128, 128, 128]  # 灰色
        building = [128, 0, 0]  # 紫红色
        pole = [192, 192, 128]  # 浅红色
        road = [128, 64, 128]  # 浅紫色
        pavement = [60, 40, 222]  # 天蓝色
        tree = [128, 128, 0]  # 绿棕色
        sign_symbol = [192, 128, 128]  # 浅红色
        fence = [64, 64, 128]  # 午夜蓝
        car = [64, 0, 128]  # 深岩暗蓝灰色
        pedestrian = [64, 64, 0]  # 棕色
        bicyclist = [0, 128, 192]  # 道奇蓝
        unlabelled = [0, 0, 0]  # 纯黑色

        self.color_dict_ = np.array([sky, building, pole, road, pavement, tree, sign_symbol, fence, car, pedestrian,
                                     bicyclist, unlabelled])

    @staticmethod
    def adjust(image, mask, flag_multi_class, num_class):
        """
        调整图像数值从 0-255到 0-1之间，并把 mask 图像调整为 0,1 二值图像。
        :param image: 原图像
        :param mask: groud truth 图像
        :param flag_multi_class: 标识是否是多类
        :param num_class: 类个数
        :return: 调整后的图像和 mask图像
        """
        if flag_multi_class:
            image = image / 255
            mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
            new_mask = np.zeros(mask.shape + (num_class,))
            for i in range(num_class):
                new_mask[mask == i, i] = 1
            mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
                                         new_mask.shape[3]))

        # 如果只有黑白两类，则把图像映射到 0~1，把 mask 转化为黑白0,1图像
        elif np.max(image) > 1:
            image = image / 255
            mask = mask / 255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0

        return image, mask

    def train_generator(self, batch_size, train_path, images_folder, masks_folder, aug_dict,
                        image_color_mode='grayscale', mask_color_mode='grayscale',
                        image_save_prefix='image', mask_save_prefix='mask', flag_multi_class=False,
                        num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
        """
        同时生成图像 image和 mask，使用相同的种子保证 image和 mask变换相同。
        如果想查看生成的结果，可设置 save_to_dir
        :param batch_size: 默认32，批数据量大小
        :param train_path: 训练集图片路径
        :param images_folder: 原始图片路径
        :param masks_folder: mask 图片路径
        :param aug_dict: 数据增强参数字典
        :param image_color_mode: 是否是灰度图像或 RGB 图像
        :param mask_color_mode: 是否是灰度图像或 RGB 图像
        :param image_save_prefix: 增强图像保存前缀名
        :param mask_save_prefix: 增加 mask 保存前缀名
        :param flag_multi_class: 多类标识 False or True， 默认只有两类，值为 False
        :param num_class: 类别数，默认是两类
        :param save_to_dir: 保存增强图片数据的路径，用于可视化查看
        :param target_size: 图片输入尺寸
        :param seed: 随机增强的种子，image 和 mask 应保持相同
        :return: 每批增强的图片和相应的 mask, 返回对象是生成器
        """
        image_data_generator = ImageDataGenerator(**aug_dict)
        mask_data_generator = ImageDataGenerator(**aug_dict)
        image_generator = image_data_generator.flow_from_directory(
            train_path,
            classes=[images_folder],
            class_mode=None,
            color_mode=image_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            seed=seed
        )
        mask_generator = mask_data_generator.flow_from_directory(
            train_path,
            classes=[masks_folder],
            class_mode=None,
            color_mode=mask_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=mask_save_prefix,
            seed=seed
        )
        for image, mask in zip(image_generator, mask_generator):
            yield self.adjust(image, mask, flag_multi_class, num_class)

    @staticmethod
    def test_generator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False,
                       as_gray=True):
        """
        产生测试数据的生成器
        :param test_path: 测试图片路径
        :param num_image: 测试图片个数
        :param target_size: 图片尺寸
        :param flag_multi_class: 是否是多类别
        :param as_gray: 是否以灰度图像作为输入
        :return: 满足格式的测试图片
        """
        for i in range(num_image):
            image = io.imread(os.path.join(test_path, '%d.png' % i), as_gray=as_gray)
            image = image / 255
            image = transform.resize(image, target_size)
            image = np.reshape(image, image.shape + (1,)) if (not flag_multi_class) else image
            image = np.reshape(image, (1,) + image.shape)
            yield image

    def generator_train_npy(self, image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix='image',
                            mask_prefix='mask', image_as_gray=True, mask_as_gray=True):
        """
        把图片和 mask 作为 array 对象生成训练数据。当运行内存不足时不建议使用此方法
        :param image_path: 图片路径
        :param mask_path: mask 路径
        :param flag_multi_class: 是否是多类
        :param num_class: 类别个数
        :param image_prefix: 图片名前缀字段
        :param mask_prefix: mask 前缀字段
        :param image_as_gray: 是否将图片转化为灰度图像
        :param mask_as_gray: 是否将 mask 转化为灰度图像
        :return: image and mask 组成的 numpy array 训练数据集
        """
        image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
        image_arr = []
        mask_arr = []
        for index, item in enumerate(image_name_arr):
            image = io.imread(item, as_gray=image_as_gray)
            image = np.reshape(image, image.shape + (1,)) if image_as_gray else image
            mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix),
                             as_gray=mask_as_gray)
            mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
            image, mask = self.adjust(image, mask, flag_multi_class, num_class)
            image_arr.append(image)
            mask_arr.append(mask)
        image_arr = np.array(image_arr)
        mask_arr = np.array(mask_arr)
        return image_arr, mask_arr

    @staticmethod
    def label_visualize(num_class, color_dict, image):
        """
        将图片可视化显示，并使用相应的颜色
        :param num_class: 类个数
        :param color_dict: 颜色字典
        :param image: 带显示的图片
        :return: 带显示的彩色图片
        """
        image = image[:, :, 0] if len(image.shape) == 3 else image
        image_out = np.zeros(image.shape + (3,))
        for i in range(num_class):
            image_out[image == i, i] = color_dict[i]
        return image_out / 255

    def save_result(self, save_path, npy_file, flag_multi_class=False, num_class=2):
        """
        把预测的 mask 结果保持起来，一共查看
        :param save_path: 保存目录
        :param npy_file: 图片组成的 numpy array 数据集
        :param flag_multi_class: 是否是多类，超过两类
        :param num_class: 类别个数
        :return: 0
        """
        for i, item in enumerate(npy_file):
            image = self.label_visualize(num_class, self.color_dict_, item) if flag_multi_class else item[:, :, 0]
            io.imsave(os.path.join(save_path, "%d_predict.png" % i), img_as_ubyte(image))


if __name__ == '__main__':
    aug = AUGMENTATION()
    path = '../data/membrane/train/'
    image_ = io.imread(path + 'images/0.png', as_gray=True)
    mask_ = io.imread(path + 'ground_truth/0.png', as_gray=True)
    multi_class_ = False
    num_class_ = 2
    aug.adjust(image=image_, mask=mask_, flag_multi_class=multi_class_, num_class=num_class_)
