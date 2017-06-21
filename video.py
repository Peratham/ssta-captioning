# coding: utf-8
'''
载入显著图特征，用vgg16提取
'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import cv2
import h5py
import numpy as np
import skimage
import torch
from torch.autograd import Variable
from model import EncoderCNN
from args import video_root, video_sort_lambda
from args import feature_h5_path, feature_h5_feats, feature_h5_lens
from args import num_frames, frame_size


def sample_frames(video_path, train=True):
    '''
    对视频帧进行采样，减少计算量。等间隔地取num_frames帧
    '''
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frame_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame_list.append(frame)
        frame_count += 1

    indices = np.linspace(0, frame_count, num_frames, endpoint=False, dtype=int)
    frame_list = np.array(frame_list)[indices]
    return frame_list, frame_count


def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # 把单通道的灰度图复制三遍变成三通道的图片
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height),
                                           target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height,
                                           int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_ubyte(image).astype(np.float32)
    # 减去在ILSVRC数据集上的图像的均值（BGR格式）
    image -= np.array([103.939, 116.779, 123.68])
    # 把BGR的图片转换成RGB的图片，因为之后的模型（caffe预训练版）用的是RGB格式
    image = image[:, :, ::-1]
    return image


def extract_full_feature(encoder):
    # 读取视频列表，让视频按照id升序排列
    videos = sorted(os.listdir(video_root), key=video_sort_lambda)
    nvideos = len(videos)

    # 创建保存视频特征的hdf5文件
    if os.path.exists(feature_h5_path):
        # 如果hdf5文件已经存在，说明之前处理过，或许是没有完全处理完
        # 使用r+ (read and write)模式读取，以免覆盖掉之前保存好的数据
        h5 = h5py.File(feature_h5_path, 'r+')
        dataset_feats = h5[feature_h5_feats]
        dataset_lens = h5[feature_h5_lens]
    else:
        h5 = h5py.File(feature_h5_path, 'w')
        dataset_feats = h5.create_dataset(feature_h5_feats,
                                          (nvideos, num_frames, frame_size),
                                          dtype='float32')
        dataset_lens = h5.create_dataset(feature_h5_lens, (nvideos,), dtype='int')

    for i, video in enumerate(videos):
        print(video, end=' ')
        video_path = os.path.join(video_root, video)
        # 提取视频帧
        frame_list, frame_count = sample_frames(video_path, train=True)
        print(frame_count)

        # 把图像做一下处理，然后转换成（batch, channel, height, width）的格式
        cropped_frame_list = np.array([preprocess_frame(x) for x in frame_list])
        cropped_frame_list = cropped_frame_list.transpose((0, 3, 1, 2))
        cropped_frame_list = Variable(torch.from_numpy(cropped_frame_list),
                                      volatile=True).cuda()

        # 视频特征的shape是num_frames x 4096
        # 如果帧的数量小于num_frames，则剩余的部分用0补足
        feats = np.zeros((num_frames, frame_size), dtype='float32')
        feats[:frame_count, :] = encoder(cropped_frame_list).data.cpu().numpy()
        dataset_feats[i] = feats
        dataset_lens[i] = frame_count


def main():
    encoder = EncoderCNN()
    encoder.eval()
    encoder.cuda()

    extract_full_feature(encoder)


if __name__ == '__main__':
    main()
