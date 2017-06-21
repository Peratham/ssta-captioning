# coding: utf-8

import pickle
import h5py
import torch
import torch.utils.data as data
from args import train_caption_pkl_path
from args import feature_h5_path, feature_h5_feats


class V2TDataset(data.Dataset):
    '''
    Video to Text数据集的描述类，用来加载和提供数据
    支持MSR-VTT和MSVD数据集
    构造的时候需要以下输入：
    1. 提供文本特征的pkl文件
    2. 包含视频帧信息的h5文件
    提供文本和视频h5特征，以及根据caption的id来返回数据
    '''

    def __init__(self, cap_pkl, feature_h5):
        with open(cap_pkl, 'rb') as f:
            self.captions, self.lengths, self.video_ids = pickle.load(f)
        h5_file = h5py.File(feature_h5, 'r')
        self.video_feats = h5_file[feature_h5_feats]

    def __getitem__(self, index):
        '''
        返回一个训练样本对（包含视频frame特征和对应的caption）
        根据caption来找对应的video，所以要求video存储的时候是按照id升序排列的
        '''
        caption = self.captions[index]
        length = self.lengths[index]
        video_id = self.video_ids[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])
        return video_feat, caption, length, video_id

    def __len__(self):
        return len(self.captions)


class VideoDataset(data.Dataset):
    '''
    仅提供视频特征以及相应ID的数据加载类，
    之所以单独提供这个类是希望加速评价指标的计算
    '''
    def __init__(self, eval_range, feature_h5):
        self.eval_list = tuple(range(*eval_range))
        h5_file = h5py.File(feature_h5, 'r')
        self.video_feats = h5_file[feature_h5_feats]

    def __getitem__(self, index):
        '''
        返回一个训练样本对（包含视频特征和对应的ID）
        '''
        video_id = self.eval_list[index]
        video_feat = torch.from_numpy(self.video_feats[video_id])
        return video_feat, video_id

    def __len__(self):
        return len(self.eval_list)


def train_collate_fn(data):
    '''
    用来把多个数据样本合并成一个minibatch的函数
    '''
    # 根据video的长度对数据进行排序
    data.sort(key=lambda x: x[-1], reverse=True)

    videos, captions, lengths, video_ids = zip(*data)

    # 把视频合并在一起（把2D Tensor的序列变成3D Tensor）
    videos = torch.stack(videos, 0)

    # 把caption合并在一起（把1D Tensor的序列变成一个2D Tensor）
    captions = torch.stack(captions, 0)
    return videos, captions, lengths, video_ids


def eval_collate_fn(data):
    '''
    用来把多个数据样本合并成一个minibatch的函数
    '''
    data.sort(key=lambda x: x[-1], reverse=True)

    videos, video_ids = zip(*data)

    # 把视频合并在一起（把2D Tensor的序列变成3D Tensor）
    videos = torch.stack(videos, 0)

    return videos, video_ids


def get_train_loader(cap_pkl, feature_h5, batch_size=10, shuffle=True, num_workers=3, pin_memory=True):
    v2t = V2TDataset(cap_pkl, feature_h5)
    data_loader = torch.utils.data.DataLoader(dataset=v2t,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=train_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


def get_eval_loader(cap_pkl, feature_h5, batch_size=200, shuffle=False, num_workers=1, pin_memory=False):
    vd = VideoDataset(cap_pkl, feature_h5)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=eval_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


if __name__ == '__main__':
    train_loader = get_train_loader(train_caption_pkl_path, feature_h5_path)
    print(len(train_loader))
    d = next(iter(train_loader))
    print(d[0].size())
    print(d[1].size())
    print(len(d[2]))
