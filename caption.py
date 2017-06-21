# coding: utf-8
'''
准备文本相关的数据集，包括：
1. 对数据集进行划分
2. 把caption变成tokens
3. 准备ground-truth
'''
from __future__ import print_function
from __future__ import absolute_import
import json
import nltk
import pickle
import pprint
from collections import Counter

from args import ds
from args import train_range, val_range, test_range
from args import anno_json_path, vocab_pkl_path
from args import train_caption_pkl_path, val_caption_pkl_path, test_caption_pkl_path
from args import num_words  # 文本序列的规定长度
from args import val_reference_txt_path, test_reference_txt_path
from utils import create_reference_json, build_msvd_annotation
import torch


class Vocabulary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.nwords = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, w):
        '''
        将新单词加入词汇表中
        '''
        if w not in self.word2idx:
            self.word2idx[w] = self.nwords
            self.idx2word.append(w)
            self.nwords += 1

    def __call__(self, w):
        '''
        返回单词对应的id
        '''
        if w not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[w]

    def __len__(self):
        '''
        得到词汇表中词汇的数量
        '''
        return self.nwords


def prepare_vocab(sentences):
    '''
    根据标注的文本得到词汇表。频数低于threshold的单词将会被略去
    '''
    counter = Counter()
    ncaptions = len(sentences)
    for i, row in enumerate(sentences):
        caption = row['caption']
        # 直接按照空格进行单词的切分
        # tokens = caption.lower().split(' ')
        # 使用nltk来进行单词切分
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        if i % 10000 == 0:
            print('[{}/{}] tokenized the captions.'.format(i, ncaptions))

    # 略去一些低频词
    threshold = 3
    words = [w for w, c in counter.items() if c >= threshold]
    # 开始构建词典！
    vocab = Vocabulary()
    for w in words:
        vocab.add_word(w)

    print('Vocabulary has %d words.' % len(vocab))
    with open(vocab_pkl_path, 'wb') as f:
        pickle.dump(vocab, f)
    print('Save vocabulary to %s' % vocab_pkl_path)
    return vocab


def prepare_split():
    '''
    为数据集生成train，val，test的划分。MSVD数据集可以根据Vsubhashini的划分：
    train:1-1200, val:1201-1300, test:1301-1970
    '''
    split_dict = {}

    for i in range(*train_range):
        split_dict[i] = 'train'
    for i in range(*val_range):
        split_dict[i] = 'val'
    for i in range(*test_range):
        split_dict[i] = 'test'

    # pprint.pprint(split_dict)

    return split_dict


def prepare_caption(vocab, split_dict, anno_data):
    '''
    把caption转换成token index表示然后存到picke中
    读取存储文本标注信息的json文件，
    并且将每一条caption以及它对应的video的id保存起来，
    放回caption word_id list和video_id list
    '''
    # 初始化数据存储字典
    captions = {'train': [], 'val': [], 'test': []}
    lengths = {'train': [], 'val': [], 'test': []}
    video_ids = {'train': [], 'val': [], 'test': []}

    count = 0
    for row in anno_data:
        caption = row['caption'].lower()
        video_id = int(row['video_id'][5:])
        if video_id in split_dict:
            split = split_dict[video_id]
        else:
            # 如果video_id不在split_dict中
            # 那么就默认它是test
            # 这样方便我修改split来做一些过拟合训练
            split = 'test'
        words = nltk.tokenize.word_tokenize(caption)
        l = len(words) + 1  # 加上一个<end>
        lengths[split].append(l)
        if l > num_words:
            # 如果caption长度超出了规定的长度，就做截断处理
            words = words[:num_words - 1]  # 最后要留一个位置给<end>
            count += 1
        # 把caption用word id来表示
        tokens = []
        for word in words:
            tokens.append(vocab(word))
        tokens.append(vocab('<end>'))
        while l < num_words:
            # 如果caption的长度少于规定的长度，就用<pad>（0）补齐
            tokens.append(vocab('<pad>'))
            l += 1
        captions[split].append(torch.LongTensor(tokens))
        video_ids[split].append(video_id)

    # 统计一下有多少的caption长度过长
    print('There are %.3f%% too long captions' % (100 * float(count) / len(anno_data)))

    # 分别对train val test这三个划分进行存储
    with open(train_caption_pkl_path, 'wb') as f:
        pickle.dump([captions['train'], lengths['train'], video_ids['train']], f)
        print('Save %d train captions to %s.' % (len(captions['train']),
                                                 train_caption_pkl_path))
    with open(val_caption_pkl_path, 'wb') as f:
        pickle.dump([captions['val'], lengths['val'], video_ids['val']], f)
        print('Save %d val captions to %s.' % (len(captions['val']),
                                               val_caption_pkl_path))
    with open(test_caption_pkl_path, 'wb') as f:
        pickle.dump([captions['test'], lengths['test'], video_ids['test']], f)
        print('Save %d test captions to %s.' % (len(captions['test']),
                                                test_caption_pkl_path))


def prepare_gt(anno_data):
    '''
    准备ground-truth,用来评估结果的好坏
    '''
    print('Preparing ground-truth...')
    val_reference_txt = open(val_reference_txt_path, 'w')
    test_reference_txt = open(test_reference_txt_path, 'w')

    val_selected_range = range(*val_range)
    test_selected_range = range(*test_range)
    error_count = 0

    for row in anno_data:
        caption = row['caption'].lower()
        video_id = int(row['video_id'][5:])
        gt = '%d\t%s\n' % (video_id, caption)
        try:
            if video_id in val_selected_range:
                val_reference_txt.write(gt)
            elif video_id in test_selected_range:
                test_reference_txt.write(gt)
        except Exception as e:
            print(e)
            print(gt)
            error_count += 1

    if error_count > 0:
        print('Error count: %d\t' % error_count, end='')

    val_reference_txt.close()
    test_reference_txt.close()

    create_reference_json(val_reference_txt_path)
    create_reference_json(test_reference_txt_path)
    print('done!')


if __name__ == '__main__':
    if ds == 'msvd':
        # 以MSR-VTT数据集的格式生成MSVD数据集的标注
        print('# Build MSVD dataset annotations:')
        build_msvd_annotation()

    # 读取json文件
    with open(anno_json_path, 'r') as f:
        anno_json = json.load(f)
    anno_data = anno_json['sentences']

    print('\n# Build vocabulary')
    vocab = prepare_vocab(anno_data)

    print('\n# Prepare dataset split')
    split_dict = prepare_split()

    print('\n# Convert each caption to token index list')
    prepare_caption(vocab, split_dict, anno_data)

    print('\n# Prepare ground-truth')
    prepare_gt(anno_data)
