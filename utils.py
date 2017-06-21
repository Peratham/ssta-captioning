# coding: utf-8

from __future__ import print_function
import os
import sys
import json
import hashlib
import pandas as pd
from args import msvd_csv_path, msvd_anno_json_path, msvd_video_name2id_map


# 关闭屏幕输出
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# 恢复屏幕输出
def enablePrint():
    sys.stdout = sys.__stdout__


class CocoAnnotations:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.img_dict = {}
        info = {
            "year": 2017,
            "version": '1',
            "description": 'Video CaptionEval',
            "contributor": 'Subhashini Venugopalan, Yangyu Chen',
            "url": 'https://github.com/vsubhashini/, https://github.com/Yugnaynehc/',
            "date_created": '',
        }
        licenses = [{"id": 1, "name": "test", "url": "test"}]
        self.res = {"info": info,
                    "type": 'captions',
                    "images": self.images,
                    "annotations": self.annotations,
                    "licenses": licenses,
                    }

    def read_multiple_files(self, filelist):
        for filename in filelist:
            print('In file %s\n' % filename)
            self.read_file(filename)

    def get_image_dict(self, img_name):
        code = img_name.encode('utf8')
        image_hash = int(int(hashlib.sha256(code).hexdigest(), 16) % sys.maxsize)
        if image_hash in self.img_dict:
            assert self.img_dict[image_hash] == img_name, 'hash colision: {0}: {1}'.format(
                image_hash, img_name)
        else:
            self.img_dict[image_hash] = img_name
        image_dict = {"id": image_hash,
                      "width": 0,
                      "height": 0,
                      "file_name": img_name,
                      "license": '',
                      "url": img_name,
                      "date_captured": '',
                      }
        return image_dict, image_hash

    def read_file(self, filename):
        count = 0
        with open(filename, 'r') as opfd:
            for line in opfd:
                count += 1
                id_sent = line.strip().split('\t')
                try:
                    assert len(id_sent) == 2
                    sent = id_sent[1]
                except Exception as e:
                    # print(line)
                    continue
                image_dict, image_hash = self.get_image_dict(id_sent[0])
                self.images.append(image_dict)

                self.annotations.append({
                    "id": len(self.annotations) + 1,
                    "image_id": image_hash,
                    "caption": sent,
                })

    def dump_json(self, outfile):
        self.res["images"] = self.images
        self.res["annotations"] = self.annotations
        with open(outfile, 'w') as fd:
            json.dump(self.res, fd, ensure_ascii=False, sort_keys=True,
                      indent=2, separators=(',', ': '))


def create_reference_json(reference_txt_path):
    output_file = '{0}.json'.format(reference_txt_path)
    crf = CocoAnnotations()
    crf.read_file(reference_txt_path)
    crf.dump_json(output_file)
    print('Created json references in %s' % output_file)


class CocoResFormat:

    def __init__(self):
        self.res = []
        self.caption_dict = {}

    def read_multiple_files(self, filelist, hash_img_name):
        for filename in filelist:
            print('In file %s\n' % filename)
            self.read_file(filename, hash_img_name)

    def read_file(self, filename, hash_img_name):
        count = 0
        with open(filename, 'r') as opfd:
            for line in opfd:
                count += 1
                id_sent = line.strip().split('\t')
                if len(id_sent) > 2:
                    id_sent = id_sent[-2:]
                assert len(id_sent) == 2
                sent = id_sent[1]

                if hash_img_name:
                    img_id = int(int(hashlib.sha256(id_sent[0].encode('utf8')).hexdigest(),
                                     16) % sys.maxsize)
                else:
                    img = id_sent[0].split('_')[-1].split('.')[0]
                    img_id = int(img)
                imgid_sent = {}

                if img_id in self.caption_dict:
                    assert self.caption_dict[img_id] == sent
                else:
                    self.caption_dict[img_id] = sent
                    imgid_sent['image_id'] = img_id
                    imgid_sent['caption'] = sent
                    self.res.append(imgid_sent)

    def dump_json(self, outfile):
        with open(outfile, 'w') as fd:
            json.dump(self.res, fd, ensure_ascii=False, sort_keys=True,
                      indent=2, separators=(',', ': '))


def build_msvd_annotation():
    '''
    仿照MSR-VTT数据集的格式，为MSVD数据集生成一个包含video信息和caption标注的json文件
    之所以要和MSR-VTT的格式相似，是因为所有的数据集要共用一套prepare_captions的代码
    '''
    # 首先根据MSVD数据集官方提供的CSV文件确定每段视频的名字
    video_data = pd.read_csv(msvd_csv_path, sep=',', encoding='utf8')
    video_data = video_data[video_data['Language'] == 'English']
    # 只使用clean的描述
    # 不行，有的视频没有clean的描述
    # video_data = video_data[video_data['Source'] == 'clean']
    video_data['VideoName'] = video_data.apply(lambda row: row['VideoID'] + '_' +
                                               str(row['Start']) + '_' +
                                               str(row['End']), axis=1)
    # 然后根据youtubeclips整理者提供的视频名字到视频id的映射构建一个词典
    video_name2id = {}
    with open(msvd_video_name2id_map, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name, vid = line.strip().split()
            # 提取出视频的数字id
            # 减1是因为id是从1开始的，但是之后处理的时候我们默认是0开始的
            # 因为实际上我们关系的是顺序，所以减1并不影响什么
            vid = int(vid[3:]) - 1
            # 再把vid变成video+数字id的形式
            # 不要问我为什么这么做<摊手>，因为MSR-VTT是这样的，好蠢啊...
            vid = 'video%d' % vid
            video_name2id[name] = vid

    # 开始准备按照MSR-VTT的结构构造json文件
    sents_anno = []
    not_use_video = []
    for name, desc in zip(video_data['VideoName'], video_data['Description']):
        if name not in video_name2id:
            if name not in not_use_video:
                print('No use: %s' % name)
                not_use_video.append(name)
            not_use_video.append(name)
            continue
        # 有个坑，SKhmFSV-XB0这个视频里面有一个caption的内容是NaN
        if type(desc) == float:
            print('Error annotation: %s\t%s' % (name, desc))
            continue
        d = {}
        # 放大招了! 过滤掉所有非ascii字符!
        desc = desc.encode('ascii', 'ignore').decode('ascii')
        # 还有很多新的坑! 有的句子带有一大堆\n或者带有\r\n
        desc = desc.replace('\n', '')
        desc = desc.replace('\r', '')
        # 有的句子有句号结尾,有的没有,甚至有的有多句.把句号以及多于一句的内容去掉
        # MSR-VTT数据集是没有句号结尾的
        desc = desc.split('.')[0]

        d['caption'] = desc
        d['video_id'] = video_name2id[name]
        sents_anno.append(d)

    anno = {'sentences': sents_anno}
    with open(msvd_anno_json_path, 'w') as f:
        json.dump(anno, f)
