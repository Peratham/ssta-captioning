# coding: utf-8

'''
利用显著图来做Temporal Attention，对视频内容进行筛选
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from builtins import range
from args import vgg_checkpoint
import random
import math


class EncoderCNN(nn.Module):

    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.vgg = models.vgg16()
        self.vgg.load_state_dict(torch.load(vgg_checkpoint))
        # 把VGG的最后一个fc层（其之前的ReLU层要保留）剔除掉
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])

    def forward(self, images):
        return self.vgg(images)


class AttentionLayer(nn.Module):
    '''
    根据LSTM的隐层状态和视频帧的CNN特征来确定该帧的权重
    '''
    def __init__(self, hidden_size, projected_size):
        '''
        hidden_size: LSTM的隐层单元数目
        frame_embed_size: CNN特征的嵌入维度
        projected_size: 处理LSTM特征和CNN特征的投影空间的维度
        '''
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.projected_size = projected_size
        self.linear1 = nn.Linear(hidden_size, projected_size)
        self.linear2 = nn.Linear(projected_size, projected_size)
        self.linear3 = nn.Linear(projected_size, 1, bias=False)

    def forward(self, h, v):
        bsz, num_frames = v.size()[:2]
        e = []
        for i in range(num_frames):
            x = self.linear1(h) + self.linear2(v[:, i, :])
            x = F.tanh(x)
            x = self.linear3(x)
            e.append(x)
        e = torch.cat(e, 0)
        a = F.softmax(e.view(bsz, num_frames))
        return a


class DecoderRNN(nn.Module):

    def __init__(self, frame_size, projected_size, hidden_size,
                 num_frames, num_words, vocab):
        '''
        frame_size: 视频帧的特征的大小，一般是4096（VGG的倒数第二个fc层）
        projected_size: 所有特征的投影维度
        hidden_size: LSTM的隐层单元个数
        num_frames: 视觉特征的序列长度，默认是60
        num_words: 文本特征的序列长度，默认是30
        '''
        super(DecoderRNN, self).__init__()

        self.frame_size = frame_size
        self.hidden_size = hidden_size
        self.num_frames = num_frames
        self.num_words = num_words
        self.projected_size = projected_size
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # frame_embed用来把视觉特征嵌入到低维空间
        self.vs_frame_embed = nn.Linear(frame_size, projected_size)
        self.vs_frame_drop = nn.Dropout(p=0.8)
        self.vf_frame_embed = nn.Linear(frame_size, projected_size)
        self.vf_frame_drop = nn.Dropout(p=0.8)
        self.frame_embed = nn.Linear(projected_size * 2, projected_size)
        self.frame_drop = nn.Dropout(p=0.8)

        # attend_layer用来做temporal attention
        self.attend_layer = AttentionLayer(hidden_size, projected_size)

        # word_embed用来把文本特征嵌入到低维空间
        self.word_embed = nn.Embedding(self.vocab_size, projected_size)
        self.word_drop = nn.Dropout(p=0.8)

        # lstm作为解码器
        self.lstm_cell = nn.LSTMCell(projected_size, hidden_size)
        self.lstm_drop = nn.Dropout(p=0.8)
        # inith用来初始化lstm的hidden
        self.inith = nn.Sequential(
            nn.Linear(projected_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        # initc用来初始化lstm的cell
        self.initc = nn.Sequential(
            nn.Linear(projected_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # linear用来把lstm的最终输出映射回文本空间
        self.linear = nn.Linear(hidden_size, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        variance = math.sqrt(2.0 / (self.frame_size + self.projected_size))
        self.vs_frame_embed.weight.data.normal_(0.0, variance)
        self.vs_frame_embed.bias.data.zero_()
        self.vf_frame_embed.weight.data.normal_(0.0, variance)
        self.vf_frame_embed.bias.data.zero_()
        self.word_embed.weight.data.uniform_(-1.73, 1.73)
        self.linear.weight.data.uniform_(-0.08, 0.08)
        self.linear.bias.data.zero_()

    def _init_lstm_state(self, v):
        mean_v = torch.mean(v, 1).squeeze(1)
        lstm_hidden = F.tanh(self.inith(mean_v))
        lstm_cell = F.tanh(self.initc(mean_v))
        return lstm_hidden, lstm_cell

    def forward(self, video_feats, captions, teacher_forcing_ratio=0.5):
        '''
        传入视频帧特征和caption，返回生成的caption
        不用teacher forcing模式（LSTM的输入来自caption的ground-truth）来训练
        而是用上一步的生成结果作为下一步的输入
        UPDATED: 最后还是采用了混合的teacher forcing模式，不然很难收敛
        '''
        batch_size = len(video_feats)
        # 根据是否传入caption判断是否是推断模式
        infer = True if captions is None else False

        # Encoding 阶段！
        # vs是视频帧的saliency区域的特征
        vs = video_feats[:, :, :self.frame_size].contiguous()
        vs = vs.view(-1, self.frame_size)
        vs = self.vs_frame_embed(vs)
        vs = self.vs_frame_drop(vs)
        vs_ = vs.view(batch_size, self.num_frames, -1)
        # vf是视频帧的完整特征
        vf = video_feats[:, :, self.frame_size:].contiguous()
        vf = vf.view(-1, self.frame_size)
        vf = self.vf_frame_embed(vf)
        vf = self.vf_frame_drop(vf)
        # vf_ = vf_.view(batch_size, self.num_frames, -1)
        # vr是视频完整特征与显著区域特征的残差
        vr = vf - vs
        # v是视频的著特征与残差特征的拼接
        v = torch.cat([vs, vr], 1)
        v = self.frame_embed(v)
        v = v.view(batch_size, self.num_frames, -1)

        # 初始化LSTM隐层
        lstm_hidden, lstm_cell = self._init_lstm_state(v)

        # Decoding 阶段！
        # 开始准备输出啦！
        outputs = []
        attens = []
        # 先送一个<start>标记
        word_id = self.vocab('<start>')
        word = Variable(vs.data.new(batch_size, 1).long().fill_(word_id))
        word = self.word_embed(word).squeeze(1)
        word = self.word_drop(word)

        for i in range(self.num_words):
            if not infer and captions[:, i].data.sum() == 0:
                # <pad>的id是0，如果所有的word id都是0，
                # 意味着所有的句子都结束了，没有必要再算了
                break
            a = self.attend_layer(lstm_hidden, vs_)
            if infer:
                attens.append(a)
            a = a.unsqueeze(1)
            # 考虑视频的完整特征与显著区域特征的拼接
            V = torch.bmm(a, v).squeeze(1)

            t = word + V
            lstm_hidden, lstm_cell = self.lstm_cell(t, (lstm_hidden, lstm_cell))
            lstm_hidden = self.lstm_drop(lstm_hidden)

            word_logits = self.linear(lstm_hidden)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                # teacher forcing模式
                word_id = captions[:, i]
            else:
                # 非 teacher forcing模式
                word_id = word_logits.max(1)[1]
            if infer:
                # 如果是推断模式，直接返回单词id
                outputs.append(word_id)
            else:
                # 否则是训练模式，要返回logits
                outputs.append(word_logits)
            # 确定下一个输入单词的表示
            word = self.word_embed(word_id).squeeze(1)
            word = self.word_drop(word)
        # unsqueeze(1)会把一个向量(n)拉成列向量(nx1)
        # outputs中的每一个向量都是整个batch在某个时间步的输出
        # 把它拉成列向量之后再横着拼起来，就能得到整个batch在所有时间步的输出
        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1).contiguous()
        return outputs, attens

    def sample(self, video_feats):
        '''
        sample就是不给caption且不用teacher forcing的forward
        '''
        return self.forward(video_feats, None, teacher_forcing_ratio=0.0)

    def decode_tokens(self, tokens):
        '''
        根据word id（token）列表和给定的字典来得到caption
        '''
        words = []
        for token in tokens:
            if token == self.vocab('<end>'):
                break
            word = self.vocab.idx2word[token]
            words.append(word)
        caption = ' '.join(words)
        return caption
