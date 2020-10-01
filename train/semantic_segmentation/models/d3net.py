"""
Inspired from
https://github.com/marcelampc/d3net_depth_estimation
"""


import functools
import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import cat
from torch.nn import init
from torchvision import models

# from ipdb import set_trace as st


__all__ = ["DenseNet", "densenet121", "densenet169", "densenet201", "densenet161"]


model_urls = {
    "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
    "densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
    "densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
    "densenet161": "https://download.pytorch.org/models/densenet161-8d451a50.pth",
}

act = nn.LeakyReLU(0.2, inplace=True)


def get_conv_type(upsample):
    if upsample:
        return nn.ConvTranspose2d
    else:
        return nn.Conv2d


def conv4x4(inplanes, outplanes, upsample=False, padding=1):
    # convolution will be Conv2d, or ConvTranspose2d
    _conv = get_conv_type(upsample)
    return _conv(inplanes, outplanes, kernel_size=4, stride=2, padding=padding)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BasicBlock, self).__init__()
        self.dropout = use_dropout

        # self.bn0 = nn.BatchNorm2d(inplanes)
        # self.relu0 = nn.ReLU()

        # Upsample/ Downsample
        self.conv1 = conv4x4(inplanes, outplanes, upsample=True)
        self.bn1 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        # self.drop = CDropout(outplanes, 0.2)

        self.conv2 = conv3x3(outplanes, outplanes)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x

        # out = self.bn0(out)
        # out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.relu1(out)

        # out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        return out


class BasicBlockToCrop(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BasicBlockToCrop, self).__init__()
        self.dropout = use_dropout

        # self.bn0 = nn.BatchNorm2d(inplanes)
        # self.relu0 = nn.ReLU()

        # Upsample/ Downsample
        self.conv1 = conv4x4(inplanes, outplanes, upsample=True, padding=0)
        self.bn1 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        # self.drop = CDropout(outplanes, 0.2)

        self.conv2 = conv3x3(outplanes, outplanes)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x

        # out = self.bn0(out)
        # out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.relu1(out)

        # out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        return out


class BasicBlockCU(nn.Module):
    # upsample at the end
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BasicBlockCU, self).__init__()
        self.dropout = use_dropout

        # self.bn0 = nn.BatchNorm2d(inplanes)
        # self.relu0 = nn.ReLU()

        # Upsample/ Downsample
        self.conv1 = conv3x3(inplanes, outplanes)
        self.bn1 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        # self.drop = CDropout(outplanes, 0.2)

        self.conv2 = conv4x4(outplanes, outplanes, upsample=True)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x

        # out = self.bn0(out)
        # out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.relu1(out)

        # out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        return out


class BasicBlock2(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BasicBlock2, self).__init__()
        self.dropout = use_dropout

        # self.bn0 = nn.BatchNorm2d(inplanes)
        # self.relu0 = nn.ReLU()

        # Upsample/ Downsample
        self.conv1 = conv4x4(inplanes, outplanes, upsample=True)
        self.bn1 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(outplanes, outplanes)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(outplanes, outplanes)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout3 = nn.Dropout(0.5)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x

        # out = self.bn0(out)
        # out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropout:
            out = self.dropout3(out)
        out = self.relu3(out)

        return out


class BasicBlock5x5(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BasicBlock5x5, self).__init__()
        self.dropout = use_dropout

        # self.bn0 = nn.BatchNorm2d(inplanes)
        # self.relu0 = nn.ReLU()

        # Upsample/ Downsample
        self.conv1 = conv4x4(inplanes, outplanes, upsample=True)
        self.bn1 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = self.conv(outplanes, outplanes)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

    def conv(self, in_planes, out_planes, stride=1):
        "5x5 convolution with padding"
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False
        )

    def forward(self, x):
        out = x

        # out = self.bn0(out)
        # out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        return out


class BilinearBlock(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BilinearBlock, self).__init__()
        self.dropout = use_dropout

        # Upsample/ Downsample
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.conv3x3 = conv3x3(inplanes, outplanes)

        self.bn2 = nn.BatchNorm2d(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.upsample(x)

        out = self.conv3x3(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        return out


class UpsampleBlock(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(UpsampleBlock, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv", conv4x4(num_input_features, num_output_features, upsample=True)
        )


# class DecoderDenseBlock(nn.Module):
#     def __init__(self, num_init_features, block_config, bn_size=4,
#                  drop_rate=0, growth_rate=32):
#         super(DecoderDenseBlock, self).__init__()

#         num_features = num_init_features
#         for i, num_layers in enumerate(reversed(block_config)):
#             block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
#                                 bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
#             self.features.add_module('denseblock%d' % (i + 1), block)
#             num_features = num_features + num_layers * growth_rate
#             if i != len(block_config) - 1:
#                 trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
#                 self.features.add_module('transition%d' % (i + 1), trans)
#                 self.features.add_module('transition%dpool' % (i + 1), nn.AvgPool2d(kernel_size=2, stride=2))
#                 num_features = num_features // 2

#     def forward(self, x):

#         return x


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm.1", nn.BatchNorm2d(num_input_features)),
        self.add_module("relu.1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv.1",
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        ),
        self.add_module("norm.2", nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module("relu.2", nn.ReLU(inplace=True)),
        self.add_module(
            "conv.2",
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate
            )
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module(
            "upsample", conv4x4(num_output_features, num_output_features, upsample=True)
        )
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


def get_decoder_block(block_type):
    if block_type == "bilinear":
        return BilinearBlock
    if block_type == "basic":
        return BasicBlock
    if block_type == "basictocrop":
        return BasicBlockToCrop
    if block_type == "basic_conv_up":
        return BasicBlockCU
    if block_type == "basic2":
        return BasicBlock2
    if block_type == "basic5x5":
        return BasicBlock5x5
    # if block_type == 'dense':
    #     return DenseBlock
    if block_type == "residual":
        pass


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find("BatchNorm2d") != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm2d") != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find("BatchNorm2d") != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type="normal"):
    # print('initialization method [%s]' % init_type)
    if init_type == "normal":
        net.apply(weights_init_normal)
    elif init_type == "uniform":
        net.apply(weights_init_uniform)
    elif init_type == "xavier":
        net.apply(weights_init_xavier)
    elif init_type == "kaiming":
        net.apply(weights_init_kaiming)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            "initialization method [%s] is not implemented" % init_type
        )


# def which_dropout(type = opt.which_dropout):
#     if type == "cdropout":
#         return CDropout
#     else:
#         F.dropout


normalization_layer = nn.BatchNorm2d
# normalization_layer = nn.LayerNorm


def denseUnet121(
    pretrained=False,
    d_block_type="basic",
    init_method="normal",
    version=1,
    type_net="t",
    **kwargs
):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    d_block = BasicBlock
    model = DenseUNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        d_block=d_block,
        **kwargs
    )

    if pretrained:
        w_init.init_weights(model, init_method)
        # Get state dict from the actual model
        model_dict = model.state_dict()
        pretrained_dict = models.densenet121(pretrained=True).state_dict()
        # exclude_model_dict = ["features.conv0.weight"]
        model_shapes = [v.shape for k, v in model_dict.items()]
        exclude_model_dict = []
        exclude_model_dict = [
            k for k, v in pretrained_dict.items() if v.shape not in model_shapes
        ]
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and k not in exclude_model_dict
        }

        # added to pytorch 0.4
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        # state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(pretrained_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                pretrained_dict[new_key] = pretrained_dict[key]
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True, use_dropout=False):
        super(BasicBlock, self).__init__()
        self.dropout = use_dropout

        # self.bn0 = nn.BatchNorm2d(inplanes)
        # self.relu0 = nn.ReLU()

        # Upsample/ Downsample
        self.conv1 = conv4x4(inplanes, outplanes, upsample=True)
        self.bn1 = normalization_layer(outplanes)
        if self.dropout:
            self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        # self.drop = CDropout(outplanes, 0.2)

        self.conv2 = conv3x3(outplanes, outplanes)
        # self.conv2 = nn.ConvTranspose2d(outplanes, outplanes, kernel_size=3, stride=1,
        #                                 padding=1, bias=False)
        self.bn2 = normalization_layer(outplanes)
        if self.dropout:
            self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x

        # out = self.bn0(out)
        # out = self.relu0(out)

        out = self.conv1(out)
        out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        out = self.relu1(out)

        # out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout:
            out = self.dropout2(out)
        out = self.relu2(out)

        return out


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", normalization_layer(num_input_features)),
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        ),
        # self.add_module("cdrop", CDropout(bn_size * growth_rate, 0.2)),  # add alexandre dropout
        self.add_module("norm2", normalization_layer(bn_size * growth_rate)),
        self.add_module("relu2", nn.ReLU(inplace=True)),
        self.add_module(
            "conv2",
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate
            )
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", normalization_layer(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


def center_crop(layer, max_height, max_width):
    # https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/merge.py#L162
    # Author does a center crop which crops both inputs (skip and upsample) to size of minimum dimension on both w/h
    batch_size, n_channels, layer_height, layer_width = layer.size()
    xy1 = (layer_width - max_width) // 2
    xy2 = (layer_height - max_height) // 2
    return layer[:, :, xy2 : (xy2 + max_height), xy1 : (xy1 + max_width)]


class _TransitionUp(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionUp, self).__init__()
        self.transition_upsample = nn.Sequential()
        self.transition_upsample.add_module(
            "d_transition1", _Transition(num_input_features, num_input_features // 2)
        )
        num_features = num_input_features // 2
        self.transition_upsample.add_module(
            "upsample", UpsampleBlock(num_features, num_features)
        )
        # center crop
        self.last_transition = nn.Sequential()
        self.last_transition.add_module(
            "d_transition2", _Transition(num_input_features, num_output_features)
        )

    def forward(self, x, skip):
        out = self.transition_upsample(x)
        # print(out.size(2))
        out = center_crop(out, skip.size(2), skip.size(3))
        # print(skip.size(2))
        out = torch.cat([out, skip], 1)
        out = self.last_transition(out)
        return out


########################################### NETWORKS ###########################################


class DenseUNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(
        self,
        d_block,
        input_nc=3,
        outputs_nc=[1],
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        use_dropout=False,
        use_skips=True,
        bilinear_trick=False,
        outputSize=[427, 571],
        tasks=["depth"],
    ):

        super(DenseUNet, self).__init__()

        self.use_skips = use_skips
        self.bilinear_trick = bilinear_trick
        self.tasks = tasks
        # for GradNorm
        # self.register_parameter('omegas', None) # not needed, wanted to do this outside, but the optimizer is not taking it into account
        # self._reset_omegas(len(tasks))

        # self.omegas = torch.nn.Parameter(torch.ones(len(tasks)).float())
        if self.use_skips:
            ngf_mult = 2
        else:
            ngf_mult = 1

        self.relu_type = nn.LeakyReLU(0.2, inplace=True)  # nn.ReLU(inplace=True)

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            input_nc,
                            num_init_features,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("norm0", normalization_layer(num_init_features)),
                    ("relu0", self.relu_type),
                    # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "downconv0",
                        nn.Conv2d(
                            num_init_features,
                            num_init_features,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("norm1", normalization_layer(num_init_features)),
                    ("relu1", self.relu_type),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                self.features.add_module(
                    "transition%dpool" % (i + 1), nn.AvgPool2d(kernel_size=2, stride=2)
                )
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", normalization_layer(num_features))

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)

        self.decoder = nn.Sequential()
        # for i in reversed(range(2, 6)): # common decoder until block 3
        for i in reversed(range(3, 6)):  # common decoder until block 3
            mult = 1 if i == 5 else ngf_mult
            dropout = use_dropout if i > 3 else False
            self.decoder.add_module(
                "d_block{}".format(i),
                self._make_decoder_layer(
                    num_features * mult,
                    int(num_features / 2),
                    block=d_block,
                    use_dropout=dropout,
                ),
            )
            num_features = int(num_features / 2)

        mult = ngf_mult
        self.decoder_tasks = nn.ModuleList()
        # each decoder block
        for task_i in range(len(tasks)):
            # print(task_i)
            task_block = nn.Sequential()
            task_block.add_module(
                "d_block{}".format(i - 1),  # 2
                self._make_decoder_layer(
                    num_features * mult,
                    num_features // 2,
                    block=d_block,
                    use_dropout=False,
                ),
            )

            num_features = num_features // 2

            task_block.add_module(
                "d_block{}".format(i - 2),  # 1
                self._make_decoder_layer(
                    num_features * mult, num_features, block=d_block, use_dropout=False
                ),
            )

            task_block.add_module(
                "last_conv", conv3x3(num_features, outputs_nc[task_i])
            )

            self.decoder_tasks.append(task_block)

            num_features = num_features * 2

    # def _reset_omegas(self, size):
    #     self.omegas = nn.Parameter(torch.ones(size).float())

    def _make_decoder_layer(self, inplanes, outplanes, block, use_dropout=True):
        layers = []
        layers.append(
            block(inplanes, outplanes, upsample=True, use_dropout=use_dropout)
        )
        return nn.Sequential(*layers)

    def get_decoder_input(self, e_out, d_out):
        if self.use_skips:
            return cat((e_out, d_out), 1)
        else:
            return d_out

    def forward(self, x):
        # features = self.features(x)
        # input is ngf x 256 x 256
        out = self.features.conv0(x)
        out = self.features.norm0(out)
        out_conv1 = self.features.relu0(out)
        # input is ngf x 128 x 128
        out = self.features.downconv0(out_conv1)
        out = self.features.norm1(out)
        out = self.features.relu1(out)

        # input is ngf x 64 x 64
        out = self.features.denseblock1(out)
        # input is ngf * 4 x 64 x 64
        tb_denseblock1 = self.features.transition1(out)  # transition block
        # input is ngf * 2 x 64 x 64
        out = self.features.transition1pool(tb_denseblock1)
        # input is ngf * 2 x 32 x 32
        out = self.features.denseblock2(out)
        # input is ngf * 8 x 32 x 32
        tb_denseblock2 = self.features.transition2(out)
        # input is ngf * 4 x 32 x 32
        out = self.features.transition2pool(tb_denseblock2)
        # input is ngf * 4 x 16 x 16
        out = self.features.denseblock3(out)
        # input is ngf * 16 x 16 x 16
        tb_denseblock3 = self.features.transition3(out)
        # input is ngf * 16 x 16 x 16
        out = self.features.transition3pool(tb_denseblock3)
        # input is ngf * 8 x 8 x 8
        out = self.features.denseblock4(out)
        # input is ngf * 16 x 8 x 8
        out = self.features.norm5(out)
        out = self.relu_type(out)

        # Here comes the decoder
        # input is ngf * 16 x 8 x 8
        out = self.decoder.d_block5(out)
        # input is (ngf * 8) x 16 x 16
        out = self.decoder.d_block4(self.get_decoder_input(tb_denseblock3, out))
        # input is (ngf * 4) x 32 x 32
        # last common layer for MTL
        out_d3 = self.decoder.d_block3(self.get_decoder_input(tb_denseblock2, out))

        self.last_common_layer = self.decoder.d_block3

        output = []

        # if 'depth' in self.tasks:
        for task_i in range(len(self.tasks)):
            # input is (ngf * 2) x 64 x 64
            # out_d2 = self.decoder_tasks.taski_block2()
            out_reg_d2 = self.decoder_tasks[task_i].d_block2(
                self.get_decoder_input(tb_denseblock1, out_d3)
            )
            # input is ngf x 128 x 128
            out_reg_d1 = self.decoder_tasks[task_i].d_block1(
                self.get_decoder_input(out_conv1, out_reg_d2)
            )
            # input is ngf x 256 x 256
            out_reg = self.decoder_tasks[task_i].last_conv(out_reg_d1)
            output.append(out_reg)

        return output

    def get_last_common_layer(self):
        return self.last_common_layer


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def define_G(in_channels, n_classes, net_architecture):
    use_dropout = False
    pretrained = False
    init_method = "normal"
    use_skips = True
    d_block_type = "basic"
    n_classes = [n_classes]  # --> output_nc
    tasks = ["depth"]

    netG = denseUnet121(
        input_nc=in_channels,
        outputs_nc=n_classes,
        pretrained=pretrained,
        init_method=init_method,
        use_dropout=use_dropout,
        use_skips=use_skips,
        d_block_type=d_block_type,
        tasks=tasks,
        type_net=net_architecture,
    )
    # print number of parameters of the network
    # print_network(netG)

    # if len(gpu_ids) > 0:
    #     netG.cuda(device_id=gpu_ids[0])

    if not pretrained:
        init_weights(netG, init_method)
    return netG


class D3Net(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(D3Net, self).__init__()
        in_channels += n_classes
        self.net = define_G(
            in_channels=in_channels, n_classes=n_classes, net_architecture="DenseUNet"
        )

    def forward(self, x):
        output = self.net(x)[0]
        return output
