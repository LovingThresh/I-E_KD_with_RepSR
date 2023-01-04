# -*- coding: utf-8 -*-
# @Time    : 2023/1/1 14:51
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : student.py
# @Software: PyCharm
import torch
import torchvision
from mmengine import MODELS
import torch.nn.functional as F
from mmengine.model import BaseModel

from mmcls.models import MobileViT

model = MobileViT()


@MODELS.register_module()
class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, inputs, data_samples, mode):
        x = self.resnet(inputs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, data_samples)}
        elif mode == 'predict':
            return x, data_samples

