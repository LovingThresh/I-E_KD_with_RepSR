# -*- coding: utf-8 -*-
# @Time    : 2023/1/1 16:11
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Attention_transfer.py
# @Software: PyCharm

import torch.nn as nn
from torch.nn.functional import normalize


class AT(nn.Module):
    """
    "Paying More Attention to Attention: Improving the Performance of
     Convolutional Neural Networks via Attention Transfer"
    Referred to https://github.com/szagoruyko/attention-transfer/blob/master/utils.py
    Discrepancy between Eq. (2) in the paper and the author's implementation
    https://github.com/szagoruyko/attention-transfer/blob/893df5488f93691799f082a70e2521a9dc2ddf2d/utils.py#L18-L23
    as partly pointed out at https://github.com/szagoruyko/attention-transfer/issues/34
    To follow the equations in the paper, use mode='paper' in place of 'code'
    """

    def __init__(self, mode='code'):
        super().__init__()
        self.mode = mode
        if mode not in ('code', 'paper'):
            raise ValueError('mode `{}` is not expected'.format(mode))

    @staticmethod
    def attention_transfer_paper(feature_map):
        return feature_map.pow(2).sum(1, keepdim=True)

    @staticmethod
    def attention_transfer(feature_map):
        return feature_map.pow(2).mean(1, keepdim=True)

    def forward(self, student_feature_map, teacher_feature_map):

        if self.mode == 'code':
            attention_student = self.attention_transfer(student_feature_map)
            attention_teacher = self.attention_transfer(teacher_feature_map)
        else:
            attention_student = self.attention_transfer_paper(
                student_feature_map)
            attention_teacher = self.attention_transfer_paper(
                teacher_feature_map)

        return attention_student, attention_teacher
