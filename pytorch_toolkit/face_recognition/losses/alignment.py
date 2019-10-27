"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import math
import torch
import torch.nn as nn

VALID_CORE_FUNC_TYPES = ['l1', 'l2', 'wing']


def wing_core(abs_x, w, eps):
    """Calculates the wing function from https://arxiv.org/pdf/1711.06753.pdf"""
    return w*math.log(1. + abs_x / eps)

class AlignmentLoss(nn.Module):
    """Regression loss to train landmarks model"""
    def __init__(self, loss_type='l2'):
        super(AlignmentLoss, self).__init__()
        assert loss_type in VALID_CORE_FUNC_TYPES
        self.uniform_weights = True
        self.weights = None
        self.core_func_type = loss_type
        self.eps = 0.031       
            loss = torch.sum(loss, 2)
            eyes_dist = (torch.norm(target[:, 0:2] - target[:, 18:20], p=1, dim=1).reshape(-1))

        if self.uniform_weights:
            loss = torch.sum(loss, 1)
            loss /= n_points
        else:
            assert self.weights.shape[0] == loss.shape[1]
            loss = torch.mul(loss, self.weights)
            loss = torch.sum(loss, 1)

        loss = torch.div(loss, eyes_dist)
        loss = torch.sum(loss)
        return loss / (2.*bs)
