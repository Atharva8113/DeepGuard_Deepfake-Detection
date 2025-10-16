# xception.py

import torch
import torch.nn as nn
import pretrainedmodels

class FFPPXception(nn.Module):
    def __init__(self, num_classes=2):
        super(FFPPXception, self).__init__()
        self.model = pretrainedmodels.__dict__['xception'](pretrained='imagenet')
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
