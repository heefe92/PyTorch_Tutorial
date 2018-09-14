import torch as t
import torch.nn as nn
import numpy as np
from losses import MyLoss

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.training=False
        #self.Loss = MyLoss()
        pass

    def use_preset(self,isTraining):
        self.training=isTraining

    def forward(self, inputs):
        if self.training:
            img_batch, labels = inputs

        else:
            img_batch = inputs
        pass

        if self.training:
            pass
        else:
            pass
