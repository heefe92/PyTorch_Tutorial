import numpy as np

import torch.nn as nn
import torch as t
from torch.nn import functional as F
from config import opt




class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self,output, label):
        pass