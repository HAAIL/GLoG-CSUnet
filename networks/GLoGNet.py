import torch
import torch.nn as nn

from networks.Filter import GaborConv2D,LoGConv2D


class GLoGNet(nn.Module):
    def __init__(self, base_model, with_gabor=False,n_gabor_filters=2,gabor_filter_kernel_size=3,
                 with_LoG=False, n_LoG_filters=2 ,LoG_filter_kernel_size=3):
        super(GLoGNet, self).__init__()
        self.base_model = base_model
        self.with_gabor = with_gabor
        self.with_LoG = with_LoG
        if self.with_gabor:
            self.gaborConv2D = GaborConv2D(1, n_gabor_filters, kernel_size=gabor_filter_kernel_size)

        if self.with_LoG:
            self.LoGConv2D = LoGConv2D(1, n_LoG_filters, kernel_size=LoG_filter_kernel_size)


    def forward(self, x):
        input_x=x.clone()
        if self.with_gabor:
            x = torch.cat([x, (self.gaborConv2D(input_x))], dim=1)
        if self.with_LoG:
            x = torch.cat([x, (self.LoGConv2D(input_x))], dim=1)
        x = self.base_model(x)


        return x
