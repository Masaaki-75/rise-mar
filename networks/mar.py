import torch
import torch.nn as nn
import sys
sys.path.append('..')
from modules.basic_layers import get_conv_block
from modules.basic_blocks import CBAM


class RiseMARNet(nn.Module):
    """
    A re-implementation of the MAR network described in "Unsupervised CT Metal Artifact 
    Learning using Attention-guided $beta$-CycleGAN (TMI, 2021)"
    """
    def __init__(self, in_channels, out_channels, base_channels=64, norm_type='INSTANCE', act_type='RELU'):
        super().__init__()
        num_channels = [base_channels * 2 ** i for i in range(4)]
        self.num_channels = num_channels
        self.in_channels = in_channels
        block_kwargs = dict(kernel_size=3, stride=1, padding=1, adn_order='CNA', norm_type=norm_type, act_type=act_type)
        
        self.in_conv = nn.Sequential(
            get_conv_block(in_channels, num_channels[0], **block_kwargs),
            get_conv_block(num_channels[0], num_channels[0], **block_kwargs),)
        
        self.down_block1 = nn.Sequential(
            get_conv_block(num_channels[0], num_channels[0], **block_kwargs),
            get_conv_block(num_channels[0], num_channels[0], **block_kwargs),)
        self.skip1 = CBAM(num_channels[0])
        self.down1 = nn.MaxPool2d(2)
        
        self.down_block2 = nn.Sequential(
            get_conv_block(num_channels[0], num_channels[1], **block_kwargs),
            get_conv_block(num_channels[1], num_channels[1], **block_kwargs),)
        self.skip2 = CBAM(num_channels[1])
        self.down2 = nn.MaxPool2d(2)
        
        self.down_block3 = nn.Sequential(
            get_conv_block(num_channels[1], num_channels[2], **block_kwargs),
            get_conv_block(num_channels[2], num_channels[2], **block_kwargs),)
        self.skip3 = CBAM(num_channels[2])
        self.down3 = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(
            get_conv_block(num_channels[2], num_channels[3], **block_kwargs),
            get_conv_block(num_channels[3], num_channels[3], **block_kwargs),)
        
        self.up3 = nn.ConvTranspose2d(num_channels[3], num_channels[2], kernel_size=2, stride=2)
        self.up_block3 = nn.Sequential(
            get_conv_block(num_channels[3], num_channels[2], **block_kwargs),
            get_conv_block(num_channels[2], num_channels[2], **block_kwargs),)
        
        self.up2 = nn.ConvTranspose2d(num_channels[2], num_channels[1], kernel_size=2, stride=2)
        self.up_block2 = nn.Sequential(
            get_conv_block(num_channels[2], num_channels[1], **block_kwargs),
            get_conv_block(num_channels[1], num_channels[1], **block_kwargs),)
        
        self.up1 = nn.ConvTranspose2d(num_channels[1], num_channels[0], kernel_size=2, stride=2)
        self.up_block1 = nn.Sequential(
            get_conv_block(num_channels[1], num_channels[0], **block_kwargs),
            get_conv_block(num_channels[0], num_channels[0], **block_kwargs),)
        
        self.out_conv = nn.Conv2d(num_channels[0], out_channels, kernel_size=1)
        
    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat((x, y), dim=1) if self.in_channels > 1 else y
            
        d1 = self.in_conv(x)
        d1 = self.down_block1(d1)  # [B, 64, H, W]
        s1 = self.skip1(d1)  # [B, 64, H, W]
        
        d2 = self.down1(d1)  # [B, 64, H/2, W/2]
        d2 = self.down_block2(d2)  # [B, 128, H/2, W/2]
        s2 = self.skip2(d2)  # [B, 128, H/2, W/2]
        
        d3 = self.down2(d2)  # [B, 128, H/4, W/4]
        d3 = self.down_block3(d3) # [B, 256, H/4, W/4]
        s3 = self.skip3(d3) # [B, 256, H/4, W/4]
        
        d4 = self.down3(d3)  # [B, 256, H/8, W/8]
        d4 = self.bottleneck(d4)  # [B, 256, H/4, W/4]
        
        u = self.up3(d4)
        u = torch.cat([s3, u], dim=1)
        u = self.up_block3(u)
        
        u = self.up2(u)  # [B, 128, H/2, W/2]
        u = torch.cat([s2, u], dim=1)  # [B, 256, H/2, W/2]
        u = self.up_block2(u)  # [B, 128, H/2, W/2]
        
        u = self.up1(u)  # [B, 64, H, W]
        u = torch.cat([s1, u], dim=1)  # [B, 128, H, W]
        u = self.up_block1(u)  # [B, 64, H, W]
        u = self.out_conv(u)  # [B, 1, H, W]
        
        return u
    
        
     
    
if __name__ == '__main__':
    g = RiseMARNet(1, 1, 64, norm_type='INSTANCE')
    x = torch.randn((2, 1, 256, 256))
    y1 = g(x)
    print(y1.shape)
    
    