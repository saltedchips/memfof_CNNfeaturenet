import torch
import torch.nn as nn
import torch.nn.functional as F

# Create ConvBlock, down dim block, and CNNEncoder classes here
# Implement time speedups based on the backbone

# Based on approach from NeuFlow

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.alpha = 0.1
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = lambda x: F.leaky_relu(x, self.alpha, inplace=False)
        self.norm = torch.nn.BatchNorm2d(out_channels, eps=1e-06, affine=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        return self.norm(x1 + x2)


class DownDimBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownDimBlock, self).__init__()
        self.alpha = 0.1
        self.relu = lambda x: F.leaky_relu(x, self.alpha, inplace=False)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        return self.conv(self.relu(x))
    
class CNNEncoder(nn.Module):
    def __init__(self, dim):
        super(CNNEncoder, self).__init__()
        
        # evaluate 1/1, 1/2, 1/4, 1/8, 1/16
        
        self.block1_1 = ConvBlock(3, dim, kernel_size=8, stride=8, padding=0)  # 1/1
        
        self.block1_2 = ConvBlock(3, dim, kernel_size=8, stride=4, padding=2)  # 1/2
        
        self.block1_3 = ConvBlock(3, dim, kernel_size=8, stride=2, padding=3)  # 1/4
        
        self.block1_4 = ConvBlock(3, dim, kernel_size=7, stride=1, padding=3)  # 1/8
        
        self.block1_dd = DownDimBlock(dim*4, dim)  # pick features
        self.block1_ds = ConvBlock(dim, dim, kernel_size=2, stride=2, padding=0) 
        
        self.block2 = ConvBlock(3, dim, kernel_size=5, stride=1, padding=3)  # 1/16
        self.block2_dd = DownDimBlock(dim*2, dim)
        
        # 1/32 
        self.block3 = ConvBlock(3, dim, kernel_size=5, stride=1, padding=2)
        self.block3_dd = DownDimBlock(dim*2, dim//2)
    
    def init_pos(self, batch_size, height, width):
        ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        ys = ys.cuda() / (height-1)
        xs = xs.cuda() / (width-1)
        pos = torch.stack([ys, xs])
        return pos[None].repeat(batch_size,1,1,1)
    
    def init_pos_12(self, batch_size, height, width):
        self.pos_1 = self.init_pos(batch_size, height, width)
        self.pos_2 = self.init_pos(batch_size, height//2, width//2)
        self.pos_3 = self.init_pos(batch_size, height//4, width//4)
        
    def forward(self, x):
        b = x.shape[0]
        
        x1_1 = self.block1_1(x)  # 1/1
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x1_2 = self.block1_2(x)  # 1/2
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x1_3 = self.block1_3(x)  # 1/4
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x1_4 = self.block1_4(x)  # 1/8
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x1 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        x1 = self.block1_dd(x1)
        
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x2 = self.block2(x)  # 1/16
        x1 = self.block1_ds(x1)
        
        x1_ds_resized = F.adaptive_avg_pool2d(x1, x2.shape[2:])
        x2 = torch.cat([x1_ds_resized, x2], dim=1)
        x2 = self.block2_dd(x2)
        
        x3 = self.block3(x)
        x2_dd_resized = F.adaptive_avg_pool2d(x2, x3.shape[2:])
        x3 = torch.cat([x2_dd_resized, x3], dim=1)
        x3 = self.block3_dd(x3)

        
 
        x1 = torch.cat([x1, F.adaptive_avg_pool2d(self.pos_1, x1.shape[2:])], dim=1)
        x2 = torch.cat([x2, F.adaptive_avg_pool2d(self.pos_2, x2.shape[2:])], dim=1)
        x3 = torch.cat([x3, F.adaptive_avg_pool2d(self.pos_3, x3.shape[2:])], dim=1)

    

        
        
        return x1, x2
