import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

# 定义ConvBNLReLU类，实现卷积、批量归一化和LeakyReLU激活的组合模块
class ConvBNLReLU(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(ConvBNLReLU,self).__init__()#调用父类构造函数，super(ConvBNLReLU,self)表示需要调用函数的子类，.__init__()表示调用的是父类的初始化函数
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)#定义了一个卷积层
        self.bn = nn.BatchNorm2d(out_channels)#定义了一个批量归一层
        self.relu = nn.LeakyReLU(0.1)#定义LeakyReLU激活函数

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

# 定义一个瓶颈结构并结合 CSP（Cross-Stage Partial Networks）结构，用于高效特征提取的网络模块。
class BottleNeckCSP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(BottleNeckCSP,self).__init__()
        mid_channels = out_channels//2 #假设中间通道数是输出通道数的一半
        # 分支1
        self.branch1 = nn.Conv2d(in_channels,mid_channels,kernel_size=1,stride=1,padding=0)
        #分支2
        self.branch2 = nn.Sequential(
            ConvBNLReLU(in_channels,mid_channels),
            ConvBNLReLU(mid_channels,mid_channels),
            nn.Conv2d(mid_channels,mid_channels,kernel_size=1,stride=1,padding=0)
        )
        #融合和输出
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(mid_channels*2),#乘以2是因为合并两个分支导致通道数翻倍
            nn.LeakyReLU(0.1),
            ConvBNLReLU(mid_channels*2,out_channels)
        )

    #前向传播函数
    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse(x)
        return x
# 定义ConvBNReLU类，实现卷积、批量归一化和ReLU激活的组合模块
class ConvBNReLU(ConvBNLReLU):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        #调用父类构造函数，父类初始化要的参数这里也要写上
        super(ConvBNReLU,self).__init__(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU() #使用ReLU替换LeakyReLU激活函数

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class SimSPPF(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5):
        super(SimSPPF,self).__init__()
        mid_channels = in_channels//2#假设中间通道数是输入通道数的一半
        self.conv1 = ConvBNReLU(in_channels,mid_channels,kernel_size=kernel_size, stride=1,padding=kernel_size//2)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1,padding=kernel_size//2)
        self.conv2 = ConvBNReLU(mid_channels * 4, out_channels, kernel_size=1,stride=1)

    def forward(self,x):
        x = self.conv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x1 = self.pool(x)
            x2 = self.pool(x1)
            x3 = self.pool(x2)
        return self.conv2(torch.cat([x,x1,x2,x3],dim=1))
class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpSample,self).__init__()
        #反卷积操作：输出特征图的尺寸将是输入特征图的两倍。
        self.conv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)

    def forward(self,x):
        return self.conv(x)
class LightweightRoadSegmentationNet(nn.Module):
    def __init__(self):
        super(LightweightRoadSegmentationNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1),
            BottleNeckCSP(64,128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            BottleNeckCSP(128, 256)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            BottleNeckCSP(256, 512)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            BottleNeckCSP(512, 512)
        )
        self.encoder4 = nn.Sequential(
            SimSPPF(512, 512),
            BottleNeckCSP(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            UpSample(512, 256)
        )
        self.encoder5 = nn.Sequential(
            BottleNeckCSP(768, 256),#拼接后输入通道数为256+512=768
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            UpSample(256, 128)
        )
        self.lane_branch = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, stride=2, padding=1),
            UpSample(128, 128),
            BottleNeckCSP(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            UpSample(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BottleNeckCSP(128, 128),
            UpSample(128, 128),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        )
        self.surface_branch = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, stride=2, padding=1),
            UpSample(128, 128),
            BottleNeckCSP(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            UpSample(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BottleNeckCSP(128, 128),
            UpSample(128, 128),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        )
    def forward(self,x):
        #编码器部分
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = torch.cat([x4,x2],1)
        x6 = self.encoder5(x5)
        x7 = torch.cat([x6,x1],1)

        #车道线分支
        lane_output = self.lane_branch(x7)
        #路面识别分支
        surface_output = self.surface_branch(x7)

        return lane_output,surface_output
        # 示例用法
if __name__ == "__main__":
    #检查网络是否有可用的GPU：
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    # 创建网络实例
    net = LightweightRoadSegmentationNet().to(device)
input_tensor = torch.randn(8,3,1280,720).to(device)
# 前向传播
lane_output, surface_output = net(input_tensor)
print("输入张量形状:", input_tensor.shape)
print("车道线检测输出形状:", lane_output.shape)
print("路面识别输出形状:", surface_output.shape)