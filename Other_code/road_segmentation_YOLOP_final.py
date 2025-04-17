import torch
import torch.nn as nn
from torchsummary import summary

"""
YOLOP网络的定义
"""
#####################################################定义需要使用的层#####################################################
class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,groups=1,act=True):
        super(Conv,self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,groups,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        try:
            self.act = nn.Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()
            
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))           

class Focus(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,groups=1,act=True):
        super(Focus,self).__init__()
        self.conv = Conv(in_channels*4,out_channels,kernel_size,stride,groups,act)
        
    def forward(self,x):
        return self.conv(torch.cat([x[..., ::2, ::2],x[..., 1::2, ::2],x[..., ::2, 1::2],x[..., 1::2, 1::2]], 1))

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,shortcut=True,groups=1,e=0.5):
        super(Bottleneck,self).__init__()
        mid_channels = int(out_channels*e)
        self.cv1 = Conv(in_channels,mid_channels,1,1)
        self.cv2 = Conv(mid_channels,out_channels,3,1)
        self.add = shortcut and in_channels == out_channels
        
    def forward(self,x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        
# 定义一个瓶颈结构并结合 CSP（Cross-Stage Partial Networks）结构，用于高效特征提取的网络模块。
class BottleNeckCSP(nn.Module):
    def __init__(self,in_channels,out_channels,num_Bottleneck=1,shortcut=True,groups=1,e=0.5):
        super(BottleNeckCSP,self).__init__()
        mid_channels = int(out_channels*e)
        #分支1
        self.cv1 = Conv(in_channels,mid_channels,1,1)
        self.m = nn.Sequential(*[Bottleneck(mid_channels, mid_channels, shortcut, groups, e=1.0) for _ in range(num_Bottleneck)])

        self.cv2 = nn.Conv2d(mid_channels,mid_channels,1,1,bias=False)
        #分支2
        self.cv3 = nn.Conv2d(in_channels,mid_channels,1,1,bias=False)
        #合并之后
        self.bn = nn.BatchNorm2d(mid_channels*2)
        self.act = nn.LeakyReLU(0.1,inplace=True)
        self.cv4 = Conv(mid_channels*2,out_channels,1,1)
    #前向传播函数
    def forward(self,x):
        y1 = self.cv2(self.m(self.cv1(x)))
        y2 = self.cv3(x)
        return self.cv4(self.act(self.bn(torch.cat((y1,y2),dim=1))))
    
class SimConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,groups=1,act=True):
        super(SimConv,self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,groups,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        try:
            self.act = nn.LeakyReLU() if act else nn.Identity()
        except:
            self.act = nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SimSPPF(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5):
        super(SimSPPF,self).__init__()
        mid_channels = in_channels // 2
        self.cv1 = SimConv(in_channels,mid_channels,1,1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size //2)
        self.cv2 = SimConv(mid_channels*4,out_channels,1,1)

    def forward(self,x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.concat((x,y1,y2,self.m(y2)),dim=1))
    
class UpSample(nn.Module):
    def __init__(self,size=None, scale_factor=None, mode='nearest'):
        super(UpSample,self).__init__()
        self.upsample = nn.Upsample(size=size,scale_factor=scale_factor,mode=mode)

    def forward(self,x):
        return self.upsample(x)

class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()    
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)

#####################################################轻量化路面分割网络#####################################################   
class LightweightRoadSegmentationNet(nn.Module):
    def __init__(self):
        super(LightweightRoadSegmentationNet, self).__init__()
        self.focus = Focus(3,32,3,1,1)
        self.encoder1 = nn.Sequential(
            Conv(32,64,kernel_size=3,stride=2),
            BottleNeckCSP(64,64,num_Bottleneck=1),
            Conv(64, 128, kernel_size=3, stride=2),
            BottleNeckCSP(128, 128,num_Bottleneck=3)
        )
        self.encoder2 = nn.Sequential(
            Conv(128, 256, kernel_size=3, stride=2),
            BottleNeckCSP(256, 256,num_Bottleneck=3)
        )
        self.encoder3 = nn.Sequential(
            Conv(256, 512, kernel_size=3, stride=2),
            SimSPPF(512,512),
            BottleNeckCSP(512,512,num_Bottleneck=1,shortcut=False),
            Conv(512,256,kernel_size=1,stride=1),
            UpSample(scale_factor=2, mode='nearest')  #特别注明，如果输入是(720,1280)，这里设置size = (45,80)
        )
        self.encoder4 = nn.Sequential(
            BottleNeckCSP(512, 256,num_Bottleneck=1,shortcut=False),#拼接后输入通道数为256+256=512
            Conv(256, 128, kernel_size=1, stride=1),
            UpSample(None,scale_factor=2, mode='nearest')
        )
        #可行驶区域分支
        self.drivable_branch = nn.Sequential(
            Conv(256,128,kernel_size=3,stride=1),
            UpSample(None,scale_factor=2, mode='nearest'),
            BottleNeckCSP(128, 64,num_Bottleneck=1,shortcut=False),
            Conv(64, 32, kernel_size=3, stride=1),
            UpSample(None,scale_factor=2, mode='nearest'),
            Conv(32, 16, kernel_size=3, stride=1),
            BottleNeckCSP(16, 8,num_Bottleneck=1,shortcut=False),
            UpSample(None,scale_factor=2, mode='nearest'),
            Conv(8, 2, kernel_size=3, stride=1)     #(8,2,3,1)
        )
        #车道线区域分支
        self.lane_branch = nn.Sequential(
            Conv(256,128,kernel_size=3,stride=1),
            UpSample(None,scale_factor=2, mode='nearest'),
            BottleNeckCSP(128, 64,num_Bottleneck=1,shortcut=False),
            Conv(64, 32, kernel_size=3, stride=1),
            UpSample(None,scale_factor=2, mode='nearest'),
            Conv(32, 16, kernel_size=3, stride=1),
            BottleNeckCSP(16, 8,num_Bottleneck=1,shortcut=False),
            UpSample(None,scale_factor=2, mode='nearest'),
            Conv(8, 2, kernel_size=3, stride=1)     #(8,2,3,1)
        )
        
    def forward(self,x):
        y1 = self.encoder1(self.focus(x))
        y2 = self.encoder2(y1)
        y3 = self.encoder3(y2)
        y4 = self.encoder4(torch.cat((y2,y3),dim=1))
        y5 = torch.cat((y1,y4),dim=1)
        #路面识别分支
        drivable_output = self.drivable_branch(y5)
        #车道线分支
        lane_output = self.lane_branch(y5)
        return drivable_output,lane_output
        
#####################################################示例用法#####################################################
if __name__ == "__main__":
    # 检查是否有可用的GPU：
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建网络实例
    torch.cuda.empty_cache()
    net = LightweightRoadSegmentationNet().to(device)
    input_tensor = torch.randn(1,3,384,640).to(device)
    # 前向传播
    drivable_output, lane_output = net(input_tensor)
    print("输入张量形状:", input_tensor.shape)
    print("车道线检测输出形状:", lane_output.shape)
    print("路面识别输出形状:", drivable_output.shape)
    for name, para in net.named_parameters():
        print(f"Name: {name}")
    # summary(net, input_size=(3, 384, 640))
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as profile:
    #     model_out = net(input_tensor)
    # print(profile)