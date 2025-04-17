import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(f"BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from lib.utils import initialize_weights
from lib.core.evaluate import SegmentationMetric

"""
YOLOP网络的定义
"""
#####################################################定义需要使用的层#####################################################
class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=None,groups=1,act=True):
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
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=None,groups=1,act=True):
        super(Focus,self).__init__()
        self.conv = Conv(in_channels*4,out_channels,kernel_size,stride,padding,groups,act)
        
    def forward(self,x):
        return self.conv(torch.cat([x[..., ::2, ::2],x[..., 1::2, ::2],x[..., ::2, 1::2],x[..., 1::2, 1::2]], 1))

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,shortcut=True,groups=1,e=0.5):
        super(Bottleneck,self).__init__()
        mid_channels = int(out_channels*e)
        self.cv1 = Conv(in_channels,mid_channels,1,1)
        self.cv2 = Conv(mid_channels,out_channels,3,1,groups=groups)
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
    
#####################################################YOLOP网络结构#####################################################
"""
YOLOP: (list) [继承的层数, 模块名称, [输入参数:(in_channels, out_channels, kernel_size, stride)]]
"""
YOLOP = [
[25, 34], #Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],  #0
[ -1, Conv, [32, 64, 3, 2]],  #1
[ -1, BottleNeckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],  #3
[ -1, BottleNeckCSP, [128, 128, 3]],  #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleNeckCSP, [256, 256, 3]],  #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SimSPPF, [512, 512, 5]],  #8
[ -1, BottleNeckCSP, [512, 512, 1, False]],  #9
[ -1, Conv, [512, 256, 1, 1]],  #10
[ -1, UpSample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],  #12
[ -1, BottleNeckCSP, [512, 256, 1, False]],  #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, UpSample, [None, 2, 'nearest']],  #15
[ [-1, 4], Concat, [1]],  #16         #Encoder

[ 16, Conv, [256, 128, 3, 1]],  #17
[ -1, UpSample, [None, 2, 'nearest']],  #18
[ -1, BottleNeckCSP, [128, 64, 1, False]],  #19
[ -1, Conv, [64, 32, 3, 1]],  #20
[ -1, UpSample, [None, 2, 'nearest']],  #21
[ -1, Conv, [32, 16, 3, 1]],  #22
[ -1, BottleNeckCSP, [16, 8, 1, False]],  #23
[ -1, UpSample, [None, 2, 'nearest']],  #24
[ -1, Conv, [8, 2, 3, 1]],  #25 Driving area segmentation head

[ 16, Conv, [256, 128, 3, 1]],  #26
[ -1, UpSample, [None, 2, 'nearest']],  #27
[ -1, BottleNeckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],  #29
[ -1, UpSample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],  #31
[ -1, BottleNeckCSP, [16, 8, 1, False]],  #32
[ -1, UpSample, [None, 2, 'nearest']],  #33
[ -1, Conv, [8, 2, 3, 1]],  #34 Lane line segmentation head
]

#####################################################构建轻量化路面分割网络#####################################################
class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        self.seg_out_idx = block_cfg[0]
        layers, save = [], []

        #Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block
            block_ = block(*args)  # 模块实例化并传入对应参数
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist [6, 4, 16, 16]
        
        # nn.Sequential会自动生成索引式的名称
        self.model, self.save = nn.Sequential(*layers), sorted(save)
        
        initialize_weights(self)

    def forward(self, x):
        cache = []  # 用来存储中间结果
        out = []  # 用来存储最终结果
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]
            x = block(x)
            cache.append(x if block.index in self.save else None) # 每层的输出结果缓存到cache中
            if i in self.seg_out_idx:
                m = nn.Sigmoid()
                out.append(m(x))
        return out


def get_net(cfg, **kwargs):
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model


#####################################################测试#####################################################
if __name__ == "__main__":
    from torchsummary import summary
    from torch.utils.tensorboard import SummaryWriter
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model = get_net(False).to(device)
    input_ = torch.randn(1, 3, 384, 640).to(device)
    gt_ = torch.rand(1, 2, 384, 640)
    metric = SegmentationMetric(2)
    dring_area_seg, lane_line_seg = model(input_)
    print(f"input_size:          {input_.shape}")
    print(f"dring_area_seg_size: {dring_area_seg.shape}")
    print(f"lane_line_seg_size:  {lane_line_seg.shape}")
    for name, param in model.named_parameters():
        print(f"Name: {name:<40}Size: {param.size()}")
    summary(model, (3, 384, 640))
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as profile:
    #     model_out = model(input_)
    # print(profile)

