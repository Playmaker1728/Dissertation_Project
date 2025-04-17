import math
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from road_segmentation_YOLOP import LightweightRoadSegmentationNet

##############################################################设置超参数##############################################################
batch_size = 4
num_epochs = 5

##############################################################设置数据集##############################################################
#数据集路径设置；先搞个小点的数据集，比较完整的那种
def dir(mode = 'train'):
    image_dir = r"E:\BDD100K\datasets\images\{}".format(mode)     #图像数据集
    drivable_dir = r"E:\BDD100K\datasets\da_seg_annotations\{}".format(mode)   #可行驶区域标签图片
    lane_dir = r"E:\BDD100K\datasets\ll_seg_annotations\{}".format(mode)   #车道线标签图片
    return image_dir,drivable_dir,lane_dir

#定义数据集类
class BDD100KDataseet(Dataset):
    def __init__(self,image_dir,drivable_dir,lane_dir,transform=None):
        self.image_dir = image_dir
        self.drivable_dir = drivable_dir
        self.lane_dir = lane_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,idx):
        #获取图像文件的路径
        image_path = os.path.join(self.image_dir,self.image_files[idx])
        drivable_path = os.path.join(self.drivable_dir,self.image_files[idx].replace('.jpg','.png'))
        lane_path = os.path.join(self.lane_dir,self.image_files[idx].replace('.jpg','.png'))

        #读取图像和标注
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB) #转换为RGB
        drivable_gray = cv2.imread(drivable_path,cv2.IMREAD_GRAYSCALE)   #转为灰度图
        lane_gray = cv2.imread(lane_path,cv2.IMREAD_GRAYSCALE)

        #转为二值图像
        drivable_binary = (drivable_gray > 0).astype(np.float32)
        lane_binary = (lane_gray > 0).astype(np.float32)

        #转换为Tensor形式
        image = transforms.ToTensor()(image)    #(3,720,1280)
        drivable_binary = torch.tensor(drivable_binary,dtype=torch.float32).unsqueeze(0)  #增加通道维度(1,720,1280)
        lane_binary = torch.tensor(lane_binary,dtype=torch.float32).unsqueeze(0)

        #进行数据增强
        if self.transform:
            image = self.transform(image)
        
        return image,drivable_binary,lane_binary

#创建数据集加载器
# 训练集和batch数据
image_dir, drivable_dir, lane_dir = dir(mode='train')
train_dataset = BDD100KDataseet(image_dir, drivable_dir, lane_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True) #shuffle表示打乱
# 验证集和batch数据
image_dir, drivable_dir, lane_dir = dir(mode='val')
val_dataset = BDD100KDataseet(image_dir, drivable_dir, lane_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

##############################################################损失函数##############################################################
#简单的交叉熵损失函数定义：
criterion = nn.BCEWithLogitsLoss()
#iou损失函数
def iou_loss(lane_output,lane_gt,smooth=1e-6):
   pred_probs = torch.sigmoid(lane_output)  #将预测值通过Sigmoid转换为概率
   lane_gt = lane_gt.to(pred_probs.device).float()  #标签是浮点型并且设备一致
   #计算交集和并集
   intersection = (pred_probs*lane_gt).sum(dim=(1,2,3))
   union = pred_probs.sum(dim=(1,2,3))+lane_gt.sum(dim=(1,2,3))-intersection
   #计算iou
   iou = (intersection+smooth)/(union+smooth)
   return 1-iou.mean()

#总的损失函数
def compute_loss(drivable_output,lane_output,drivable_gt,lane_gt,alpha,beta):
    #可行驶区域部分：交叉熵损失函数
    drivable_loss = criterion(drivable_output, drivable_gt)
    #车道线部分：交叉熵损失函数与iou损失函数的和
    lane_loss = criterion(lane_output, lane_gt) + iou_loss(lane_output,lane_gt)
    #返回总的损失值
    return alpha*drivable_loss + beta*lane_loss

##############################################################创建模型##############################################################
# 模型实例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightweightRoadSegmentationNet().to(device)

#优化器设置
# optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
optimizer = torch.optim.Adam(
    filter(lambda p:p.requires_grad,model.parameters()),
    lr = 0.001,
    betas = (0.9,0.999)
)

# 学习率调整的函数
lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - 0.2) + 0.2

# 创建学习率调度器(余弦退火)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=-1)
##############################################################训练和验证##############################################################
for epoch in range(num_epochs):
    model.train()
    for idx,(image,drivable_gt,lane_gt) in enumerate(train_loader):
        image = image.to(device)
        drivable_gt = drivable_gt.to(device)
        lane_gt = lane_gt.to(device)

        #前向传播
        drivable_output,lane_output = model(image)
        #计算损失
        loss = compute_loss(drivable_output,lane_output,drivable_gt,lane_gt,alpha = 1.0,beta = 1.0)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        #每个epoch之后更新学习率
        lr_scheduler.step()

        #输出每个epoch的平均损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.data:.4f}')