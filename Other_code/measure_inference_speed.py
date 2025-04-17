import torch
import time
from road_segmentation_YOLOP import LightweightRoadSegmentationNet
"""
测试模型推理速度
"""
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型
# model = torch.load('model.pth',map_location=device)
# model.eval()
model = LightweightRoadSegmentationNet().to(device)
model.eval()
#创建随机输入张量
input_size = (3,320,1280)
input_tensor = torch.randn(1,*input_size).to(device)

# 模型预热
with torch.no_grad():   #没有梯度
    for _ in range(10):
        _ = model(input_tensor)

#测试推理时间
num_samples = 100
start_time = time.time()
with torch.no_grad():
    for _ in range(num_samples):
        _ = model(input_tensor)
end_time = time.time()

#计算平均推理时间
inference_time = (end_time - start_time)*1000/num_samples
print(f'推理速度为：{inference_time:.3f}毫秒')
FPS = 1000/inference_time
print(f'FPS为：{FPS:.1f}帧/秒')