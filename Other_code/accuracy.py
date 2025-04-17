import torch

#计算像素级准确率函数
def compute_pixel_accuracy(pred,target):
    pred = torch.sigmoid(pred)>0.5
    return (pred == target).float().mean().item()
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pth',map_location=device)
model.eval()

def accuracy(val_loader):
    drivable_accuracy_list = []
    lane_accuracy_list = []
    with torch.no_grad():
        for images,drivable_gt,lane_gt in val_loader:
            images = images.to(device)
            drivable_gt = drivable_gt.to(device)
            lane_gt = lane_gt.to(device)
            
            drivable_pred,lane_pred = model(images)
            drivable_accuracy.append(compute_pixel_accuracy(drivable_pred,drivable_gt))
            lane_accuracy.append(compute_pixel_accuracy(lane_pred,lane_gt))
            
        drivable_accuracy = sum(drivable_accuracy_list)/len(val_loader)
        lane_accuracy = sum(lane_accuracy_list)/len(val_loader)
        return drivable_accuracy,lane_accuracy