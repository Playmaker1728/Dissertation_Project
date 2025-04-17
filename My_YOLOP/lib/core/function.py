import cv2
import os
import time
import torch
import torch.amp as amp
import math
import numpy as np
from lib.core.evaluate import SegmentationMetric
from lib.core.general import check_img_size
from lib.utils import show_seg_result
from lib.utils.utils import time_synchronized
from tqdm import tqdm

def train(cfg, train_loader, model, criterion, optimizer, scaler, 
          epoch, num_batch, num_warmup, writer_dict, logger, device, rank=-1):
    """
    train for one epoch

    Inputs:
    - cfg: configurations
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return total_loss, head_losses
    - writer_dict:
    outputs(2,)
    output[0] len:1, [2,256,256]
    output[1] len:1, [2,256,256]
    target(2,)
    target[0] [2,256,256]
    target[1] [2,256,256]
    Returns:
    None

    """     
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    for i, (input, target, paths, shapes) in enumerate(train_loader):
        num_iter = i + num_batch * (epoch - 1) # 计算当前迭代次数
        if num_iter < num_warmup:              # warm up 预热
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
            xi = [0, num_warmup]
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])
        
        data_time.updata(time.time() - start)
        if not cfg.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
        with amp.autocast('cuda', enabled=device.type != 'cpu'):
            outputs = model(input)
            total_loss, head_losses = criterion(outputs, target, shapes, model)
        
        # compute gradient and do update step
        optimizer.zero_grad()   
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if rank in [-1, 0]:
            # measure accuracy and record loss
            losses.updata(total_loss.item(), input.size(0)) 
            # measure elapsed time
            batch_time.updata(time.time() - start)
            if i % cfg.PRINT_FREQ == 0:
                msg = f'Epoch: [{epoch}][{i}]/[{len(train_loader)}]\t'\
                      f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'\
                      f'Speed {input.size(0)/batch_time.val:.1f} samples/s\t'\
                      f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'\
                      f'Loss {losses.val:.5f} ({losses.avg:.5f})'
                logger.info(msg)

                #画图，横坐标：losses.val 纵坐标：global_steps
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def validate(epoch, cfg, val_loader, val_dataset, model, criterion,output_dir,
             tb_log_dir, writer_dict=None, logger=None, device='cpu', rank=-1):
    """
    validata
    
    Inputs:
    - cfg: configurations
    - train_loader: loader for data
    - model:
    - criterion: (function) calculate all the loss, return
    - writer_dict:

    Return:
    None
    """
    # setting
    max_stride = 32
    save_dir = output_dir + os.path.sep + 'visualization'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    test_batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    
    da_metric = SegmentationMetric(cfg.num_seg_class) #segment confusion matrix   
    ll_metric = SegmentationMetric(2) #segment confusion matrix   

    losses = AverageMeter()
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    T_inf = AverageMeter() # 推理时间

    # switch to train mode
    model.eval()
    
    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        if not cfg.DEBUG:
            img = img.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            _, _, height, width = img.shape #batch size, channel, height, width
        
        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[0][1][0][0]
            
            t = time_synchronized() #等GPU所有计算执行完
            da_seg_out, ll_seg_out = model(img) # [batch_size, 2, h, w]
            t_inf = time_synchronized() - t # 推理时间
            if batch_i > 0:
                T_inf.updata(t_inf/img.size(0),img.size(0)) # img.size(0) batch_size,图片数
            
            # driving area segment evaluation
            _,da_predict = torch.max(da_seg_out, dim=1) # max对应的索引
            _,da_gt = torch.max(target[0], dim=1)
            da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w] # 去掉填充
            da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            da_metric.reset()
            da_metric.addBatch(da_predict.cpu(),da_gt.cpu())
            da_acc = da_metric.pixelAccuracy()
            da_IoU = da_metric.IntersectionOverUnion()
            da_mIoU = da_metric.meanIntersectionOverUnion()

            da_acc_seg.updata(da_acc, img.size(0)) # (acc, batch_size)
            da_IoU_seg.updata(da_IoU, img.size(0)) # (iou, batch_size)
            da_mIoU_seg.updata(da_mIoU, img.size(0)) # (miou, batch_size)
                    
            # lane line segment evaluation
            _,ll_predict = torch.max(ll_seg_out, dim=1)
            _,ll_gt = torch.max(target[1], dim=1) 
            ll_predict = ll_predict[:,pad_h:height-pad_h,pad_w:width-pad_w]
            ll_gt = ll_gt[:,pad_h:height-pad_h,pad_w:width-pad_w]

            ll_metric.reset()
            ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
            ll_acc = ll_metric.lineAccuracy()
            ll_IoU = ll_metric.IntersectionOverUnion()
            ll_mIoU = ll_metric.meanIntersectionOverUnion()

            ll_acc_seg.updata(ll_acc, img.size(0))
            ll_IoU_seg.updata(ll_IoU, img.size(0))
            ll_mIoU_seg.updata(ll_mIoU, img.size(0))

            total_loss, head_losses = criterion((da_seg_out, ll_seg_out),target, shapes, model)
            losses.updata(total_loss.item(), img.size(0))

            if cfg.TEST.PLOTS:
                if batch_i == 0:
                    for i in range(test_batch_size):
                        img_test = cv2.imread(paths[i])
                        da_seg_mask = da_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        da_seg_mask = torch.nn.functional.interpolate(da_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, da_seg_mask = torch.max(da_seg_mask, dim=1) #返回的是个索引
                        
                        da_gt_mask = target[0][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        da_gt_mask = torch.nn.functional.interpolate(da_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, da_gt_mask = torch.max(da_gt_mask, dim=1)
                        
                        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
                        da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()

                        img_test1 = img_test.copy()
                        _ = show_seg_result(img_test, da_seg_mask, i, epoch, save_dir)
                        _ = show_seg_result(img_test1, da_gt_mask, i, epoch, save_dir, is_gt=True)

                        img_ll = cv2.imread(paths[i])
                        ll_seg_mask = ll_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, ll_seg_mask = torch.max(ll_seg_mask, dim=1)

                        ll_gt_mask = target[1][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_gt_mask = torch.nn.functional.interpolate(ll_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _,ll_gt_mask = torch.max(ll_gt_mask, dim=1)

                        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
                        ll_gt_mask = ll_gt_mask.int().squeeze().cpu().numpy()

                        img_ll1 = img_ll.copy()
                        _ = show_seg_result(img_ll, ll_seg_mask, i, epoch, save_dir, is_ll=True)
                        _ = show_seg_result(img_ll1, ll_gt_mask, i, epoch, save_dir, is_ll=True, is_gt=True)
        
    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)
    t = [T_inf.avg]
    return da_segment_result, ll_segment_result, losses.avg, t
        
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def updata(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        