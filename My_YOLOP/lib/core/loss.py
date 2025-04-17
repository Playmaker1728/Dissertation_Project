import torch
import torch.nn as nn
from lib.core.evaluate import SegmentationMetric


class MultiHeadLoss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [da_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(3)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = losses
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]
        """   
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)
        return total_loss, head_losses
    
    def _forward_impl(self, predictions, targets, shapes, model):
        """
        Args:
            predictions: predicts of [[det_head1, det_head2, det_head3], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses
        """
        cfg = self.cfg
        BCEseg = self.losses

        drive_area_seg_predicts = predictions[0].view(-1) #.view(-1) 展平成一维
        drive_area_seg_targets = targets[0].view(-1)
        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)

        lane_line_seg_predicts = predictions[1].view(-1)
        lane_line_seg_targets = targets[1].view(-1)
        lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)
       
        # 除边界的填充区域padding
        nb, _, height, width = targets[0].shape #(batch_size, channels, height, width)
        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        _, lane_line_pred = torch.max(predictions[1], 1)
        _, lane_line_gt = torch.max(targets[1], 1)
        lane_line_pred = lane_line_pred[:,pad_h:height-pad_h, pad_w:width-pad_w]
        lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
        # 获取车道线的IoU损失
        metric = SegmentationMetric(2) 
        metric.reset()
        metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
        IoU = metric.IntersectionOverUnion()
        liou_ll = 1 - IoU
        
        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[0]
        lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[1]
        liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[2]

        if cfg.TRAIN.DRIVABLE_ONLY or cfg.TRAIN.ENC_DRIVABLE_ONLY:
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll            

        if cfg.TRAIN.LANE_ONLY or cfg.TRAIN.ENC_LANE_ONLY:  
            lseg_da = 0 * lseg_da

        loss = lseg_da + lseg_ll + liou_ll
        return loss,(lseg_da.item(), lseg_ll.item(), liou_ll.item(), loss.item())

      
def get_loss(cfg, device):
    """
    get MultiHeadLoss
    
    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)
    -device: cpu or gpu device

    Returns:
    -loss: (MultiHeadLoss)
    """
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    losses = BCEseg
    loss = MultiHeadLoss(losses, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss    
