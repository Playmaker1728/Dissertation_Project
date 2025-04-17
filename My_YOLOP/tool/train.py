import argparse
import os,sys
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from tensorboardX import SummaryWriter
from lib.utils import DataLoaderX
import lib.dataset as dataset

from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.utils import is_parallel
from lib.utils.utils import create_logger, select_device
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.models import get_net

"""
模型训练
"""
def parase_args():
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='Train Multitask Network')
    # 添加参数
    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument(
        '--logDir', help='log directory', type=str, default='runs/')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--prevModelDir', help='prev Model directory', type=str, default='')
    
    parser.add_argument(
        '--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument(
        '--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument(
        '--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument(
        '--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()
    
    return args


def main():
    args = parase_args()
    update_config(cfg, args)
    
    # Set DDP variables
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    rank = global_rank
    
    logger, final_output_dir,tensorboard_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank)
    
    if rank in [-1, 0]:
        logger.info(pprint.pformat(args))
        logger.info(cfg)
        
        writer_dict = {
            'writer': SummaryWriter(log_dir=tensorboard_log_dir),
            'train_global_steps': 0,
            'valib_global_steps': 0,
        }
    else:
        writer_dict = None
    
    #cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    print("begin to build up model...")

    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')
        
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init='env://')
        
    print("load model to device")
    model = get_net(cfg).to(device)
    criterion = get_loss(cfg, device)
    optimizer = get_optimizer(cfg, model)
    
    # load checkpoint model
    best_perf = 0.0
    best_model = False
    last_epoch = -1

    Encoder_para_idx = [str(i) for i in range(0, 17)]
    Da_Seg_Head_para_idx = [str(i) for i in range(17, 26)]
    Ll_Seg_Head_para_idx = [str(i) for i in range(26, 35)]

    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    if rank in [-1, 0]:
        checkpoint_file = os.path.join(
            os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), 'checkpoint.pth'
        )
        if os.path.exists(cfg.MODEL.TRAIN_CONTINUE):
            logger.info(f"=> loading model '{cfg.MODEL.TRAIN_CONTINUE}'")
            checkpoint = torch.load(cfg.MODEL.TRAIN_CONTINUE)
            begin_epoch = 0
            last_epoch = -1
            model.load_state_dict(checkpoint['state_dict'])
            logger.info(f"=> loaded checkpoint '{cfg.MODEL.TRAIN_CONTINUE}' (epoch {begin_epoch})")
            
        if os.path.exists(cfg.MODEL.PRETRAINED):
            logger.info(f"=> loading model '{cfg.MODEL.PRETRAINED}'")
            checkpoint = torch.load(cfg.MODEL.PRETRAINED)
            begin_epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(f"=> loaded checkpoint '{cfg.MODEL.PRETRAINED}' (epoch {checkpoint['epoch']})")

        if os.path.exists(cfg.MODEL.PRETRAINED_DRIVABLE):
            logger.info(f"=> loading model weight in det branch from '{cfg.MODEL.PRETRAINED}'")
            drivable_idx_range = [str(i) for i in range(0, 26)]
            model_dict = model.state_dict()
            checkpoint_file = cfg.MODEL.PRETRAINED_DRIVABLE
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch']
            checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in drivable_idx_range}
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            logger.info(f"=> loaded det branch checkpoint '{checkpoint_file}'")

        if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
            logger.info(f"=> loading checkpoint '{checkpoint_file}'")
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(f"=> loaded checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']})")

        if cfg.TRAIN.SEG_ONLY:
            logger.info('freeze encoder...')
            for k,v in model.named_parameters():
                v.requires_grad = True
                if k.split(".")[1] in Encoder_para_idx:
                    print(f"freezing {k}")
                    v.requires_grad = False
        
        if cfg.TRAIN.ENC_DRIVABLE_ONLY:
            logger.info("freeze Lane head...")
            for k, v in model.named_parameters():
                v.requires_grad = True
                if k.split(".")[1] in Ll_Seg_Head_para_idx:
                    print(f"freezing {k}")
                    v.requires_grad = False
                    
        if cfg.TRAIN.ENC_LANE_ONLY:
            logger.info("freeze Drivable head...")
            for k, v in model.named_parameters():
                v.requires_grad = True
                if k.split(".")[1] in Da_Seg_Head_para_idx:
                    print(f"freezing {k}")
                    v.requires_grad = False
                
        if cfg.TRAIN.DRIVABLE_ONLY:
            logger.info("freeze encoder and Ll_Seg heads...")
            for k, v in model.named_parameters():
                v.requires_grad = True
                if k.split(".")[1] in Encoder_para_idx + Ll_Seg_Head_para_idx:
                    print(f"freezing {k}")
                    v.requires_grad = False
        
        if cfg.TRAIN.LANE_ONLY: 
            logger.info("freeze encoder and Da_Seg heads...")
            for k, v in model.named_parameters():
                v.requires_grad = True
                if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx:
                    print(f"freezing {k}")
                    v.requires_grad = False
                    
    if rank == -1 and torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
    if rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # assign model params
    model.gr = 1.0
    model.nc = 1

    print("begin to load data")
    # Data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg = cfg,
        is_train = True,
        inputsize = cfg.MODEL.IMAGE_SIZE,
        transform = transforms.Compose([transforms.ToTensor(), normalize,])
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank !=-1 else None
    train_loader = DataLoaderX(
        train_dataset, #获取样本的数据集
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS), #批次大小
        shuffle=(cfg.TRAIN.SHUFFLE & rank == -1), #是否打乱
        num_workers=cfg.WORKERS, # 加载数据时候的多线程数
        sampler=train_sampler, #指定数据采样器
        pin_memory=cfg.PIN_MEMORY, # 是否用GPU加速数据上传
        collate_fn=dataset.AutoDriveDataset.collate_fn # 从数据集中读取的样本批处理成一个批次的函数：变成更大的张量(batch_size, channel, height, width)
    )
    num_batch = len(train_loader)

    if rank in [-1, 0]:
        valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg = cfg,
            is_train = False,
            inputsize = cfg.MODEL.IMAGE_SIZE,
            transform = transforms.Compose([transforms.ToTensor(), normalize,])
        )
        valid_loader = DataLoaderX(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )            
        print('load data finished')    

    # training
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler('cuda', enabled=device.type != 'cpu')
    print("=> start training...")
    for epoch in range(begin_epoch+1, cfg.TRAIN.END_EPOCH+1):
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
        # train for one epoch 
        train(cfg, train_loader, model, criterion, optimizer, scaler,
              epoch, num_batch, num_warmup, writer_dict, logger, device, rank)
            
        lr_scheduler.step() # 更新学习率

        # evaluate on validation set
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH) and rank in [-1, 0]:
            da_segment_results, ll_segment_results, total_loss, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tensorboard_log_dir, writer_dict,
                logger, device, rank
            )
            
            # 使用 f-string 对每列进行格式化，确保每列对齐
            msg = f'Epoch: [{epoch}]      Loss: {total_loss:.3f}\n'\
                f'Driving area Segment:   Acc: {da_segment_results[0]:>6.3f}   IOU: {da_segment_results[1]:>6.3f}   mIOU: {da_segment_results[2]:>6.3f}\n'\
                f'Lane line Segment:      Acc: {ll_segment_results[0]:>6.3f}   IOU: {ll_segment_results[1]:>6.3f}   mIOU: {ll_segment_results[2]:>6.3f}\n'\
                f'Time:                   inference: {times[0]:>6.4f}s/frame'
            logger.info(msg)

        if rank in [-1, 0]:
            savepath = os.path.join(final_output_dir, f'epoch-{epoch}.pth')
            logger.info(f'=> saving checkpoint to {savepath}')
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                optimizer=optimizer,
                output_dir=final_output_dir,
                filename=f'epoch-{epoch}.pth'
            )
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                optimizer=optimizer,
                output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET),
                filename='checkpoint.pth'
            )

    # save final model
    if rank in [-1, 0]:
        final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info(f'=> saving final model state to {final_model_state_file}')
        model_state = model.module.state_dict() if is_parallel(model) else model.state_dict() # 获取模型的状态字典
        torch.save(model_state, final_model_state_file) # 将模型的状态字典保存到指定路径  
        writer_dict['writer'].close()
    else:
        dist.destroy_process_group() # 子进程，销毁进程组
    

if __name__ =='__main__':
    main()