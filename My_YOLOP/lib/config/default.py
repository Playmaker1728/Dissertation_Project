import os
from yacs.config import CfgNode as CN


"""
参数默认值
"""
_C = CN()
_C.LOG_DIR = 'runs/BddDataset'
_C.GPUS = [0]
_C.WORKERS = 8
_C.PIN_MEMORY = False
_C.PRINT_FREQ = 20
_C.AUTO_RESUME =True      #设置自动恢复训练
_C.DEBUG = False
_C.num_seg_class = 2


# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.PRETRAINED = ""
_C.MODEL.PRETRAINED_DRIVABLE = ""
_C.MODEL.TRAIN_CONTINUE = ""
_C.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256

# _C.MODEL.TRAIN_CONTINUE = r"E:\YOLOP\Road_Area_Recognition\YOLOP\runs\BddDataset\_2025-04-14-03-54\checkpoint.pth"

# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.DA_SEG_GAIN = 0.2  # driving area segmentation loss gain
_C.LOSS.LL_SEG_GAIN = 0.2  # lane line segmentation loss gain
_C.LOSS.LL_IOU_GAIN = 0.2 # lane line iou loss gain


# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = r"E:\BDD100K\datasets\images" # the path of images folder
_C.DATASET.LABELROOT = r"E:\BDD100K\datasets\det_annotations" # the path of det_annotations folder
_C.DATASET.MASKROOT = r"E:\BDD100K\datasets\da_seg_annotations" # the path of da_seg_annotations folder
_C.DATASET.LANEROOT = r"E:\BDD100K\datasets\ll_seg_annotations" # the path of ll_seg_annotations folder
_C.DATASET.DATASET = 'BddDataset'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.SELECT_DATA = False
_C.DATASET.ORG_IMG_SIZE = [720, 1280]

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # 色相 image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # 饱和度 image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # 亮度 image HSV-Value augmentation (fraction)


# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.LRF = 0.2  # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 3.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1 # 预热学习率
_C.TRAIN.WARMUP_MOMENTUM = 0.8
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 240
_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.SHUFFLE = True

# if training multiple tasks end-to-end, set all parameters as False
# Alternating optimization 交替训练选项,依次训练顺序, Da,Ll->E,Da->E,Ll->Da->Ll
_C.TRAIN.SEG_ONLY = True           # 只训练两个分割分支，冻结编码器
_C.TRAIN.ENC_DRIVABLE_ONLY = False       # 只训练编码器和可行驶区域分割分支
_C.TRAIN.ENC_LANE_ONLY = False       # 只训练编码器和车道线分割分支

# Single task 单一任务训练
_C.TRAIN.DRIVABLE_ONLY = False      # 只训练可行驶区域分割分支
_C.TRAIN.LANE_ONLY = False          # 只训练车道线分割分支


# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 24
_C.TEST.MODEL_FILE = ''
_C.TEST.PLOTS = True


def update_config(cfg,args):
    cfg.defrost()   #使cfg配置对象从冻结变为可修改
    
    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir
        
    if args.logDir:
        cfg.LOG_DIR = args.logDir
    
    cfg.freeze()    #设置配置对象为冻结
