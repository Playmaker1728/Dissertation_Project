import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path
from ..utils import letterbox, augment_hsv, random_perspective

class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """   
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET # 'train'
        else:
            indicator = cfg.DATASET.TEST_SET # 'val'
        self.img_root: Path = img_root / indicator
        self.mask_root: Path = mask_root / indicator
        self.lane_root: Path = lane_root / indicator
        self.mask_list = list(self.mask_root.iterdir())
        self.db = [] # 数据库列表 database[]
        self.data_format = cfg.DATASET.DATA_FORMAT # 'jpg'
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError
     
    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError

    def __len__(self):
        """
        number of objects in the dataset
        """
        return len(self.db)
    
    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':image_path,'mask':mask_path,'lane':lane_path}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """      
        data = self.db[idx]        
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.cfg.num_seg_class == 3:
            seg_label = cv2.imread(data["mask"])
        else:
            seg_label = cv2.imread(data["mask"], 0) # 读取为灰度图
        lane_label = cv2.imread(data["lane"], 0)
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2] #(height, width, channels)提取前两个元素
        r = resized_shape / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
           
        (img, seg_label, lane_label), pad = letterbox((img, seg_label, lane_label), resized_shape, auto=True, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        
        if self.is_train: # 训练模式下对图像进行随机几何变换
            (img, seg_label, lane_label) = random_perspective(
                combination=(img, seg_label, lane_label),
                degrees=self.cfg.DATASET.ROT_FACTOR, # 随机旋转角度范围
                translate=self.cfg.DATASET.TRANSLATE, # 平移比例范围
                scale=self.cfg.DATASET.SCALE_FACTOR, # 图像的缩放比例
                shear=self.cfg.DATASET.SHEAR # 剪切角度范围
            )
        
            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)

            lr_flip = True # 水平翻转
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                seg_label = np.fliplr(seg_label)
                lane_label = np.fliplr(lane_label)

            ud_flip = False # 上下翻转
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                seg_label = np.flipud(seg_label)
                lane_label = np.flipud(lane_label)
        
        img = np.ascontiguousarray(img) # 返回内存连续的数组
        if self.cfg.num_seg_class == 3:
            _,seg0 = cv2.threshold(seg_label[:,:,0],128,255,cv2.THRESH_BINARY)
            _,seg1 = cv2.threshold(seg_label[:,:,1],1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        else:
            _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY) #大于1设为255（白色）小于1设为0（黑色）
            _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV) #大于1设为0（黑色）小于1设为255（白色）
        _,lane1 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY)
        _,lane2 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY_INV)

        if self.cfg.num_seg_class == 3:
            seg0 = self.Tensor(seg0) # 转为Pytorch张量，并且归一化，然后格式改为(C, H, W)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)

        if self.cfg.num_seg_class == 3:
            seg_label = torch.stack((seg0[0], seg1[0], seg2[0]),0)
        else:
            seg_label = torch.stack((seg2[0], seg1[0]),0)
        lane_label = torch.stack((lane2[0], lane1[0]),0)
        
        target = [seg_label, lane_label]
        img = self.transform(img)
        return img,target,data["image"], shapes

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected
    
    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes = zip(*batch)
        label_seg, label_lane = [], []
        for i, l in enumerate(label):
            l_seg, l_lane = l
            label_seg.append(l_seg)
            label_lane.append(l_lane)
        return torch.stack(img,0),[torch.stack(label_seg,0),torch.stack(label_lane,0)],paths,shapes    
        
        
            

            

        
