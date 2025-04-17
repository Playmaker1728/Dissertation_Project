# from pathlib import Path
# import sys, os
# import pprint
# import torchvision.transforms as transforms

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(BASE_DIR)
# print(BASE_DIR)
# print(f"工作目录为：{os.getcwd()}")
# print(f"搜索路径为：{pprint.pformat(sys.path)}")

# from lib.config import cfg
# import lib.dataset as dataset

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# transforms = transforms.Compose([transforms.ToTensor(), normalize,])

# train_dataset = dataset.BddDataset(cfg, True, 640, transforms)
# print(train_dataset.__len__())
# print(train_dataset.db[0])
# train_dataset.__getitem__(0)












# img_root = Path(cfg.DATASET.DATAROOT) / cfg.DATASET.TRAIN_SET
# label_root = Path(cfg.DATASET.LABELROOT) / cfg.DATASET.TRAIN_SET 
# mask_root:Path = Path(cfg.DATASET.MASKROOT) / cfg.DATASET.TRAIN_SET
# lane_root = Path(cfg.DATASET.LANEROOT) / cfg.DATASET.TRAIN_SET

# mask1_list = list(mask_root.iterdir())
# print(mask1_list[1])
# mask_list = [file.name for file in mask_root.iterdir()]
# mask2_list = [mask_root / file_name for file_name in mask_list]
# print(mask2_list[1])

# import numpy as np
# a = np.mod(280,32)
# print(f'a = {a}')