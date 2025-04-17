from .AutoDriveDataset import AutoDriveDataset
from tqdm import tqdm


class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train: bool, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg
        
    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){
                    'image':image_path,
                    'mask':mask_path,
                    'lane':lane_path
                    }
        """
        print('building database...')
        gt_db = []
        for mask in tqdm(self.mask_list):
            mask_path = str(mask)
            image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
            lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
        
            rec = [{
                'image': image_path,
                'mask': mask_path,
                'lane': lane_path
            }]
            gt_db += rec
        
        print('database build finish')
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """
        """
        pass

