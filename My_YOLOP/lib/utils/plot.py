import cv2
import numpy as np


def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False,palette=None,is_demo=False,is_gt=False):
    if palette is None:
        palette = np.random.randint(0, 255, size=(3,3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3),dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[1].shape[1], 3),dtype=np.uint8)  
        color_area[result[0] == 1] = [0, 255, 0] # 路面分割部分设为绿色
        color_area[result[1] == 1] = [255, 0, 0] # 车道线分割部分设为红色
        color_seg = color_area
    
    # convert to BGR
    color_seg = color_seg[..., ::-1] # RGB2BGR
    color_mask = np.mean(color_seg, 2) # color_mask.shape->(720,1280),三个通道的平均值。
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(f"{save_dir}/batch_{epoch}_{index}_da_segresult.png",img)
            else:
                cv2.imwrite(f"{save_dir}/batch_{epoch}_{index}_ll_segresult.png",img)
        else:
            if not is_ll:
                cv2.imwrite(f"{save_dir}/batch_{epoch}_{index}_da_seg_gt.png",img)    
            else:
                cv2.imwrite(f"{save_dir}/batch_{epoch}_{index}_ll_seg_gt.png",img)
    return img
    