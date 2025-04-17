import os
import glob
import time
from tqdm import tqdm
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
from ..utils import letterbox_for_img, clean_str

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']


class LoadImages:
    def __init__(self, path, img_size=640):
        p = str(Path(path)) # os-agnostic
        p = os.path.abspath(p)
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception(f'ERROR {p} does not exist.')
        
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv # numbers of files
        self.vedio_flag = [False] * ni + [True] * nv
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. Supported formats are:\n'\
                             'images:{img_formats}\nvideos:{vid_formats}'
    
    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total frames

    def __len__(self):
        return self.nf # number of files   

    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count] 

        if self.vedio_flag[self.count]:
            # Read Video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
            h0, w0 = img0.shape[:2]
            
            self.frame += 1
            print(f'\n video {self.count+1}/{self.nf} ({self.frame}/{self.nframes}) {path}:',end='')
        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) # BGR    
            assert img0 is not None, f'Image Not Found {path}'
            print(f'image {self.count}/{self.nf} {path}: \n', end='')
            h0, w0 = img0.shape[:2]
        
        # Padding resize
        img, ratio, pad = letterbox_for_img(img0, new_shape=self.img_size,auto=True)
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # Convert
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, shapes
    
    
class LoadStreams:
    def __init__(self, sources='streams.txt', img_size=640, auto=True):
        self.mode = 'stream'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]  
        
        n = len(sources)
        # 图像， 帧率， 帧数， 线程
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]
        self.auto = auto
        for i, s in enumerate(sources):
            print(f"{i+1}/{n}: {s}...", end='')
            s = eval(s) if s.isnumeric() else s
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0 # 30帧
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf') # infinite stream fallback

            _,self.imgs[i] = cap.read()
            self.threads[i] = Thread(target=self.update(), args=([i, cap]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print(' ') # newline

        # check for common shapes
        s = np.stack([letterbox_for_img(x, self.img_size, auto=self.auto)[0].shape for x in self.imgs], 0) # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, i: int, cap: cv2.VideoCapture):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()            
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.img[i] * 0
            time.sleep(1 / self.fps[i]) # wait time

    def __iter__(self):
        self.count = -1
        return self
    
    def __next__(self):
        self.count += 1
        if not all(x.isalive() for x in self.threads) or cv2.waitKey(1) == ord('q'): #q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        
        # Letterbox
        img0 = self.imgs.copy()
        h0, w0 = img0[0].shape[:2]
        img, _, pad = letterbox_for_img(img0[0], self.img_size, auto=self.rect and self.auto)

        #Stack
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # Convert
        img = np.ascontiguousarray(img)

        return self.sources, img, img0[0], None, shapes

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
        


