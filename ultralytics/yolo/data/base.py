# Ultralytics YOLO 🚀, GPL-3.0 license

import glob
import math
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from ..utils import NUM_THREADS, TQDM_BAR_FORMAT
from .utils import HELP_URL, IMG_FORMATS, LOCAL_RANK


class BaseDataset(Dataset):
    """Base Dataset.
    Args:
        img_path (str): image path.
        pipeline (dict): a dict of image transforms.
        label_path (str): label path, this can also be an ann_file or other custom label path.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        center_box=False,
        cache=False,
        augment=True,
        hyp=None,
        prefix='',
        rect=False,
        batch_size=None,
        stride=32,
        pad=0.5,
        single_cls=False,
    ):
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.hyp = hyp
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.im_files = self.get_img_files(self.img_path)#获取所有图片路径
        self.labels = self.get_labels(hyp) #读取每个image的label

        if self.single_cls: #不区分类别，当做同一个
            self.update_labels(include_class=[])

        self.ni = len(self.labels) #图片数量

        # rect stuff
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # cache stuff
        self.ims = [None] * self.ni
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache:
            self.cache_images(cache)

        # transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'{self.prefix}No images found'
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)"""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class:
                cls = self.labels[i]['cls']
                bboxes = self.labels[i]['bboxes']
                segments = self.labels[i]['segments']
                j = (cls == include_class_array).any(1)
                self.labels[i]['cls'] = cls[j]
                self.labels[i]['bboxes'] = bboxes[j]
                if segments:
                    self.labels[i]['segments'] = segments[j]
            if self.single_cls:
                self.labels[i]['cls'][:, 0] = 0

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i] #image, png_file, npy_file

        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                if self.hyp.single_channel:
                    im = self.clahe.apply(cv2.imread(f, 0))  # BGR
                    im = im[:,:, np.newaxis]
                else:
                    im = cv2.imread(f)  # BGR

                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')

            h0, w0 = im.shape[:2]  # orig hw
            
            r = self.imgsz / max(h0, w0)  # ratio

            # if r != 1:  # if sizes are not equal 保持长宽比resize到imgsz大小
            #     interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA #
            #     im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)#
                # if len(im.shape) == 2:
                #     im = im[:,:,np.newaxis]
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA #
            im = cv2.resize(im, (self.imgsz // 3, self.imgsz), interpolation=interp)            
            
            if len(im.shape) == 2:
                im = im[:,:,np.newaxis]
                   
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images(self, cache):
        # cache images to memory or disk
        gb = 0  # Gigabytes of cached images
        self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni
        fcn = self.cache_images_to_disk if cache == 'disk' else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = tqdm(enumerate(results), total=self.ni, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == 'disk':
                    gb += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.ims[i].nbytes
                pbar.desc = f'{self.prefix}Caching images ({gb / 1E9:.1f}GB {cache})'
            pbar.close()

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def set_rectangle(self):
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # [0,1,2,3,4...]/ batch_size， self.ni 总共的数据数量, 每个数据对应的batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop('shape') for x in self.labels])  # hw， 所有resize后的hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect] #按照长宽比重新排序

        # Set training image shapes
        shapes = [[1, 1]] * nb #batch数量，每个batch一个[1,1]
        for i in range(nb):
            ari = ar[bi == i]#选出第i个batch的所有数据
            mini, maxi = ari.min(), ari.max()#该batch中长宽比最大最小值
            if maxi < 1:#长宽比最大仍然小于1，比如最大长宽比是0.8，所有小于0.8的数据都变为0.8，使得长宽比不过于失衡
                shapes[i] = [maxi, 1]
            elif mini > 1:#最小的长宽比大于1，极大的长宽比向最小的长宽比对齐
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride #该batch的shape大小
        self.batch = bi  # batch index of image

    def __getitem__(self, index):


        return self.transforms(self.get_label_info(index))

    def get_label_info(self, index):
        label = self.labels[index].copy()
        
        # len_cls = len(label['cls'])
        # for ij in range(len_cls):
        #     rx, ry, rw, rh = (torch.rand(1)-0.5)/50, (torch.rand(1)-0.5)/100, (torch.rand(1)-0.5)/50, (torch.rand(1)-0.5)/100
        #     label['bboxes'][ij][0] += rx
        #     label['bboxes'][ij][1] += ry
        #     label['bboxes'][ij][2] += rw
        #     label['bboxes'][ij][3] += rh
        
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)

        label['ratio_pad'] = (#长宽各自的放缩比例
            label['resized_shape'][0] / label['ori_shape'][0],
            label['resized_shape'][1] / label['ori_shape'][1],
        )  # for evaluation
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        label = self.update_labels_info(label)
        return label

    def __len__(self):
        return len(self.labels)

    def update_labels_info(self, label):
        """custom your label format here"""
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # training transforms
                return Compose([])
            else:
                # val transforms
                return Compose([])
        """
        raise NotImplementedError

    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        raise NotImplementedError
