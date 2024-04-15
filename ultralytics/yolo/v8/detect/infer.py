import numpy as np
import cv2
from pathlib import Path
import glob
import os
import torch
import time
import onnxruntime
import torchvision
import shutil
import sys
sys.path.append('../../../')

# from ultralytics.yolo.engine.model import YOLO
# import matplotlib.pyplot as plt

def box_label(img, results, names, ans_dir, colors, txt_color=(255, 255, 255), args = None):

    lw = args['line_thickness']

    for d in reversed(results):
        box = d[:4]
        cls, conf = d[-1], d[-2]
        
        c = int(cls)  # integer class
        name =  names[c]
        
        color=colors(c, True)    
    
        label = f'{name} {conf:.2f}'
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)
    cv2.imwrite(ans_dir, img)


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ( '3DDB86', '1A9334', '00D4BB',
                '2C99A8', 'FF37C7', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    


class LoadImages:
    def __init__(self, args, path, transforms=None, single_channel=False):
        self.files = []
        self.files.extend(sorted(glob.glob(os.path.join(path, '*.{}'.format(args['imgType'])))))  # dir
        self.files = self.files
        self.nf = len(self.files) # number of files
        self.single_channel = single_channel
        self.new_shape=[args['imgsz'], args['imgsz']]

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        self.count += 1
        if self.single_channel:
            im0 = cv2.imread(path, 0) 
        else:
            im0 = cv2.imread(path)
        s = f'image {self.count}/{self.nf} {path}: '#image 1/511 /home/lead/data/Conv_light/allimgs/xxx.png:   输出提示
        print(s)
        im = cv2.resize(im0, (self.new_shape[1], self.new_shape[0]))
        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        return path, im, im0, None, s

    def __len__(self):
        return self.nf  # number of files

def preprocess(img):
    img = torch.from_numpy(img).to('cpu')
    img =  img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    return img

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
       y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """

    将所处理规则的图像上的box映射到原始图像

    Returns:
      boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    # gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    gain1, gain2 = img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]  # gain  = old / new

    boxes[..., 0] /= gain2
    boxes[..., 2] /= gain2
    boxes[..., 1] /= gain1
    boxes[..., 3] /= gain1
    clip_boxes(boxes, img0_shape)#可能越界，上界和下界
    return boxes

def clip_boxes(boxes, shape):

    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,
        classes=None, agnostic=True, multi_label=False, labels=(),
        max_det=300, nc=0, max_time_img=0.05, max_nms=30000, max_wh=7680,
):

    # Checks
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    #pre: [b, 4+cls_num, box_num]
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)# 5:xywh+conf
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # print(scores, iou_thres)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output


def postprocess(preds, img, orig_img):

    preds = non_max_suppression(preds,
                                    args['conf'],
                                    args['iou'],
                                    agnostic=args['agnostic_nms'],
                                    max_det=args['max_det'],
                                    classes=args['classes'])

    for i, pred in enumerate(preds):
        orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
        shape = orig_img.shape
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], shape).round()
    return pred


def saveYoloTxt(args, xywhn, results, path):

    txtname = os.path.basename(path).replace(args['imgType'], 'txt')
    list_result = results.detach().cpu().numpy().tolist()
    #存储result， [xmin, ymin, xmax, ymax, conf, label]
    ans_str = []
    for it in list_result:
        str_it = [str(tt) for tt in it]
        ans_str.append(str_it)
    
    with open(os.path.join(args['txt_dir'], txtname), 'w') as f:
        for ict in range(xywhn.shape[0]):
            box_tmp = xywhn[ict, :]
            box_str = [str(itt.item()) for itt in box_tmp]
            label = str(int(results[ict, -1].item()))
            box_str.insert(0, label)
            f.write(' '.join(box_str) + '\n')
         

def inference(args):

    
    names = {0: 'WBC', 1: 'RBC', 2:'Platelets'}  #根据需要设置名字
    colors = Colors()  
    
    
    dataset = LoadImages(args, path=args['source'],
                transforms=None, single_channel=args['single_channel'])

    cuda = torch.cuda.is_available() and args['device'] != 'cpu'
    model = args['model']#路径
    

    #加载model，创建session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(model, providers=providers)


    output_names = [x.name for x in session.get_outputs()]    
    for batch in dataset:
        
        path, im, im0s, vid_cap, s = batch

        im = preprocess(im)

        txtP = path.replace('images', 'labels').replace(args['imgType'], 'txt')
        txtdata= []
        with open(txtP, 'r') as f:
            lines = f.readlines()
            for line in lines:
                txtdata.append(line[0])
        
        im = im.unsqueeze(0)
        im = im.cpu().numpy()  # torch to numpy

        y = session.run(output_names, {session.get_inputs()[0].name: im})
        
        preds = torch.from_numpy(y[0]) if len(y) == 1 else [torch.from_numpy(x) for x in y]

        results = postprocess(preds, im, im0s)

        ori_shape = torch.as_tensor(im0s.shape[0:2])
        
        box_ans = xyxy2xywh(results[:,:4]) 
        xywhn = box_ans/ori_shape[[1,0,1,0]]

        # 将结果保存为yolo版本的txt格式
        saveYoloTxt(args, xywhn, results, path)
        # if box_num > 0 and len(txtdata) == 0: 
        #     # # 如果不保存图片，下面的代码可以注释
        ans_dir = args['save_dir'] + '/' + os.path.basename(path)
        box_label(im0s, results, names,ans_dir, colors, args=args)


if __name__ == '__main__':
    args = dict()
    itrea = True
    args['model']='/home//data/code/ObjectDetect/ultralytics_bccd/ultralytics/yolo/v8/detect/bccd/0319/weights/best.onnx'  #最佳权重的onnx
    args['imgsz']=480 #图片尺寸
    args['save']=True #是否保存数据
    args['single_channel']=False # 是否单通道灰度图
    args['device']='cpu' #使用cpu/gpu
    args['conf']=0.3 #置信度
    args['imgType']='jpg' #图片的类型
    
    args['iou']=0.01
    args['max_det']=300
    args['line_thickness']=1
    args['agnostic_nms']=True #不同类之间进行nms
    args['classes']=None
    

    args['source']='/home//data/dataset/BCCD/images/val'  # 需要验证图片的路径
    args['dst']='/home//data/dataset/BCCD/images/pred'   # 结果保存路径
    
    save_dir = args['dst']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    args['save_dir']=save_dir
    args['txt_dir']=save_dir

    gen = inference(args)