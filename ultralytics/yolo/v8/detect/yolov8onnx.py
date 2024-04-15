import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../../../')
sys.path.append('../../../../')
from ultralytics import YOLO

model=YOLO("/home//data/code/ObjectDetect/ultralytics_bccd/ultralytics/yolo/v8/detect/bccd/0319/weights/best.pt")

success=model.export(format="onnx", imgsz = (1, 3, 480, 480), half=False) #当dynamic为False时，input shape会默认使用训练的imgz
#(1, 3, 480, 480), 3表示通道数量，如果是灰度图像需要为1， 480是图片大小根据需要修改