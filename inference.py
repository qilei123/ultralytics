from ultralytics import YOLO #pip install ultralytics
import cv2,glob
from geo2d.geometry import Polygon #pip install git+https://github.com/qilei123/Geo2D
import pickle
import numpy as np
class ThyroidSegPredictor():
    def __init__(self,model_dir,
                 conf = 0.25,
                 agnostic_nms = False,
                 iou_nms = 0.7,
                 device = 0):
        """用于检测甲状腺结节的AI模型

        Args:
            model_dir (str): 模型路径
            conf (float, optional): 目标置信度阈值. Defaults to 0.25.
            agnostic_nms (bool, optional): 是否返回一种类别，若是，表示只检测甲状腺结节，若非，表示可检测非恶性和恶性甲状腺结节. Defaults to False.
            iou_nms (float, optional): 结果进行nms时的iou阈值. Defaults to 0.7.
            device (int, optional): 模型使用到的显卡ID. Defaults to 0.
        """
        self.model = YOLO(model_dir) 
        self.conf = conf 
        self.agnostic_nms = agnostic_nms 
        self.iou_nms = iou_nms
        self.model.cuda(device)
        
    def predict(self,img):
        """模型处理由opencv读取的单张图像数据

        Args:
            img (cv2 mat): 单张图像数据

        Returns:
            result (dict): {'boxes':[], 位置信息 xywh
                  'segmentations':[], 分割位置信息 x y x y ...
                  'diameters':[], 直径长度（像素）float
                  'labels':[], 类别标签，若agnostic_nms为True,输出只有0代表甲状腺结节，若agnostic_nms为False,0代表良性结节，1代表恶性结节
                  'confs':[]} 检测出的每个结节的置信度
        """
        result = {'boxes':[],
                  'segmentations':[],
                  'diameters':[],
                  'labels':[],
                  'confs':[]}
        result_yolo =  self.model.predict(img,
                                          conf = self.conf,
                                          iou = self.iou_nms,
                                          agnostic_nms = self.agnostic_nms or True,
                                          verbose=False)[0]
        
        boxes = result_yolo.boxes.cpu().numpy()
        masks = result_yolo.masks.xy
        
        for box,mask,prob,conf in zip(boxes.xywh,
                                    masks,
                                    boxes.cls,
                                    boxes.conf):
            result['boxes'].append(box)
            result['segmentations'].append(mask.tolist())
            result['diameters'].append(self.diameter(mask.tolist()))
            if self.agnostic_nms:
                result['labels'].append(0)
            else:
                result['labels'].append(prob)
            result['confs'].append(conf)
            
        
        return result
 
    def diameter(self,mask):
        
        polyp =[]
        for point in mask:
            polyp.append(tuple(i for i in point))
        polygon = Polygon((*polyp,))
        
        return polygon.diameter
 
def testTSP():
    model_dir = 'data_thyroid_yl/ultralytics_run/seg/v8x_mixup05/weights/best.pt'
    cfg_dir = ''
    PTS = ThyroidSegPredictor(model_dir,agnostic_nms=False)
    
    image_dir = 'data_thyroid_yl/images/test/3004_1.bmp'
    image_dir = 'data_thyroid_yl/images/test/3521_1.bmp'
    image = cv2.imread(image_dir)
    
    iter_time = 1
    for i in range(iter_time):
        prediction = PTS.predict(image)
        print(prediction)
    

def process_eval_videos():
    video_dir_list = glob.glob('data_gc/videos_test/xiehe2111_2205/*.mp4')
    
    test_img_dir = 'data_gc/gc_df_e100/gc_df_rd/crop_images/1/00000_2017-09-13_0006412323_NJ20170913-086_NJ20170913-086_8505.jpg'
    
    # crop_model_dir = '/data3/echen/yolov8/train2/capture_image/runs/detect/train/weights/best.pt'
    # crop_model = YOLO(crop_model_dir)
    # crop_model.to(0)
    # print(crop_model.predict(test_img_dir,verbose = False))
    
    gc_model_dir = 'data_gc/ultralytics_run/v8l_data51_mixup05/weights/best.pt'
    gc_model = YOLO(gc_model_dir)
    gc_model.to(0)
    print(gc_model.predict(test_img_dir,verbose = False))

def parse_gc_dataset():
    gc_data_path = 'data_gc/train_5.2.cache'
    cache_fptr = open(gc_data_path, "rb")
    cache_datas = pickle.load(cache_fptr)

    labels = cache_datas['labels']
    segments = cache_datas['segments']
    shapes = cache_datas['shapes']
    im_hw0 = cache_datas['img_hw0']
    im_hw = cache_datas['img_hw']
    ims = cache_datas['imgs']

    print(len(ims))
    prefix = 'gc_pseudo_empty'
    selected_options = {'gc':[0],'gc_pseudo':[0,3],'gc_empty':[0,1,2],'gc_pseudo_empty':[0,1,2,3]}
    selected_cats = selected_options[prefix]

    filted_labels = []
    filted_segments = []
    filted_shapes = []
    filted_im_hw0 = []
    filted_im_hw = []
    filted_ims = []

    for index_id, _labels in enumerate(labels):
        temp_labels=[]
        temp_segments=[]
        _labels = _labels.tolist()
        empty_cats = []
        for label_index,label in enumerate(_labels):
            if (label[0] in selected_cats) and (label[0] in [0,3]):
                temp_labels.append(label)
                temp_segments.append(segments[index_id][label_index])
            else:
                empty_cats.append(label[0])
        if len(temp_labels)>0:
            filted_segments.append(temp_segments)
            filted_labels.append(np.array(temp_labels))
            filted_shapes.append(shapes[index_id])
            filted_im_hw0.append(im_hw0[index_id])
            filted_im_hw.append(im_hw[index_id])
            filted_ims.append(ims[index_id])
        
        if len(temp_labels)==0:
            usable = False
            for cat in empty_cats:
                if cat in selected_cats:
                    usable = True
                    break
            if usable:
                filted_segments.append([np.empty([0,2])])
                filted_labels.append(np.array(temp_labels))
                filted_shapes.append(shapes[index_id])
                filted_im_hw0.append(im_hw0[index_id])
                filted_im_hw.append(im_hw[index_id])
                filted_ims.append(ims[index_id])

    print(len(filted_ims))

if __name__=='__main__':
    #testTSP()
    #process_eval_videos()
    parse_gc_dataset()
    