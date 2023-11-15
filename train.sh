# yolo train model=cfgs_thyroid/yolov8l-seg.yaml data=cfgs_thyroid/thyroid-seg.yaml \
#     mixup=0.5 batch=64 single_cls=True \
#     save_dir=data_thyroid_yl/ultralytics_run/seg/v8l_mixup05_single device=0,1,2,3

# yolo train model=cfgs_thyroid/yolov8n-seg.yaml data=cfgs_thyroid/thyroid-seg.yaml \
#     mixup=0.5 batch=64 single_cls=True \
#     save_dir=data_thyroid_yl/ultralytics_run/seg/v8n_mixup05_single device=1

# yolo train model=cfgs_thyroid/yolov8s-seg.yaml data=cfgs_thyroid/thyroid-seg.yaml \
#     mixup=0.5 batch=64 single_cls=True \
#     save_dir=data_thyroid_yl/ultralytics_run/seg/v8s_mixup05_single device=1

# yolo train model=cfgs_thyroid/yolov8m-seg.yaml data=cfgs_thyroid/thyroid-seg.yaml \
#     mixup=0.5 batch=64 single_cls=True \
#     save_dir=data_thyroid_yl/ultralytics_run/seg/v8m_mixup05_single device=0,1

yolo train model=cfgs_thyroid/yolov8x-seg.yaml data=cfgs_thyroid/thyroid-seg.yaml \
    mixup=0.5 batch=64 single_cls=True \
    save_dir=data_thyroid_yl/ultralytics_run/seg/v8x_mixup05_single device=0,1