yolo train model=cfgs_thyroid/yolov8l-seg.yaml data=cfgs_thyroid/thyroid-seg.yaml \
    mixup=0.5 batch=64 single_cls=True \
    save_dir=data_thyroid_yl/ultralytics_run/seg/v8l_mixup05_single device=0,1,2,3