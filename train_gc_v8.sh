#yolo train cfg=gc_cfgs/gc_default.yaml model=yolov8x.yaml data=gc_cfgs/gc5.2.yaml mixup=0.5 batch=64 save_dir=data_gc/ultralytics_run/v8x_data52_mixup05_gc device=2,3 nc=1 pseudo_empty=gc
#yolo train cfg=gc_cfgs/gc_default.yaml model=yolov8x.yaml data=gc_cfgs/gc5.2.yaml mixup=0.5 batch=64 save_dir=data_gc/ultralytics_run/v8x_data52_mixup05_gc_pesudo device=2,3 nc=1 pseudo_empty=gc_pseudo
yolo train cfg=gc_cfgs/gc_default.yaml model=yolov8x.yaml data=gc_cfgs/gc5.2.yaml mixup=0.5 batch=64 save_dir=data_gc/ultralytics_run/v8x_data52_mixup05_gc_empty device=2,3 nc=1 pseudo_empty=gc_empty
yolo train cfg=gc_cfgs/gc_default.yaml model=yolov8x.yaml data=gc_cfgs/gc5.2.yaml mixup=0.5 batch=64 save_dir=data_gc/ultralytics_run/v8x_data52_mixup05_gc_pseudo_empty device=2,3 nc=1 pseudo_empty=gc_pseudo_empty