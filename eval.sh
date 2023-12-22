yolo cfg=gc_cfgs/gc_default.yaml mode=val \
    model=data_gc/ultralytics_run/v8m_data51_mixup05/weights/best.pt \
    data=gc_cfgs/gc5.1.yaml batch=64 imgsz=640 \
    save_json=True show=True