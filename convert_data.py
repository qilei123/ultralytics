from ultralytics.data.converter import convert_coco

convert_coco('data_thyroid/annotations', '/data3/qilei_chen/DATA/TN-SCUI2020/data_thyroid_yl', use_segments=True, use_keypoints=False)