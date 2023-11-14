from ultralytics import YOLO

model = YOLO('data_thyroid_yl/ultralytics_run/seg/v8l_mixup05/weights/best.pt') 

image_dir = 'data_thyroid_yl/images/test/3005_2.bmp'

results = model(image_dir)  

print(results)