from ultralytics import YOLO

model = YOLO('train/weights/best.pt')

model.export(format="engine")