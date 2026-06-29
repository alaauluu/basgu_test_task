from ultralytics import YOLO

model = YOLO("train/weights/best.pt")
results = model.predict(source = "test_videos/inference1_without_yolo.MP4", stream = True, show = True, save=True)

for result in results:
    print(result.boxes)