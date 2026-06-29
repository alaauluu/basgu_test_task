from clearml import Task
from ultralytics import YOLO

task = Task.init(
    project_name="yolo11_glory",
    task_name="my_yolo11s_train_colab_gpu"
)

model_variant = "yolo11s.pt"
task.set_parameter("model_variant", model_variant)
model = YOLO(model_variant)

args = dict(
    data="/content/glory_dataset/glory_dataset/glory_dataset.yaml",
    epochs=75,
    imgsz=896,
    batch=16,
    device="0",
    workers=4,
    amp=True,
    lr0=0.005,
    patience=15,
    seed=42,
    cls_pw=0.25,
    mosaic=1.0,
    close_mosaic=15,
    degrees=45.0,
    flipud=0.3,
    fliplr=0.5,
    scale=0.75,
    translate=0.1,
    perspective=0.0005,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.5,
    mixup=0.0,
    copy_paste=0.0
)

task.connect(args)
results = model.train(**args)