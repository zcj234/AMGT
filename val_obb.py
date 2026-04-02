from ultralytics import YOLO

# Load a model
# 加载YOLO模型的配置文件，并加载预训练权重文件
model = YOLO("cfg/models/11/yolo11-ms-obb.yaml").load("runs/obb/E-YOLOv11s_obb_20k/weights/best.pt")   

# Validate the model
metrics = model.val(data="cfg/datasets/DOTAv1.yaml")  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list containing mAP50-95(B) for each category