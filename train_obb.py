from ultralytics import YOLO
 
if __name__ == '__main__':
    # 加载YOLO模型的配置文件，并加载预训练权重文件
    # model = YOLO("cfg/models/11/yolo11-ms-obb.yaml").load("yolo11s-obb.pt")
    model = YOLO("runs/obb/train2/weights/last.pt")  
    
    # 使用dota8.yaml数据集进行训练，训练10个epoch，并将图像大小设置为640像素
    results = model.train(data="cfg/datasets/DOTAv1.yaml", epochs=150, device=0, workers=8, batch=16, amp=False, resume=True)  # 默认是16
