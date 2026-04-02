from ultralytics import YOLO
 
if __name__ == '__main__':
    # 加载YOLO模型的配置文件，并加载预训练权重文件
    model = YOLO("cfg/models/v8/yolov8-ms-obb.yaml").load("runs/obb/train18/weights/best.pt")  

    # # 视频路径
    # video_path = r'E:\Desktop\pythoncode\MSF-VIT\visual-data\original-videos/DJI_0003.mov'

    # 图片路径
    image_path = r'/home/mipc/Desktop/image_fusion_for_test/DJI_0037_0038_fusion'
    
    # 使用dota8.yaml数据集进行训练，训练10个epoch，并将图像大小设置为640像素
    results = model.predict(source=image_path, save=True, show=False)
