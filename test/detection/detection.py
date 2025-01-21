# from model.box_detect.rmdet_trt import *
import cv2
# import torch
# import numpy as np
import os
import onnx
from ultralytics import YOLO

def box_detect(model,image_path, conf_threshold=0.65, iou_threshold=0.6, img_size=1280):
    """
    使用预训练模型检测给定图像中的边框。
    参数:
        image_path (str): 输入图像的路径。
        conf_threshold (float): 置信度阈值。
        iou_threshold (float): IOU 阈值。
        img_size (int): 输入图像大小。

    返回:
        tuple: (processed_image, detections)
            - processed_image: 绘制了边框的图像。
            - detections: 包含 (x1, y1, x2, y2, confidence, class_id, class_name) 的元组列表。
    """    
    try:
        result = model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=img_size,
            half=False,
            device='cpu',
            save=False,
        )

        if isinstance(result, list):
            result = result[0]

        boxes_data = result.boxes.data.cpu().numpy()  # 转换为 NumPy 数组
        class_ids = result.boxes.cls.cpu().numpy()  # 提取类别 ID
        confidences = result.boxes.conf.cpu().numpy()  # 提取置信度

        img = result.orig_img
        class_names = result.names  # 获取类别名称
        detection_results = []

        for i, box in enumerate(boxes_data):
            x1, y1, x2, y2 = map(int, box[:4])  # 获取坐标
            conf = confidences[i]  # 获取当前框的置信度
            cls_id = int(class_ids[i])  # 获取当前框的类别 ID
            
            class_name = class_names.get(cls_id, "未知类别")  # 获取类别名称

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f'{class_name},{conf:.2f}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 2)

            detection_results.append((x1, y1, x2, y2, conf, cls_id, class_name))

        # 调整图像大小
        height, width = img.shape[:2]
        img = cv2.resize(img, (width // 2, height // 2))

        return img, detection_results

    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None, []