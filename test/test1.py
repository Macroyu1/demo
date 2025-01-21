import pandas as pd
import open3d as o3d
import numpy as np
import cv2
from lib.pointcloud import Point_Cloud
from lib.Logger import LoggerHandler
from typing import NamedTuple
from detection.detection import box_detect
import onnx
from ultralytics import YOLO


class Box(NamedTuple):
    img: np.ndarray  
    pcd: o3d.geometry.PointCloud


class Processor(LoggerHandler,Point_Cloud):
    def __init__(self, img_file, pcd_file):
        LoggerHandler.__init__(self, "logger")
        Point_Cloud.__init__(self)
        self.img, self.pcd = self.read_img_pcd(img_file, pcd_file)

    def read_img_pcd(self, img_file, pcd_file) -> Box:
        # 读取点云与图像文件
        img = cv2.imread(img_file)
        pcd = o3d.io.read_point_cloud(pcd_file)
        if not pcd.has_points():
            self.error(f"点云文件 {pcd_file} 加载失败或为空！")
            raise
        if img is None or img.size == 0:
            self.error(f"图像文件 {img_file} 加载失败或为空！")
            raise 
        return Box(img, pcd)

    def Point_Cloud_preprocessing(self,flag = "uniform",size=1) -> o3d.geometry.PointCloud:

        self.ROI([[-900, 900, -900, 900, 0, 5000]])
        self.down_sample(flag,size)

        points = self.top_mask(100,10)   
        
        return points


def main():
    # Load model
    model = YOLO("test\detection\\best.onnx", task='detect')
    processed_img,detect_result = box_detect(model,"test\image\\test.jpg")
    points = []     
    cv2.imshow("img",processed_img)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()
