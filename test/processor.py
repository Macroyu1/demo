import cv2
import open3d as o3d
import numpy as np
from lib.Logger import LoggerHandler
from typing import NamedTuple
from lib.pointcloud import Point_Cloud

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
    processor = Processor("image\\test.jpg", "image\\test.pcd")
    # 相机内参矩阵和畸变系数
    intrinsic_matrix = np.array([[3444.26, 0, 1161.62],
                               [0, 3447.16, 1053.98],
                               [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.0582, -0.6964, -0.003, 0.0175, 1.1209], dtype=np.float32)

    extrinsic_matrix = np.array([ [0.892025113436806   ,   0.102339548729281 ,  -4.23273148957745e-21],
   [-0.00960122221607244  ,  0.901616289223988  , -8.89413307636275e-21],
    [  0.0130152726953502   ,    -0.0160708018827668   ,   5.63090355422365e-20],
    [1216.78729765925    ,      1138.99834759068         ,                1]])
   
    
    

    points = processor.Point_Cloud_preprocessing("voxel",size=10)

    # processor.show(points)

    # 转换点云坐标到齐次坐标
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # 将相机坐标系的点投影到图像平面
    points_image = points_homogeneous @ extrinsic_matrix  # 乘以外参矩阵进行坐标转换

    points_image[:, :2] /= points_image[:, 2][:, None]  # 归一化，转换到像素坐标

    # 转换为像素坐标
    pixel_coords = points_image[:, :2].astype(int)

    image_shape = (2448, 2048)
    # 创建空的图像掩膜
    mask = np.zeros(image_shape, dtype=np.uint8)

    
    # 过滤掉超出图像范围的点
    valid_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_shape[1]) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_shape[0])
    valid_pixel_coords = pixel_coords[valid_mask]

    # 在掩膜中标记点
    mask[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = 255

    # 确保掩膜是二值的
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # 使用膨胀操作填充孤立点，连接相邻区域
    kernel = np.ones((15, 15), np.uint8)  # 定义膨胀的内核
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)  # 膨胀操作

    # 读取图像
    image = cv2.imread("image\\test.jpg")  # 替换为你的图像文件路径
    # 确保掩膜和图像大小一致
    dilated_mask_resized = cv2.resize(dilated_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 创建一个与图像大小相同的黑色背景
    result_image = np.zeros_like(image)

    # 将掩膜区域从原图像复制到结果图像中
    result_image[dilated_mask_resized == 255] = image[dilated_mask_resized == 255]

    # 调整显示图像大小
    result_image_resized = cv2.resize(result_image,(1224, 1024))
    image_resized = cv2.resize(image,(1224, 1024))
     # 显示最终图像
    cv2.imshow("result Image", result_image_resized)
    cv2.imshow("result1 Image", image_resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
