import numpy as np
import pandas as pd
import cv2
import glob
import yaml
import os
import open3d as o3d
from scipy.spatial import KDTree
import onnx
from ultralytics import YOLO
from detection.detection import box_detect
def read_config(config_path = r"test\\calibration\\config.yaml") :
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在。")
    with open(config_path, 'r',encoding="utf-8") as file:
        config = yaml.safe_load(file)
        if config == None:
            print("无标定配置文件\n")
            config = {}
    return config

def highlight_multiple_spheres_in_pcd(pcd, center_points, radius, highlight_colors):
    # 加载点云
    points = np.asarray(pcd.points)
    # 如果点云没有颜色，则保留其原始颜色（灰色）
    if not pcd.has_colors():
        colors = np.tile(np.array([0.5, 0.5, 0.5]), (points.shape[0], 1))  # 默认灰色
        print("点云没有颜色属性，初始化为灰色...")
    else:
        colors = np.asarray(pcd.colors)  # 原始点云颜色

    # 构建 KDTree
    kdtree = KDTree(points)
    print("KDTree 构建完成！")

    # 遍历每个中心点，查找球面范围内的点
    for idx, center_point in enumerate(center_points):
        highlight_color = highlight_colors[idx % len(highlight_colors)]  # 循环使用高亮颜色
        indices = kdtree.query_ball_point(center_point, radius)
        print(f"中心点 {center_point} 在球面范围内找到 {len(indices)} 个点")
        for i in indices:
            colors[i] = highlight_color  # 将这些点设置为高亮颜色

    # 更新点云颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    points[:,1] = -points[:,1]
    points[:,2] = -points[:,2]

    # 显示点云
    print("显示高亮后的点云...")
    o3d.visualization.draw_geometries([pcd])


def find_3d_point_from_2d_with_kdtree(pcd, pixel_coords, intrinsic_matrix, extrinsic_matrix, dist_coeffs, tolerance=1e-3):
    # 加载点云
    points = np.asarray(pcd.points)

    # 将点云从世界坐标转换到相机坐标
    rotation_matrix = extrinsic_matrix[:3, :3]
    translation_vector = extrinsic_matrix[:3, 3]
    points_camera = (rotation_matrix @ points.T).T + translation_vector

    # 将 2D 像素点校正到图像坐标系
    undistorted_points = cv2.undistortPoints(np.array(pixel_coords, dtype=np.float32), intrinsic_matrix, dist_coeffs, None, intrinsic_matrix)
    undistorted_points = undistorted_points.reshape(-1, 2)  # 扁平化成 (N, 2) 的形式

    # 将点云从相机坐标系投影到图像平面
    points_image = (intrinsic_matrix @ points_camera.T).T
    points_image /= points_image[:, 2:].reshape(-1, 1)  # 归一化投影，保持形状一致

    # 计算投影点与校正后的像素点之间的距离
    closest_points = []
    for undistorted_point in undistorted_points:
        # 计算每个校正后的 2D 点与点云投影点之间的距离
        distances = np.linalg.norm(points_image[:, :2] - undistorted_point, axis=1)
        closest_index = np.argmin(distances)

        # 获取最近的 3D 点
        closest_3d_point = points[closest_index]
        if distances[closest_index] <= tolerance:
            closest_points.append(closest_3d_point)

    if len(closest_points) > 0:
        print(f"找到的最近 3D 点: {closest_points}")
        return closest_points
    else:
        print("未找到匹配的 3D 点")
        return None


def read_point_cloud(pcd_file, min_bound, max_bound):
    # 加载点云
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    # 筛选位于 ROI 范围内的点
    mask = (
        (points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0]) &
        (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1]) &
        (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2])
    )
    roi_points = points[mask]
    roi_colors = np.asarray(pcd.colors)[mask] if pcd.has_colors() else None

    # 创建新的点云
    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    if roi_colors is not None:
        roi_pcd.colors = o3d.utility.Vector3dVector(roi_colors)

    return roi_pcd


def main():
    config = read_config()
    image_point = np.array([1121,405])
    # 示例调用
    pcd_file = "test\image\\test.pcd"  # 替换为实际点云文件路径

    min_bound = (-1800, -1900, 3000)  # 替换为 ROI 的最小边界
    max_bound = (1800, 1900, 15000)    # 替换为 ROI 的最大边界

    point_cloud = read_point_cloud(pcd_file, min_bound, max_bound)

    pixel_coords = (1243	,1258)  # 替换为目标 2D 像素坐标 (u, v)
    
    # Load model
    model = YOLO("test\detection\\best.onnx", task='detect')
    processed_img,detect_result = box_detect(model,"test\image\\test.jpg")
    points = []
    
    # 提取检测框的角点并计算对应的3D点
    for result in detect_result:
        x1, y1, x2, y2, conf, cls_id, class_name = result

        # 计算四个角点
        top_left = np.array([x1, y1])  # 左上角
        top_right = np.array([x2, y1])  # 右上角
        bottom_left = np.array([x1, y2])  # 左下角
        bottom_right = np.array([x2, y2])  # 右下角

        # 将四个角点添加到列表中
        corners = np.array([top_left, top_right, bottom_left, bottom_right])

        # 记录每个检测框的四个角点
        points.append(corners)

    intrinsic_matrix = np.array(config['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(config['dist_coeffs'], dtype=np.float32)
    extrinsic_matrix = np.array(config['RT_2D_3D'], dtype=np.float32)
    tolerance = 100  # 替换为所需的容差

    closest_point = find_3d_point_from_2d_with_kdtree(point_cloud, np.array(points, dtype=np.float32).reshape(-1, 2), intrinsic_matrix, extrinsic_matrix, dist_coeffs, tolerance)
    
    print(closest_point)

    radius = 10  # 球面半径
    highlight_colors = [[1, 0, 0], [0, 1, 0]]  # 红色和绿色

    # 高亮多个点
    highlight_multiple_spheres_in_pcd(point_cloud, closest_point, radius, highlight_colors)
if __name__ == "__main__":
    main()