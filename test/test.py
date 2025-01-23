import numpy as np
import cv2
import os
import yaml
import open3d as o3d
from scipy.spatial import KDTree
from ultralytics import YOLO
from detection.detection import box_detect


# 读取配置文件
def read_config(config_path=r"test\\calibration\\config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在。")
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
        if config is None:
            print("无标定配置文件\n")
            config = {}
    return config


# 生成类别对应的颜色
def generate_colors(num_classes):
    np.random.seed(42)  # 固定随机种子
    return np.random.rand(num_classes, 3)


# 在点云中高亮多个框
def highlight_boxes_in_pcd(pcd, boxes_3d, highlight_colors):
    points = np.asarray(pcd.points)
    if not pcd.has_colors():
        colors = np.tile(np.array([0.5, 0.5, 0.5]), (points.shape[0], 1))  # 默认灰色
        print("点云没有颜色属性，初始化为灰色...")
    else:
        colors = np.asarray(pcd.colors)  # 原始点云颜色

    kdtree = KDTree(points)
    print("KDTree 构建完成！")

    # 遍历每个检测框，找到边界范围内的点
    for idx, box in enumerate(boxes_3d):
        highlight_color = highlight_colors[idx % len(highlight_colors)]  # 循环使用颜色
        for i in range(len(box)):
            p1 = box[i]
            p2 = box[(i + 1) % len(box)]  # 获取下一点，形成闭合矩形
            line_points = interpolate_line(p1, p2, step=0.05)  # 生成线段上的插值点
            for point in line_points:
                indices = kdtree.query_ball_point(point, r=10)  # 搜索线段点附近的点

                for i in indices:
                    colors[i] = highlight_color

    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("显示高亮后的点云...")
    o3d.visualization.draw_geometries([pcd])


# 从2D像素点找到对应的3D点
def find_3d_points_from_2d(pcd, pixel_coords, intrinsic_matrix, extrinsic_matrix, dist_coeffs, tolerance=1e-3):
    points = np.asarray(pcd.points)

    rotation_matrix = extrinsic_matrix[:3, :3]
    translation_vector = extrinsic_matrix[:3, 3]
    points_camera = (rotation_matrix @ points.T).T + translation_vector

    undistorted_points = cv2.undistortPoints(
        np.array(pixel_coords, dtype=np.float32),
        intrinsic_matrix, dist_coeffs, None, intrinsic_matrix
    ).reshape(-1, 2)

    points_image = (intrinsic_matrix @ points_camera.T).T
    points_image /= points_image[:, 2:].reshape(-1, 1)

    box_points_3d = []
    for undistorted_point in undistorted_points:
        distances = np.linalg.norm(points_image[:, :2] - undistorted_point, axis=1)
        closest_index = np.argmin(distances)
        if distances[closest_index] <= tolerance:
            box_points_3d.append(points[closest_index])

    return box_points_3d


# 生成两点之间的插值点
def interpolate_line(p1, p2, step=0.05):
    num_points = int(np.linalg.norm(p2 - p1) / step)
    return np.linspace(p1, p2, num=num_points)


# 读取并筛选点云
def read_point_cloud(pcd_file, min_bound, max_bound):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    mask = (
        (points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0]) &
        (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1]) &
        (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2])
    )
    roi_points = points[mask]
    roi_colors = np.asarray(pcd.colors)[mask] if pcd.has_colors() else None

    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    if roi_colors is not None:
        roi_pcd.colors = o3d.utility.Vector3dVector(roi_colors)

    return roi_pcd


# 主函数
def main():
    config = read_config()

    # 示例文件路径和参数
    pcd_file = "test/image/test.pcd"
    min_bound = (-1800, -1900, 3000)
    max_bound = (1800, 1900, 15000)
    point_cloud = read_point_cloud(pcd_file, min_bound, max_bound)

    # 加载YOLO模型并检测
    model = YOLO("test/detection/best.onnx", task='detect')
    processed_img, detect_result = box_detect(model, "test/image/test.jpg")

    # 摄像机参数
    intrinsic_matrix = np.array([[2111.84955910501, 0, 1290.97644079674],
                                  [0, 2104.38342695474, 909.415077675728],
                                  [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.0582, -0.6964, -0.003, 0.0175, 1.1209], dtype=np.float32)
    extrinsic_matrix = np.array([
        [0.992949194608488, -0.000183697262421475, -0.00752634335280559, -8.37491630316323e-05],
        [-2.57858190291198e-05, 0.992861890735398, -0.00529848127847806, 0.000101933510194447],
        [-0.00139121994487156, -0.000959970207155758, 0.99292337549335, -0.0124224916904693],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    tolerance = 100

    # 为每个检测框投影角点并高亮点云
    num_classes = 10  # 假设有10个类别
    highlight_colors = generate_colors(num_classes)
    all_boxes_3d = []

    for result in detect_result:
        x1, y1, x2, y2, conf, cls_id, class_name = result
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        box_3d = find_3d_points_from_2d(
            point_cloud, corners, intrinsic_matrix, extrinsic_matrix, dist_coeffs, tolerance
        )
        if len(box_3d) == 4:
            all_boxes_3d.append(box_3d)

    if all_boxes_3d:
        highlight_boxes_in_pcd(point_cloud, all_boxes_3d, highlight_colors)
    else:
        print("未找到匹配的 3D 检测框！")


if __name__ == "__main__":
    main()
